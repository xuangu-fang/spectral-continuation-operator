"""Matched WaveBench test of frequency-efficient complex-field representations.

The public WaveBench files align the same wavespeed realization across four
time-harmonic frequencies.  We invert a variable, unordered subset of those
complex responses back to the wavespeed field.  All representations use the
same set operator and training schedule; only the response coordinates change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .wave_set_encoder import ScalarFieldDecoder, WaveSetEncoder

FREQUENCIES = (10, 15, 20, 40)
REPRESENTATIONS = (
    "raw",
    "amplitude",
    "amplitude_soft_phase_gradient",
    "amplitude_windowed_coherence",
    "amplitude_phase_windowed_coherence",
    "amplitude_phase_wave_packets",
    "amplitude_phase_wave_packet_gram",
    "amplitude_phase_multiscale_packet_gram",
    "amplitude_edge_connection",
    "amplitude_transport_connection",
)

HEADER_DTYPE = np.dtype(
    [
        ("version", "<u2"),
        ("num_fields", "<u2"),
        ("page_size", "<u4"),
        ("num_samples", "<u8"),
        ("alloc_table_ptr", "<u8"),
    ],
    align=True,
)
FIELD_DTYPE = np.dtype(
    [("type_id", "<u1"), ("name", ("<u1", 16)), ("arguments", ("<u1", (1024,)))],
    align=True,
)
METADATA_DTYPE = np.dtype([("", "<u8"), ("", "<u8")], align=True)
GRID = 128
INPUT_SHAPE = (1, GRID, GRID)
TARGET_SHAPE = (2, GRID, GRID)


class BetonReader:
    """Dependency-free reader for the fixed WaveBench acoustic FFCV layout."""

    def __init__(self, path: Path):
        self.path = path
        header = np.fromfile(path, dtype=HEADER_DTYPE, count=1)
        if len(header) != 1 or int(header[0]["version"]) != 2:
            raise ValueError(f"invalid FFCV v2 header: {path}")
        if int(header[0]["num_fields"]) != 2:
            raise ValueError(f"expected two WaveBench fields: {path}")
        fields = np.fromfile(path, dtype=FIELD_DTYPE, count=2, offset=HEADER_DTYPE.itemsize)
        names = tuple(bytes(field["name"]).split(b"\0", 1)[0].decode("ascii") for field in fields)
        if names != ("input", "target"):
            raise ValueError(f"unexpected fields {names} in {path}")
        self.num_samples = int(header[0]["num_samples"])
        self._raw = np.memmap(path, mode="r", dtype=np.uint8)
        self._metadata = np.memmap(
            path,
            mode="r",
            dtype=METADATA_DTYPE,
            offset=HEADER_DTYPE.itemsize + fields.nbytes,
            shape=(self.num_samples,),
        )

    def read_many(
        self, indices: np.ndarray, *, include_medium: bool
    ) -> tuple[np.ndarray | None, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        medium = (
            np.empty((len(indices), *INPUT_SHAPE), dtype=np.float32) if include_medium else None
        )
        response = np.empty((len(indices), *TARGET_SHAPE), dtype=np.float32)
        for output, index in enumerate(indices):
            input_ptr, target_ptr = (int(value) for value in self._metadata[int(index)])
            if medium is not None:
                medium[output] = np.ndarray(
                    INPUT_SHAPE, dtype="<f4", buffer=self._raw, offset=input_ptr
                )
            response[output] = np.ndarray(
                TARGET_SHAPE, dtype="<f4", buffer=self._raw, offset=target_ptr
            )
        return medium, response


@dataclass(frozen=True)
class FrozenSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def make_split(seed: int, train_count: int, validation_count: int, test_count: int) -> FrozenSplit:
    if train_count + validation_count + test_count > 50_000:
        raise ValueError("requested split exceeds WaveBench")
    permutation = np.random.default_rng(seed).permutation(50_000)
    test = np.sort(permutation[:test_count])
    remaining = permutation[test_count:]
    train = np.sort(remaining[:train_count])
    validation = np.sort(remaining[train_count : train_count + validation_count])
    return FrozenSplit(train=train, validation=validation, test=test)


def ids_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _view_rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value.square().sum(2).mean(dim=(-2, -1), keepdim=True) + 1e-12)


def _shifted_products(
    value: torch.Tensor, dy: int, dx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gauge-invariant products and paired endpoint energies."""

    height, width = value.shape[-2:]
    base_y = slice(max(0, -dy), min(height, height - dy))
    base_x = slice(max(0, -dx), min(width, width - dx))
    shifted_y = slice(max(0, -dy) + dy, min(height, height - dy) + dy)
    shifted_x = slice(max(0, -dx) + dx, min(width, width - dx) + dx)
    product = torch.zeros_like(value)
    first_energy = torch.zeros_like(value.real)
    second_energy = torch.zeros_like(value.real)
    base = value[:, base_y, base_x]
    shifted = value[:, shifted_y, shifted_x]
    product[:, base_y, base_x] = shifted * base.conj()
    first_energy[:, base_y, base_x] = shifted.abs().square()
    second_energy[:, base_y, base_x] = base.abs().square()
    return product, first_energy, second_energy


def _windowed_coherence_coordinates(
    complex_value: torch.Tensor,
    log_amplitude: torch.Tensor,
    reference: torch.Tensor,
    frequency_hz: tuple[int, ...],
    *,
    reference_speed: float = 2.5,
) -> torch.Tensor:
    """Amplitude plus wavelength-scaled local complex autocorrelations.

    Two radial lags and four undirected orientations yield eight complex
    coherence samples.  Unlike a hard top-K spectrum, this representation is
    continuous when two local branches exchange order.  The default reference
    speed is in the public WaveBench coordinate units and is not a target-field
    input.
    """

    if len(frequency_hz) != complex_value.shape[1]:
        raise ValueError("one physical frequency is required for every response view")
    spacing = 1.0 / (complex_value.shape[-1] - 1)
    output = []
    for view, frequency in enumerate(frequency_hz):
        wavelength_pixels = reference_speed / (float(frequency) * spacing)
        window = int(np.clip(round(1.5 * wavelength_pixels), 7, 47))
        if window % 2 == 0:
            window += 1 if window < 47 else -1
        radii = (max(1, round(0.25 * wavelength_pixels)), max(1, round(0.50 * wavelength_pixels)))
        offsets = []
        for radius in radii:
            diagonal = max(1, round(radius / np.sqrt(2.0)))
            offsets.extend(((0, radius), (radius, 0), (diagonal, diagonal), (diagonal, -diagonal)))

        packed = []
        for dy, dx in offsets:
            product, first_energy, second_energy = _shifted_products(
                complex_value[:, view], dy, dx
            )
            packed.extend((product.real, product.imag, first_energy, second_energy))
        smoothed = torch.nn.functional.avg_pool2d(
            torch.stack(packed, dim=1), window, stride=1, padding=window // 2
        )
        channels = [log_amplitude[:, view]]
        floor = 0.05 * reference[:, view].square()
        for index in range(len(offsets)):
            real = smoothed[:, 4 * index]
            imaginary = smoothed[:, 4 * index + 1]
            denominator = torch.sqrt(
                smoothed[:, 4 * index + 2] * smoothed[:, 4 * index + 3]
            )
            denominator = denominator + floor + 1e-8
            channels.extend((real / denominator, imaginary / denominator))
        output.append(torch.stack(channels, dim=1))
    return torch.stack(output, dim=1).float()


def _wave_packet_components(
    complex_value: torch.Tensor,
    frequency_hz: tuple[int, ...],
    *,
    reference_speed: float = 2.5,
    orientations: int = 8,
    radial_scales: tuple[float, ...] = (1.0,),
) -> list[torch.Tensor]:
    """Analyze every view with a fixed directional complex packet bank."""

    if len(frequency_hz) != complex_value.shape[1]:
        raise ValueError("one physical frequency is required for every response view")
    height, width = complex_value.shape[-2:]
    ky = torch.fft.fftfreq(height, device=complex_value.device)
    kx = torch.fft.fftfreq(width, device=complex_value.device)
    grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
    spacing = 1.0 / (width - 1)
    angles = torch.arange(orientations, device=complex_value.device) * (
        2.0 * torch.pi / orientations
    )
    output: list[torch.Tensor] = []
    for view, frequency in enumerate(frequency_hz):
        center_frequency = float(frequency) * spacing / reference_speed
        bandwidth = max(0.45 * center_frequency, 1.0 / width)
        scale = torch.as_tensor(radial_scales, device=complex_value.device)
        center_y = (center_frequency * scale[:, None] * torch.sin(angles)[None]).reshape(-1)
        center_x = (center_frequency * scale[:, None] * torch.cos(angles)[None]).reshape(-1)
        packet_bandwidth = (
            bandwidth * scale[:, None].expand(-1, orientations).reshape(-1)
        ).clamp_min(1.0 / width)
        filters = torch.exp(
            -0.5
            * (
                (grid_y[None] - center_y[:, None, None]).square()
                + (grid_x[None] - center_x[:, None, None]).square()
            )
            / packet_bandwidth[:, None, None] ** 2
        )
        spectrum = torch.fft.fft2(complex_value[:, view])
        components = torch.fft.ifft2(spectrum[:, None] * filters[None])
        output.append(components)
    return output


def _wave_packet_coordinates(
    complex_value: torch.Tensor,
    log_amplitude: torch.Tensor,
    reference: torch.Tensor,
    frequency_hz: tuple[int, ...],
    *,
    reference_speed: float = 2.5,
    orientations: int = 8,
) -> torch.Tensor:
    """Fixed directional complex wave packets with branch-wise phase advance."""

    component_views = _wave_packet_components(
        complex_value,
        frequency_hz,
        reference_speed=reference_speed,
        orientations=orientations,
    )
    spacing = 1.0 / (complex_value.shape[-1] - 1)
    angles = torch.arange(orientations, device=complex_value.device) * (
        2.0 * torch.pi / orientations
    )
    output = []
    for view, (frequency, components) in enumerate(zip(frequency_hz, component_views, strict=True)):
        center_frequency = float(frequency) * spacing / reference_speed
        wavelength_pixels = 1.0 / max(center_frequency, 1e-8)
        step = max(1, round(0.25 * wavelength_pixels))
        channels = [log_amplitude[:, view]]
        for orientation in range(orientations):
            dy = round(step * float(torch.sin(angles[orientation])))
            dx = round(step * float(torch.cos(angles[orientation])))
            product, _, _ = _shifted_products(components[:, orientation], dy, dx)
            branch_rms = torch.sqrt(
                components[:, orientation].abs().square().mean(dim=(-2, -1), keepdim=True)
                + 1e-12
            )
            branch_amplitude = torch.log1p(
                components[:, orientation].abs() / (branch_rms + 1e-8)
            )
            advance = product / (product.abs() + 0.05 * branch_rms.square() + 1e-8)
            channels.extend((branch_amplitude, advance.real, advance.imag))
        output.append(torch.stack(channels, dim=1))
    return torch.stack(output, dim=1).float()


def _wave_packet_gram_coordinates(
    complex_value: torch.Tensor,
    log_amplitude: torch.Tensor,
    frequency_hz: tuple[int, ...],
    *,
    reference_speed: float = 2.5,
    orientations: int = 8,
) -> torch.Tensor:
    """Encode packet energy and inter-branch phase modulo one global phase."""

    component_views = _wave_packet_components(
        complex_value,
        frequency_hz,
        reference_speed=reference_speed,
        orientations=orientations,
    )
    # Eight edges make a connected orientation graph while explicitly linking
    # counter-propagating packets, which are important for standing waves.
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7))
    output = []
    for view, components in enumerate(component_views):
        energy = components.abs().square()
        total_energy = energy.sum(dim=1, keepdim=True).clamp_min(1e-10)
        channels = [log_amplitude[:, view]]
        channels.extend((energy / total_energy).unbind(dim=1))
        for first, second in edges:
            cross = components[:, second] * components[:, first].conj()
            denominator = torch.sqrt(energy[:, first] * energy[:, second])
            cross = cross / (denominator + 0.01 * total_energy[:, 0] + 1e-8)
            channels.extend((cross.real, cross.imag))
        output.append(torch.stack(channels, dim=1))
    return torch.stack(output, dim=1).float()


def _multiscale_wave_packet_gram_coordinates(
    complex_value: torch.Tensor,
    log_amplitude: torch.Tensor,
    frequency_hz: tuple[int, ...],
    *,
    reference_speed: float = 2.5,
    orientations: int = 8,
    radial_scales: tuple[float, ...] = (0.75, 1.5),
) -> torch.Tensor:
    """Two-radius packet Gram with within-scale and cross-scale phase edges."""

    component_views = _wave_packet_components(
        complex_value,
        frequency_hz,
        reference_speed=reference_speed,
        orientations=orientations,
        radial_scales=radial_scales,
    )
    orientation_edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7))
    output = []
    for view, components in enumerate(component_views):
        energy = components.abs().square()
        total_energy = energy.sum(dim=1, keepdim=True).clamp_min(1e-10)
        channels = [log_amplitude[:, view]]
        channels.extend((energy / total_energy).unbind(dim=1))
        edges = []
        for scale_index in range(len(radial_scales)):
            offset = scale_index * orientations
            edges.extend((offset + first, offset + second) for first, second in orientation_edges)
        for scale_index in range(len(radial_scales) - 1):
            first_offset = scale_index * orientations
            second_offset = (scale_index + 1) * orientations
            edges.extend(
                (first_offset + orientation, second_offset + orientation)
                for orientation in range(orientations)
            )
        for first, second in edges:
            cross = components[:, second] * components[:, first].conj()
            denominator = torch.sqrt(energy[:, first] * energy[:, second])
            cross = cross / (denominator + 0.01 * total_energy[:, 0] + 1e-8)
            channels.extend((cross.real, cross.imag))
        output.append(torch.stack(channels, dim=1))
    return torch.stack(output, dim=1).float()


def response_coordinates(
    value: torch.Tensor,
    representation: str,
    frequency_hz: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Map ``[B,V,2,H,W]`` responses to matched real-valued coordinates."""

    if representation == "raw":
        return value
    complex_value = torch.complex(value[:, :, 0], value[:, :, 1])
    amplitude = torch.abs(complex_value)
    reference = _view_rms(value)
    log_amplitude = torch.log1p(amplitude / reference)
    if representation == "amplitude":
        return log_amplitude[:, :, None]
    if representation == "amplitude_windowed_coherence":
        if frequency_hz is None:
            raise ValueError("windowed coherence requires physical frequencies")
        return _windowed_coherence_coordinates(
            complex_value, log_amplitude, reference, frequency_hz
        )
    if representation == "amplitude_phase_windowed_coherence":
        if frequency_hz is None:
            raise ValueError("windowed coherence requires physical frequencies")
        coherence = _windowed_coherence_coordinates(
            complex_value, log_amplitude, reference, frequency_hz
        )
        floor = 0.05 * reference.square()
        horizontal = torch.ones_like(complex_value)
        vertical = torch.ones_like(complex_value)
        horizontal[:, :, :, 1:] = (
            complex_value[:, :, :, 1:] * complex_value[:, :, :, :-1].conj()
        )
        vertical[:, :, 1:, :] = (
            complex_value[:, :, 1:, :] * complex_value[:, :, :-1, :].conj()
        )
        horizontal = horizontal / (horizontal.abs() + floor + 1e-8)
        vertical = vertical / (vertical.abs() + floor + 1e-8)
        base = torch.stack(
            (
                log_amplitude,
                horizontal.real,
                horizontal.imag,
                vertical.real,
                vertical.imag,
            ),
            dim=2,
        )
        return torch.cat((base, coherence[:, :, 1:]), dim=2).float()
    if representation == "amplitude_phase_wave_packets":
        if frequency_hz is None:
            raise ValueError("wave packets require physical frequencies")
        packets = _wave_packet_coordinates(
            complex_value, log_amplitude, reference, frequency_hz
        )
        floor = 0.05 * reference.square()
        horizontal = torch.ones_like(complex_value)
        vertical = torch.ones_like(complex_value)
        horizontal[:, :, :, 1:] = (
            complex_value[:, :, :, 1:] * complex_value[:, :, :, :-1].conj()
        )
        vertical[:, :, 1:, :] = (
            complex_value[:, :, 1:, :] * complex_value[:, :, :-1, :].conj()
        )
        horizontal = horizontal / (horizontal.abs() + floor + 1e-8)
        vertical = vertical / (vertical.abs() + floor + 1e-8)
        base = torch.stack(
            (
                log_amplitude,
                horizontal.real,
                horizontal.imag,
                vertical.real,
                vertical.imag,
            ),
            dim=2,
        )
        return torch.cat((base, packets[:, :, 1:]), dim=2).float()
    if representation == "amplitude_phase_wave_packet_gram":
        if frequency_hz is None:
            raise ValueError("wave packet Gram coordinates require physical frequencies")
        packet_gram = _wave_packet_gram_coordinates(
            complex_value, log_amplitude, frequency_hz
        )
        floor = 0.05 * reference.square()
        horizontal = torch.ones_like(complex_value)
        vertical = torch.ones_like(complex_value)
        horizontal[:, :, :, 1:] = (
            complex_value[:, :, :, 1:] * complex_value[:, :, :, :-1].conj()
        )
        vertical[:, :, 1:, :] = (
            complex_value[:, :, 1:, :] * complex_value[:, :, :-1, :].conj()
        )
        horizontal = horizontal / (horizontal.abs() + floor + 1e-8)
        vertical = vertical / (vertical.abs() + floor + 1e-8)
        base = torch.stack(
            (
                log_amplitude,
                horizontal.real,
                horizontal.imag,
                vertical.real,
                vertical.imag,
            ),
            dim=2,
        )
        return torch.cat((base, packet_gram[:, :, 1:]), dim=2).float()
    if representation == "amplitude_phase_multiscale_packet_gram":
        if frequency_hz is None:
            raise ValueError("multiscale packet Gram coordinates require physical frequencies")
        packet_gram = _multiscale_wave_packet_gram_coordinates(
            complex_value, log_amplitude, frequency_hz
        )
        floor = 0.05 * reference.square()
        horizontal = torch.ones_like(complex_value)
        vertical = torch.ones_like(complex_value)
        horizontal[:, :, :, 1:] = (
            complex_value[:, :, :, 1:] * complex_value[:, :, :, :-1].conj()
        )
        vertical[:, :, 1:, :] = (
            complex_value[:, :, 1:, :] * complex_value[:, :, :-1, :].conj()
        )
        horizontal = horizontal / (horizontal.abs() + floor + 1e-8)
        vertical = vertical / (vertical.abs() + floor + 1e-8)
        base = torch.stack(
            (
                log_amplitude,
                horizontal.real,
                horizontal.imag,
                vertical.real,
                vertical.imag,
            ),
            dim=2,
        )
        return torch.cat((base, packet_gram[:, :, 1:]), dim=2).float()
    if representation not in {
        "amplitude_soft_phase_gradient",
        "amplitude_edge_connection",
        "amplitude_transport_connection",
    }:
        raise ValueError(representation)
    floor = 0.05 * reference.square()
    horizontal = torch.ones_like(complex_value)
    vertical = torch.ones_like(complex_value)
    horizontal[:, :, :, 1:] = complex_value[:, :, :, 1:] * complex_value[:, :, :, :-1].conj()
    vertical[:, :, 1:, :] = complex_value[:, :, 1:, :] * complex_value[:, :, :-1, :].conj()
    if representation == "amplitude_soft_phase_gradient":
        horizontal = horizontal / (horizontal.abs() + floor + 1e-8)
        vertical = vertical / (vertical.abs() + floor + 1e-8)
    elif representation == "amplitude_edge_connection":
        # Normalized Hermitian edge products.  Their phase is the local phase
        # advance, while their magnitude records local amplitude transport.
        # The construction is invariant to an arbitrary global complex phase.
        horizontal_energy = torch.ones_like(amplitude)
        vertical_energy = torch.ones_like(amplitude)
        horizontal_energy[:, :, :, 1:] = 0.5 * (
            amplitude[:, :, :, 1:].square() + amplitude[:, :, :, :-1].square()
        )
        vertical_energy[:, :, 1:, :] = 0.5 * (
            amplitude[:, :, 1:, :].square() + amplitude[:, :, :-1, :].square()
        )
        horizontal = horizontal / (horizontal_energy + floor + 1e-8)
        vertical = vertical / (vertical_energy + floor + 1e-8)
    else:
        # A directed discrete complex connection.  Relative to the local
        # center energy it retains both the amplitude transport ratio and the
        # phase advance.  Unlike the symmetric coherence above, increasing
        # and decreasing envelopes are distinguishable.  The same definition
        # adds one complex edge per coordinate direction in any dimension.
        horizontal_energy = torch.ones_like(amplitude)
        vertical_energy = torch.ones_like(amplitude)
        horizontal_energy[:, :, :, 1:] = amplitude[:, :, :, :-1].square()
        vertical_energy[:, :, 1:, :] = amplitude[:, :, :-1, :].square()
        horizontal = horizontal / (horizontal_energy + floor + 1e-8)
        vertical = vertical / (vertical_energy + floor + 1e-8)
        horizontal[:, :, :, 0] = 0.0
        vertical[:, :, 0, :] = 0.0
    return torch.stack(
        (log_amplitude, horizontal.real, horizontal.imag, vertical.real, vertical.imag),
        dim=2,
    ).float()


class WaveBenchBank:
    def __init__(self, root: Path, scale_ids: np.ndarray, scale_count: int = 256):
        self.readers = {
            frequency: BetonReader(root / f"isotropic_{frequency}.beton")
            for frequency in FREQUENCIES
        }
        selected = scale_ids[: min(scale_count, len(scale_ids))]
        scales = []
        media = []
        for frequency in FREQUENCIES:
            medium, response = self.readers[frequency].read_many(selected, include_medium=not media)
            if medium is not None:
                media.append(medium)
            scales.append(float(np.sqrt(np.mean(response.astype(np.float64) ** 2))))
        target = media[0]
        self.target_mean = float(target.mean())
        self.target_std = float(target.std())
        self.scales = np.asarray(scales, dtype=np.float32)

    def load(
        self, indices: np.ndarray, frequency_indices: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        responses = []
        medium = None
        for position in frequency_indices:
            current_medium, response = self.readers[FREQUENCIES[position]].read_many(
                indices, include_medium=medium is None
            )
            if current_medium is not None:
                medium = current_medium
            responses.append(response / self.scales[position])
        assert medium is not None
        # [views,batch,complex,h,w] -> [batch,views,complex,h,w]
        return np.stack(responses, axis=1), medium

    @staticmethod
    def metadata(batch: int, frequency_indices: tuple[int, ...]) -> np.ndarray:
        frequency = np.asarray([FREQUENCIES[index] for index in frequency_indices], np.float32)
        normalized = (frequency - min(FREQUENCIES)) / (max(FREQUENCIES) - min(FREQUENCIES))
        metadata = np.stack(
            (np.zeros_like(normalized), np.zeros_like(normalized), normalized), axis=1
        )
        return np.broadcast_to(metadata, (batch, len(frequency_indices), 3)).copy()


def _subsets(budget: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations(range(len(FREQUENCIES)), budget))


@torch.no_grad()
def evaluate(
    encoder: WaveSetEncoder,
    decoder: ScalarFieldDecoder,
    bank: WaveBenchBank,
    indices: np.ndarray,
    representation: str,
    device: str,
    batch_size: int,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float | int]]]:
    encoder.eval()
    decoder.eval()
    budget_metrics = {}
    per_system: dict[int, list[float]] = {int(index): [] for index in indices}
    for budget in range(1, len(FREQUENCIES) + 1):
        subset_errors: list[np.ndarray] = []
        subset_physical: list[np.ndarray] = []
        named_subsets: dict[str, dict[str, float]] = {}
        for subset in _subsets(budget):
            errors = []
            physical_errors = []
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                response, target = bank.load(batch_indices, subset)
                value = response_coordinates(
                    torch.from_numpy(response).to(device),
                    representation,
                    tuple(FREQUENCIES[index] for index in subset),
                )
                metadata = torch.from_numpy(bank.metadata(len(batch_indices), subset)).to(device)
                prediction_normalized = decoder(encoder.spatial(value, metadata))
                prediction = (
                    prediction_normalized[:, 0].cpu().numpy() * bank.target_std + bank.target_mean
                )
                truth = target[:, 0]
                numerator = np.linalg.norm((prediction - truth).reshape(len(truth), -1), axis=1)
                contrast = np.linalg.norm(
                    (truth - bank.target_mean).reshape(len(truth), -1), axis=1
                )
                physical = np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
                errors.append(numerator / np.maximum(contrast, 1e-8))
                physical_errors.append(numerator / np.maximum(physical, 1e-8))
            current_error = np.concatenate(errors)
            current_physical = np.concatenate(physical_errors)
            subset_errors.append(current_error)
            subset_physical.append(current_physical)
            name = "+".join(str(FREQUENCIES[index]) for index in subset)
            named_subsets[name] = {
                "contrast_rel_l2_mean": float(current_error.mean()),
                "physical_rel_l2_mean": float(current_physical.mean()),
            }
        error_matrix = np.stack(subset_errors)
        physical_matrix = np.stack(subset_physical)
        mean_per_system = error_matrix.mean(0)
        for index, error in zip(indices, mean_per_system, strict=True):
            if budget == len(FREQUENCIES):
                per_system[int(index)].append(float(error))
        budget_metrics[str(budget)] = {
            "frequency_subsets": len(_subsets(budget)),
            "contrast_rel_l2_mean": float(error_matrix.mean()),
            "contrast_rel_l2_system_std": float(mean_per_system.std()),
            "physical_rel_l2_mean": float(physical_matrix.mean()),
            "subsets": named_subsets,
        }
    records = [
        {"system": index, "full_context_contrast_rel_l2": values[0]}
        for index, values in per_system.items()
    ]
    return budget_metrics, records


def train(args: argparse.Namespace) -> dict:
    if args.representation not in REPRESENTATIONS:
        raise ValueError(args.representation)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = make_split(
        args.split_seed, args.train_systems, args.validation_systems, args.test_systems
    )
    bank = WaveBenchBank(args.data_root, split.train)
    channels = {
        "raw": 2,
        "amplitude": 1,
        "amplitude_windowed_coherence": 17,
        "amplitude_phase_windowed_coherence": 21,
        "amplitude_phase_wave_packets": 29,
        "amplitude_phase_wave_packet_gram": 29,
        "amplitude_phase_multiscale_packet_gram": 69,
    }.get(args.representation, 5)
    encoder = WaveSetEncoder(
        channels,
        width=args.width,
        dimension=args.dimension,
        aggregation="slots",
        slots=args.slots,
    ).to(device)
    decoder = ScalarFieldDecoder(args.width).to(device)
    initialization = None
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        source_encoder = checkpoint["encoder"]
        target_encoder = encoder.state_dict()
        for name, target in target_encoder.items():
            source = source_encoder[name]
            if source.shape == target.shape:
                target.copy_(source)
            elif name == "backbone.first.0.weight" and (
                source.shape[0] == target.shape[0]
                and source.shape[2:] == target.shape[2:]
                and source.shape[1] < target.shape[1]
            ):
                target.zero_()
                target[:, : source.shape[1]].copy_(source)
            else:
                raise ValueError(
                    f"cannot transfer {name}: source {tuple(source.shape)} target {tuple(target.shape)}"
                )
        encoder.load_state_dict(target_encoder)
        decoder.load_state_dict(checkpoint["decoder"])
        initialization = {
            "checkpoint": str(args.initial_checkpoint),
            "source_representation": checkpoint["summary"]["representation"],
            "source_best_validation": checkpoint["summary"]["best_validation"],
            "extra_input_weights_zero_initialized": True,
        }
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    best_validation = float("inf")
    best_state = None
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        indices = rng.choice(split.train, args.batch_size, replace=False)
        budget = (
            len(FREQUENCIES)
            if rng.random() < args.full_context_probability
            else int(rng.integers(1, 5))
        )
        frequency_indices = tuple(int(value) for value in rng.choice(4, budget, replace=False))
        response, target = bank.load(indices, frequency_indices)
        value = response_coordinates(
            torch.from_numpy(response).to(device),
            args.representation,
            tuple(FREQUENCIES[index] for index in frequency_indices),
        )
        metadata = torch.from_numpy(bank.metadata(len(indices), frequency_indices)).to(device)
        truth = torch.from_numpy((target - bank.target_mean) / bank.target_std).to(device)
        encoder.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = decoder(encoder.spatial(value, metadata))
        loss = (prediction - truth).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.evaluate_every == 0 or step == args.steps:
            # Model selection averages two sparse views and the complete view.
            validation_errors = []
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                for subset in ((0,), (3,), (0, 3), (0, 1, 2, 3)):
                    response, target = bank.load(split.validation, subset)
                    predictions = []
                    for start in range(0, len(target), args.eval_batch_size):
                        batch_response = torch.from_numpy(
                            response[start : start + args.eval_batch_size]
                        ).to(device)
                        batch_metadata = torch.from_numpy(
                            bank.metadata(len(batch_response), subset)
                        ).to(device)
                        coordinate = response_coordinates(
                            batch_response,
                            args.representation,
                            tuple(FREQUENCIES[index] for index in subset),
                        )
                        predictions.append(
                            decoder(encoder.spatial(coordinate, batch_metadata)).cpu().numpy()
                        )
                    prediction = np.concatenate(predictions) * bank.target_std + bank.target_mean
                    numerator = np.linalg.norm(
                        (prediction - target).reshape(len(target), -1), axis=1
                    )
                    denominator = np.linalg.norm(
                        (target - bank.target_mean).reshape(len(target), -1), axis=1
                    )
                    validation_errors.append(
                        float(np.mean(numerator / np.maximum(denominator, 1e-8)))
                    )
            selection = float(np.mean(validation_errors))
            history.append(
                {
                    "step": step,
                    "train_normalized_mse": float(loss.detach()),
                    "selection_contrast_rel_l2": selection,
                    "validation_subsets": validation_errors,
                }
            )
            print(
                f"{args.representation} seed {args.seed} step {step:5d} "
                f"loss {float(loss.detach()):.5f} val {selection:.4f}",
                flush=True,
            )
            if selection < best_validation:
                best_validation = selection
                best_state = {
                    "encoder": {
                        name: value.detach().cpu() for name, value in encoder.state_dict().items()
                    },
                    "decoder": {
                        name: value.detach().cpu() for name, value in decoder.state_dict().items()
                    },
                }
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    budgets, records = evaluate(
        encoder, decoder, bank, split.test, args.representation, device, args.eval_batch_size
    )
    summary = {
        "schema_version": 1,
        "task": "wavebench_variable_frequency_system_identification",
        "dataset": "WaveBench isotropic acoustic time-harmonic",
        "representation": args.representation,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "frequencies_hz": list(FREQUENCIES),
        "split": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "train_sha256": ids_sha256(split.train),
            "validation_sha256": ids_sha256(split.validation),
            "test_sha256": ids_sha256(split.test),
        },
        "target_mean": bank.target_mean,
        "target_std": bank.target_std,
        "response_scales": bank.scales.tolist(),
        "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in parameters),
        "frequency_budget": budgets,
        "best_validation": best_validation,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "initialization": initialization,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.representation}_seed{args.seed}"
    torch.save({**best_state, "summary": summary}, args.output / f"{stem}.pt")
    (args.output / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    with (args.output / f"{stem}_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"representation": args.representation, "budgets": budgets}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--train-systems", type=int, default=8000)
    parser.add_argument("--validation-systems", type=int, default=500)
    parser.add_argument("--test-systems", type=int, default=500)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--evaluate-every", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--full-context-probability", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--initial-checkpoint", type=Path)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
