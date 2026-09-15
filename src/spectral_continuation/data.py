"""Leakage-safe loaders for paired complex frequency fields."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

EPS = 1e-8


def sha256_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


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
WAVEBENCH_GRID = 128
WAVEBENCH_INPUT_SHAPE = (1, WAVEBENCH_GRID, WAVEBENCH_GRID)
WAVEBENCH_TARGET_SHAPE = (2, WAVEBENCH_GRID, WAVEBENCH_GRID)


class WaveBenchBetonReader:
    """Minimal reader for the fixed WaveBench acoustic FFCV records."""

    def __init__(self, path: Path):
        self.path = Path(path)
        header = np.fromfile(self.path, dtype=HEADER_DTYPE, count=1)
        if header.size != 1 or int(header[0]["version"]) != 2:
            raise ValueError(f"unsupported FFCV header: {self.path}")
        self.header = header[0]
        self.fields = np.fromfile(
            self.path, dtype=FIELD_DTYPE, count=2, offset=HEADER_DTYPE.itemsize
        )
        names = tuple(
            bytes(field["name"]).split(b"\0", 1)[0].decode("ascii") for field in self.fields
        )
        if names != ("input", "target"):
            raise ValueError(f"unexpected WaveBench fields {names}")
        self.num_samples = int(self.header["num_samples"])
        self.raw = np.memmap(self.path, mode="r", dtype=np.uint8)
        self.metadata = np.memmap(
            self.path,
            mode="r",
            dtype=METADATA_DTYPE,
            offset=HEADER_DTYPE.itemsize + self.fields.nbytes,
            shape=(self.num_samples,),
        )

    def read_many(self, indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(list(indices), dtype=np.int64)
        medium = np.empty((len(indices), *WAVEBENCH_INPUT_SHAPE), dtype=np.float32)
        field = np.empty((len(indices), *WAVEBENCH_TARGET_SHAPE), dtype=np.float32)
        for out_index, sample_index in enumerate(indices):
            input_ptr, target_ptr = (int(v) for v in self.metadata[int(sample_index)])
            medium[out_index] = np.ndarray(
                WAVEBENCH_INPUT_SHAPE, dtype="<f4", buffer=self.raw, offset=input_ptr
            )
            field[out_index] = np.ndarray(
                WAVEBENCH_TARGET_SHAPE, dtype="<f4", buffer=self.raw, offset=target_ptr
            )
        return medium, field


@dataclass
class FieldBundle:
    fields: dict[float, torch.Tensor]
    environment: torch.Tensor
    ids: np.ndarray
    split: np.ndarray
    frequencies: tuple[float, ...]
    lineage: dict


def load_wavebench_development(
    source_run: Path,
    frequencies: Sequence[float],
    train_count: int | None = None,
) -> FieldBundle:
    """Load only frozen development IDs; held-out field bytes are never accessed."""
    source_run = Path(source_run).resolve()
    manifest_path = source_run / "manifest.json"
    split_path = source_run / "split_ids.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(split_path, allow_pickle=False) as archive:
        train_ids = np.asarray(archive["lf_train"], dtype=np.int64)
        val_ids = np.asarray(archive["lf_val"], dtype=np.int64)
        heldout_ids = np.asarray(archive["heldout_test"], dtype=np.int64)
    expected = manifest["split"]["hashes"]
    for name, values in (
        ("lf_train", train_ids),
        ("lf_val", val_ids),
        ("heldout_test", heldout_ids),
    ):
        if sha256_int64(values) != expected[name]:
            raise RuntimeError(f"frozen WaveBench split hash mismatch: {name}")
    if train_count is not None:
        if not 0 < train_count <= len(train_ids):
            raise ValueError("train_count is outside the frozen training pool")
        train_ids = train_ids[:train_count]
    ids = np.concatenate((train_ids, val_ids))
    split = np.asarray(["train"] * len(train_ids) + ["val"] * len(val_ids))
    root = Path(manifest["args"]["data_root"])
    medium = None
    tensors: dict[float, torch.Tensor] = {}
    for frequency in frequencies:
        path = root / f"{manifest['args']['medium']}_{int(frequency)}.beton"
        current_medium, current_field = WaveBenchBetonReader(path).read_many(ids)
        if medium is None:
            medium = current_medium
        elif not np.allclose(medium, current_medium, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"medium mismatch at {frequency:g} Hz")
        tensors[float(frequency)] = torch.from_numpy(current_field)
    assert medium is not None
    return FieldBundle(
        fields=tensors,
        environment=torch.from_numpy(medium),
        ids=ids,
        split=split,
        frequencies=tuple(map(float, frequencies)),
        lineage={
            "kind": "wavebench-development",
            "source_run": str(source_run),
            "source_manifest": str(manifest_path),
            "heldout_policy": "IDs verified; held-out fields never read",
            "heldout_hash": sha256_int64(heldout_ids),
        },
    )


def load_ocean_npz(path: Path) -> FieldBundle:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["complex_field"], dtype=np.float32)
        frequencies = np.asarray(archive["frequencies_hz"], dtype=np.float64)
        environment = np.asarray(archive["environment_channels"], dtype=np.float32)
        ids = np.asarray(archive["environment_ids"]).astype(str)
        split = np.asarray(archive["split"]).astype(str)
    if values.ndim != 5 or values.shape[-1] != 2:
        raise ValueError("Ocean complex_field must have shape [N,F,H,W,2]")
    fields = {
        float(frequency): torch.from_numpy(values[:, index].transpose(0, 3, 1, 2).copy())
        for index, frequency in enumerate(frequencies)
    }
    return FieldBundle(
        fields=fields,
        environment=torch.from_numpy(environment),
        ids=ids,
        split=split,
        frequencies=tuple(map(float, frequencies)),
        lineage={"kind": "ocean-paired-npz", "path": str(path)},
    )


def as_complex(field: torch.Tensor) -> torch.Tensor:
    return torch.complex(field[..., 0, :, :], field[..., 1, :, :])


def unit_complex(field: torch.Tensor) -> torch.Tensor:
    value = as_complex(field)
    return value / value.abs().clamp_min(EPS)


@dataclass(frozen=True)
class Normalization:
    environment_mean: torch.Tensor
    environment_std: torch.Tensor
    field_scales: dict[float, float]
    target_scale: float
    frequency_scale: float


def fit_normalization(
    bundle: FieldBundle,
    train_positions: np.ndarray,
    visible_frequencies: Sequence[float],
    target_frequency: float,
) -> Normalization:
    env = bundle.environment[train_positions].double()
    mean = env.mean(dim=(0, 2, 3), keepdim=True).float()
    std = env.std(dim=(0, 2, 3), keepdim=True).clamp_min(EPS).float()
    scales = {}
    for frequency in visible_frequencies:
        value = bundle.fields[float(frequency)][train_positions].double()
        scales[float(frequency)] = max(float(value.square().mean().sqrt()), EPS)
    target = bundle.fields[float(target_frequency)][train_positions].double()
    return Normalization(
        environment_mean=mean,
        environment_std=std,
        field_scales=scales,
        target_scale=max(float(target.square().mean().sqrt()), EPS),
        frequency_scale=max(bundle.frequencies),
    )


def spectral_carrier(
    fields: dict[float, torch.Tensor],
    visible_frequencies: Sequence[float],
    target_frequency: float,
    positions: np.ndarray,
) -> torch.Tensor:
    visible = sorted(map(float, visible_frequencies))
    top = visible[-1]
    q_top = unit_complex(fields[top][positions])
    if len(visible) == 1:
        phase = (target_frequency / top) * torch.angle(q_top)
    else:
        gaps = np.diff(visible)
        if len(visible) >= 3 and np.allclose(gaps, gaps[0]):
            # Accumulate principal adjacent increments, then regress the
            # unwrapped local spectral phase. This averages noise without
            # requiring an absolute phase gauge. It is exact only when the
            # adjacent frequency sampling resolves the local delay.
            unwrapped = [torch.angle(unit_complex(fields[visible[0]][positions]))]
            for previous, current in pairwise(visible):
                ratio = unit_complex(fields[current][positions]) * torch.conj(
                    unit_complex(fields[previous][positions])
                )
                unwrapped.append(unwrapped[-1] + torch.angle(ratio))
            frequency_offsets = torch.as_tensor(
                np.asarray(visible) - np.mean(visible), dtype=unwrapped[0].dtype
            )
            phase_stack = torch.stack(unwrapped, dim=0)
            phase_center = phase_stack.mean(dim=0)
            slope = torch.sum(frequency_offsets.view(-1, 1, 1, 1) * phase_stack, dim=0) / torch.sum(
                frequency_offsets.square()
            )
            phase = phase_center + (target_frequency - np.mean(visible)) * slope
        else:
            previous = visible[-2]
            q_previous = unit_complex(fields[previous][positions])
            spectral_increment = torch.angle(q_top * torch.conj(q_previous))
            steps = (target_frequency - top) / (top - previous)
            phase = torch.angle(q_top) + steps * spectral_increment
    carrier = torch.polar(torch.ones_like(phase), phase)
    return torch.stack((carrier.real, carrier.imag), dim=1).float()


class SpectralQueryDataset(Dataset):
    """Fixed-slot single-query dataset used for the stage-1 causal comparison."""

    def __init__(
        self,
        bundle: FieldBundle,
        positions: np.ndarray,
        visible_frequencies: Sequence[float],
        target_frequency: float,
        normalization: Normalization,
        max_slots: int = 3,
        field_only: bool = False,
    ):
        visible = tuple(sorted(map(float, visible_frequencies)))
        if not visible or len(visible) > max_slots:
            raise ValueError("visible frequency count must be between one and max_slots")
        if target_frequency not in bundle.fields:
            raise KeyError(f"missing target frequency {target_frequency}")
        self.bundle = bundle
        self.positions = np.asarray(positions, dtype=np.int64)
        self.visible = visible
        self.target_frequency = float(target_frequency)
        self.norm = normalization
        self.max_slots = max_slots
        self.field_only = field_only
        self.carrier = spectral_carrier(
            bundle.fields, visible, self.target_frequency, self.positions
        )

    @property
    def input_channels(self) -> int:
        return self.bundle.environment.shape[1] + 4 * self.max_slots + 3

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int):
        position = int(self.positions[index])
        env = (
            self.bundle.environment[position] - self.norm.environment_mean[0]
        ) / self.norm.environment_std[0]
        if self.field_only:
            env = torch.zeros_like(env)
        height, width = env.shape[-2:]
        channels = [env]
        padding = self.max_slots - len(self.visible)
        for _ in range(padding):
            channels.append(torch.zeros((4, height, width), dtype=env.dtype))
        for frequency in self.visible:
            field = self.bundle.fields[frequency][position] / self.norm.field_scales[frequency]
            frequency_map = torch.full(
                (1, height, width), frequency / self.norm.frequency_scale, dtype=env.dtype
            )
            mask = torch.ones((1, height, width), dtype=env.dtype)
            channels.append(torch.cat((field, frequency_map, mask), dim=0))
        y_coord = (
            torch.linspace(-1.0, 1.0, height, dtype=env.dtype)
            .view(1, height, 1)
            .expand(1, height, width)
        )
        x_coord = (
            torch.linspace(-1.0, 1.0, width, dtype=env.dtype)
            .view(1, 1, width)
            .expand(1, height, width)
        )
        query = torch.full(
            (1, height, width), self.target_frequency / self.norm.frequency_scale, dtype=env.dtype
        )
        channels.extend((y_coord, x_coord, query))
        target = self.bundle.fields[self.target_frequency][position]
        return torch.cat(channels, dim=0), self.carrier[index], target


class MultiQueryDataset(Dataset):
    """Cartesian product of environments and query frequencies."""

    def __init__(
        self,
        bundle: FieldBundle,
        positions: np.ndarray,
        visible_frequencies: Sequence[float],
        query_frequencies: Sequence[float],
        normalization: Normalization,
        max_slots: int = 3,
        field_only: bool = False,
    ):
        self.positions = np.asarray(positions, dtype=np.int64)
        self.queries = tuple(map(float, query_frequencies))
        if not self.queries:
            raise ValueError("at least one query frequency is required")
        for query in self.queries:
            if query not in bundle.fields:
                raise KeyError(f"missing query frequency {query:g}")
        self.datasets = [
            SpectralQueryDataset(
                bundle,
                self.positions,
                visible_frequencies,
                query,
                normalization,
                max_slots,
                field_only,
            )
            for query in self.queries
        ]

    @property
    def input_channels(self) -> int:
        return self.datasets[0].input_channels

    def __len__(self) -> int:
        return len(self.positions) * len(self.queries)

    def __getitem__(self, index: int):
        query_index = index % len(self.queries)
        position_index = index // len(self.queries)
        inputs, carrier, target = self.datasets[query_index][position_index]
        return inputs, carrier, target, query_index
