"""Cross-domain adapters for source--frequency response-query experiments.

The immutable archive schema deliberately stores only measured responses and
query geometry. Hidden media may be retained by the original dataset for
diagnostics, but are never exposed to the response-query model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "cross-domain-response-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


@dataclass(frozen=True)
class CrossDomainResponseArchive:
    responses: np.ndarray
    sources: np.ndarray
    receivers: np.ndarray
    frequencies: np.ndarray
    source_receiver_index: np.ndarray
    split: np.ndarray
    system_ids: np.ndarray
    valid_response_mask: np.ndarray
    provenance: dict

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.responses.shape)


def validate_archive(archive: CrossDomainResponseArchive) -> None:
    if archive.responses.ndim != 4 or not np.iscomplexobj(archive.responses):
        raise ValueError("responses must be complex [environment,source,frequency,receiver]")
    systems, sources, frequencies, receivers = archive.responses.shape
    if archive.sources.shape[0] != sources or archive.receivers.shape[0] != receivers:
        raise ValueError("coordinate counts do not match response axes")
    if archive.frequencies.shape != (frequencies,):
        raise ValueError("frequency count does not match response axis")
    if archive.source_receiver_index.shape != (sources,):
        raise ValueError("source_receiver_index must have one entry per source")
    if archive.split.shape != (systems,) or archive.system_ids.shape != (systems,):
        raise ValueError("split and system_ids must have one entry per environment")
    if archive.valid_response_mask.shape != (sources, receivers):
        raise ValueError("valid_response_mask must have shape [source,receiver]")
    if not np.isfinite(archive.responses.real).all() or not np.isfinite(
        archive.responses.imag
    ).all():
        raise ValueError("responses contain non-finite values")
    if np.any(archive.source_receiver_index >= receivers) or np.any(
        archive.source_receiver_index < -1
    ):
        raise ValueError("source_receiver_index is out of range")
    if len(set(archive.system_ids.astype(str).tolist())) != systems:
        raise ValueError("system_ids must be unique")


def load_cross_domain_archive(path: Path) -> CrossDomainResponseArchive:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema_version"])
        if schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported response archive schema {schema!r}")
        archive = CrossDomainResponseArchive(
            responses=np.asarray(data["responses"], dtype=np.complex64),
            sources=np.asarray(data["sources"], dtype=np.float32),
            receivers=np.asarray(data["receivers"], dtype=np.float32),
            frequencies=np.asarray(data["frequencies"], dtype=np.float32),
            source_receiver_index=np.asarray(data["source_receiver_index"], dtype=np.int64),
            split=np.asarray(data["split"]).astype(str),
            system_ids=np.asarray(data["system_ids"]).astype(str),
            valid_response_mask=np.asarray(data["valid_response_mask"], dtype=bool),
            provenance=json.loads(str(data["provenance_json"])),
        )
    validate_archive(archive)
    return archive


def save_archive(path: Path, archive: CrossDomainResponseArchive) -> dict:
    validate_archive(archive)
    atomic_savez(
        path,
        schema_version=np.asarray(SCHEMA_VERSION),
        responses=archive.responses.astype(np.complex64),
        sources=archive.sources.astype(np.float32),
        receivers=archive.receivers.astype(np.float32),
        frequencies=archive.frequencies.astype(np.float32),
        source_receiver_index=archive.source_receiver_index.astype(np.int64),
        split=archive.split.astype("U"),
        system_ids=archive.system_ids.astype("U"),
        valid_response_mask=archive.valid_response_mask.astype(bool),
        provenance_json=np.asarray(json.dumps(archive.provenance, sort_keys=True)),
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "shape": list(archive.shape),
        "splits": {
            name: int(np.sum(archive.split == name)) for name in sorted(set(archive.split))
        },
        "frequencies": archive.frequencies.astype(float).tolist(),
        "provenance": archive.provenance,
    }


def prepare_openfwi(config: dict, output: Path) -> dict:
    domain = config["domains"]["openfwi_curvefault_a"]
    root = Path(domain["raw_root"])
    dt = float(domain["dt_seconds"])
    selected_frequencies = np.asarray(domain["frequencies_hz"], dtype=np.float64)
    source_receiver_index = np.asarray(domain["source_receiver_index"], dtype=np.int64)
    responses = []
    splits = []
    system_ids = []
    input_files = []
    for split, shard in domain["shards"].items():
        path = root / f"seis2_1_{int(shard)}.npy"
        values = np.load(path, mmap_mode="r")
        if values.ndim != 4 or values.shape[1:] != (5, 1000, 70):
            raise ValueError(f"unexpected OpenFWI shape {values.shape} in {path}")
        bins = np.rint(selected_frequencies * values.shape[2] * dt).astype(np.int64)
        actual = np.fft.rfftfreq(values.shape[2], dt)[bins]
        if not np.allclose(actual, selected_frequencies, rtol=0.0, atol=1e-6):
            raise ValueError("requested OpenFWI frequencies do not lie on FFT bins")
        transformed = np.fft.rfft(np.asarray(values, dtype=np.float32), axis=2)
        responses.append(transformed[:, :, bins, :].astype(np.complex64))
        splits.extend([split] * len(values))
        system_ids.extend([f"curvefault_a_shard{shard}_{index:04d}" for index in range(len(values))])
        input_files.append(
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    response = np.concatenate(responses, axis=0)
    receiver_x = np.linspace(0.0, 1.0, response.shape[-1], dtype=np.float32)
    receivers = np.stack((receiver_x, np.zeros_like(receiver_x)), axis=1)
    sources = receivers[source_receiver_index]
    valid = np.ones((response.shape[1], response.shape[-1]), dtype=bool)
    archive = CrossDomainResponseArchive(
        responses=response,
        sources=sources,
        receivers=receivers,
        frequencies=selected_frequencies.astype(np.float32),
        source_receiver_index=source_receiver_index,
        split=np.asarray(splits),
        system_ids=np.asarray(system_ids),
        valid_response_mask=valid,
        provenance={
            "domain": "OpenFWI CurveFault-A",
            "operation": "unwindowed real FFT along the 1000-sample time axis",
            "dt_seconds": dt,
            "source_receiver_alignment": source_receiver_index.tolist(),
            "alignment_policy": "official evenly spaced surface geometry; train-shard reciprocity audit",
            "input_files": input_files,
            "test_status": "frozen replication shard; previously used by a different inverse task",
        },
    )
    manifest = save_archive(output, archive)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _well_source_indices(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    first = min(root.glob("helmholtz_staircase_omega_*.hdf5"))
    with h5py.File(first, "r") as handle:
        real = handle["t0_fields/pressure_re"][:, 0]
        imaginary = handle["t0_fields/pressure_im"][:, 0]
        field = np.asarray(real) + 1j * np.asarray(imaginary)
        wall = np.asarray(handle["boundary_conditions/xy_wall/mask"], dtype=bool)
        x_axis = np.asarray(handle["dimensions/x"], dtype=np.float32)
        y_axis = np.asarray(handle["dimensions/y"], dtype=np.float32)
    amplitude = np.where((~wall)[None], np.abs(field), 0.0)
    flat = np.argmax(amplitude.reshape(len(field), -1), axis=1)
    row, column = np.unravel_index(flat, wall.shape)
    # The stored field y axis is reversed relative to dimensions/y.
    xy = np.stack((x_axis[row], y_axis[::-1][column]), axis=1).astype(np.float32)
    return np.stack((row, column), axis=1), xy, wall


def prepare_the_well(config: dict, output: Path) -> dict:
    import h5py

    root = Path(config["domains"]["the_well_staircase"]["root"])
    paths = sorted(root.glob("helmholtz_staircase_omega_*.hdf5"))
    if len(paths) != 16:
        raise ValueError(f"expected 16 The Well frequencies, found {len(paths)}")
    indices, coordinates, _ = _well_source_indices(root)
    fields = []
    omegas = []
    input_files = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            omegas.append(float(handle["scalars/omega"][()]))
            real = np.asarray(handle["t0_fields/pressure_re"][:, 0])
            imaginary = np.asarray(handle["t0_fields/pressure_im"][:, 0])
        complex_field = np.conj(real + 1j * imaginary)
        sampled = complex_field[:, indices[:, 0], indices[:, 1]]
        fields.append(sampled.astype(np.complex64))
        input_files.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size})
    order = np.argsort(omegas)
    response = np.stack([fields[index] for index in order], axis=1)[None]
    frequency = np.asarray(omegas, dtype=np.float32)[order] / (2.0 * np.pi)
    unique_coordinates = len(np.unique(coordinates, axis=0))
    source_receiver_index = (
        np.arange(len(coordinates), dtype=np.int64)
        if unique_coordinates == len(coordinates)
        else np.full(len(coordinates), -1, dtype=np.int64)
    )
    archive = CrossDomainResponseArchive(
        responses=response,
        sources=coordinates,
        receivers=coordinates.copy(),
        frequencies=frequency,
        source_receiver_index=source_receiver_index,
        split=np.asarray(["sanity"]),
        system_ids=np.asarray(["the_well_staircase_fixed_geometry"]),
        valid_response_mask=np.ones((len(coordinates), len(coordinates)), dtype=bool),
        provenance={
            "domain": "The Well helmholtz_staircase train split",
            "operation": "sample each complex full field at the 26 inferred point-source coordinates",
            "time_convention": "conjugated to exp(-i omega tau)",
            "evidence_role": "fixed-system public sanity check; not system-held-out validation",
            "source_coordinate_status": (
                "inferred_unique"
                if unique_coordinates == len(coordinates)
                else "unresolved: all low-frequency peak estimates collapse to one cell; "
                "HDF5 does not store source coordinates"
            ),
            "input_files": input_files,
        },
    )
    manifest = save_archive(output, archive)
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _random_velocity_models(count: int, grid: int, seed: int, ood: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, grid), np.linspace(-1, 1, grid), indexing="ij")
    models = np.empty((count, grid, grid), dtype=np.float32)
    for sample in range(count):
        phase = rng.uniform(0, 2 * np.pi, size=4)
        smooth = (
            np.sin(np.pi * xx + phase[0])
            + np.cos(np.pi * yy + phase[1])
            + 0.5 * np.sin(np.pi * (xx + yy) + phase[2])
            + 0.5 * np.cos(2 * np.pi * (xx - yy) + phase[3])
        )
        velocity = 1500.0 + 18.0 * smooth
        inclusion_count = int(rng.integers(1, 5))
        for _ in range(inclusion_count):
            cx, cy = rng.uniform(-0.48, 0.48, size=2)
            if ood:
                radius = rng.uniform(0.10, 0.25)
                angle = np.arctan2(yy - cy, xx - cx)
                radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                boundary = radius * (1.0 + 0.28 * np.cos(5.0 * angle + rng.uniform(0, 2 * np.pi)))
                mask = radial < boundary
                contrast = rng.choice((-1.0, 1.0)) * rng.uniform(90.0, 180.0)
            else:
                rx, ry = rng.uniform(0.08, 0.23, size=2)
                rotation = rng.uniform(0, np.pi)
                xr = np.cos(rotation) * (xx - cx) + np.sin(rotation) * (yy - cy)
                yr = -np.sin(rotation) * (xx - cx) + np.cos(rotation) * (yy - cy)
                mask = (xr / rx) ** 2 + (yr / ry) ** 2 < 1.0
                contrast = rng.choice((-1.0, 1.0)) * rng.uniform(60.0, 140.0)
            velocity[mask] += contrast
        models[sample] = np.clip(velocity, 1250.0, 1800.0)
    return models


def _ring_coordinates(grid: int, count: int, radius: float) -> np.ndarray:
    angle = np.arange(count, dtype=np.float64) * (2.0 * np.pi / count)
    center = (grid - 1) / 2.0
    row = np.rint(center + radius * np.sin(angle)).astype(np.int64)
    column = np.rint(center + radius * np.cos(angle)).astype(np.int64)
    return np.stack((row, column), axis=1)


def prepare_deepwave(config: dict, output: Path, smoke: bool = False) -> dict:
    import deepwave
    import torch

    domain = config["domains"]["deepwave_ring"]
    grid = int(domain["grid"])
    spacing = float(domain["spacing_m"])
    source_count = int(domain["transducers"])
    receiver_count = int(domain.get("receivers", 64))
    frequency = np.asarray(domain["frequencies_hz"], dtype=np.float64)
    counts = dict(domain["systems"])
    if smoke:
        counts = {"train": 8, "validation": 4, "test": 4, "ood": 4}
    # Keep the physical array outside the absorbing PML; otherwise the direct
    # arrival is attenuated before it enters the modeled domain.
    receiver_rc = _ring_coordinates(grid, receiver_count, 0.32 * grid)
    if len(np.unique(receiver_rc, axis=0)) != receiver_count:
        raise ValueError(
            f"{receiver_count} receivers collapse to duplicate cells on the {grid}-grid ring"
        )
    if receiver_count % source_count:
        raise ValueError("receiver count must be divisible by source count for exact angular alignment")
    source_receiver_index = np.arange(0, receiver_count, receiver_count // source_count)
    source_rc = receiver_rc[source_receiver_index]
    receiver_xy = receiver_rc[:, ::-1].astype(np.float32) / float(grid - 1)
    source_xy = source_rc[:, ::-1].astype(np.float32) / float(grid - 1)
    max_velocity = 1800.0
    dt = 1.5e-7
    nt = 720
    peak_frequency = float(domain["peak_frequency_hz"])
    wavelet = deepwave.wavelets.ricker(peak_frequency, nt, dt, 1.5 / peak_frequency)
    bins = np.rint(frequency * nt * dt).astype(np.int64)
    actual_frequency = np.fft.rfftfreq(nt, dt)[bins]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_response = []
    all_split = []
    all_ids = []
    batch_systems = 2 if device.type == "cuda" else 1
    for split_index, (split, count) in enumerate(counts.items()):
        velocity = _random_velocity_models(
            int(count), grid, 20260924 + split_index, ood=split == "ood"
        )
        split_responses = []
        for start in range(0, len(velocity), batch_systems):
            batch = torch.from_numpy(velocity[start : start + batch_systems]).to(device)
            systems = len(batch)
            shot_velocity = batch[:, None].expand(-1, source_count, -1, -1).reshape(
                systems * source_count, grid, grid
            )
            source_locations = torch.from_numpy(
                np.broadcast_to(source_rc[None], (systems, source_count, 2)).reshape(-1, 1, 2).copy()
            ).to(device)
            receiver_locations = torch.from_numpy(
                np.broadcast_to(
                    receiver_rc[None, None], (systems, source_count, receiver_count, 2)
                ).reshape(-1, receiver_count, 2).copy()
            ).to(device)
            amplitudes = wavelet.to(device)[None, None].expand(systems * source_count, 1, -1)
            with torch.no_grad():
                trace = deepwave.scalar(
                    shot_velocity,
                    spacing,
                    dt,
                    source_amplitudes=amplitudes,
                    source_locations=source_locations,
                    receiver_locations=receiver_locations,
                    accuracy=4,
                    pml_width=12,
                    pml_freq=float(frequency.max()),
                    max_vel=max_velocity,
                    storage_mode="none",
                )[-1]
            trace = trace.reshape(systems, source_count, receiver_count, nt).cpu().numpy()
            transformed = np.fft.rfft(trace, axis=-1)[..., bins]
            transformed = transformed.transpose(0, 1, 3, 2).astype(np.complex64)
            split_responses.append(transformed)
        all_response.append(np.concatenate(split_responses, axis=0))
        all_split.extend([split] * int(count))
        all_ids.extend([f"deepwave_{split}_{index:05d}" for index in range(int(count))])
    archive = CrossDomainResponseArchive(
        responses=np.concatenate(all_response, axis=0),
        sources=source_xy,
        receivers=receiver_xy,
        frequencies=actual_frequency.astype(np.float32),
        source_receiver_index=source_receiver_index.astype(np.int64),
        split=np.asarray(all_split),
        system_ids=np.asarray(all_ids),
        valid_response_mask=np.ones((source_count, receiver_count), dtype=bool),
        provenance={
            "domain": "Deepwave 2D ring ultrasound",
            "deepwave_version": "0.0.26",
            "solver": "constant-density scalar time-domain finite difference",
            "grid": grid,
            "spacing_m": spacing,
            "dt_seconds": dt,
            "time_steps": nt,
            "peak_frequency_hz": peak_frequency,
            "requested_frequencies_hz": frequency.tolist(),
            "actual_frequencies_hz": actual_frequency.tolist(),
            "sources": source_count,
            "receivers": receiver_count,
            "test_policy": "IID held-out systems; no test use for selection",
            "ood_shift": "five-lobed inclusions and wider contrast",
            "smoke": smoke,
        },
    )
    manifest = save_archive(output, archive)
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _ocean_parameter_table(counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    splits = []
    ids = []
    for split_index, (split, count) in enumerate(counts.items()):
        generator = np.random.default_rng(20260924 + split_index)
        for index in range(int(count)):
            if split == "ood":
                water_speed = generator.uniform(1440.0, 1560.0)
                seabed_speed = generator.uniform(2050.0, 2350.0)
                density = generator.uniform(2150.0, 2500.0)
            else:
                water_speed = generator.uniform(1470.0, 1530.0)
                seabed_speed = generator.uniform(1750.0, 2050.0)
                density = generator.uniform(1750.0, 2200.0)
            rows.append((water_speed, seabed_speed, density))
            splits.append(split)
            ids.append(f"ocean_{split}_{index:05d}")
    return np.asarray(rows, dtype=np.float64), np.asarray(splits), np.asarray(ids)


def prepare_ocean(config: dict, output: Path, smoke: bool = False) -> dict:
    """Generate range-invariant multistatic Pekeris response matrices."""

    from juliacall import Main as jl  # type: ignore

    domain = config["domains"]["ocean_multistatic"]
    counts = dict(domain["systems"])
    if smoke:
        counts = {"train": 4, "validation": 2, "test": 2, "ood": 2}
    parameters, split, system_ids = _ocean_parameter_table(counts)
    depth_config = domain["receiver_depths_m"]
    receiver_depth = np.linspace(
        float(depth_config["start"]),
        float(depth_config["stop"]),
        int(depth_config["count"]),
    )
    source_count = int(domain["transducers"])
    source_receiver_index = np.rint(
        np.linspace(0, len(receiver_depth) - 1, source_count)
    ).astype(np.int64)
    source_depth = receiver_depth[source_receiver_index]
    frequencies = np.asarray(domain["frequencies_hz"], dtype=np.float64)
    range_m = float(domain["range_m"])
    jl.seval("using UnderwaterAcoustics")
    jl.seval(
        """
        function __sco_multistatic_pekeris(
            frequency_hz, water_speed, seabed_speed, seabed_density,
            source_depths, receiver_depths, range_m)
          env = UnderwaterEnvironment(
            bathymetry=120.0,
            seabed=FluidBoundary(seabed_density, seabed_speed, 0.0),
            soundspeed=water_speed)
          model = PekerisModeSolver(env; nmodes=0, ngrid=0)
          receivers = [AcousticReceiver(range_m, -depth) for depth in receiver_depths]
          values = Matrix{ComplexF64}(undef, length(source_depths), length(receiver_depths))
          for (index, depth) in enumerate(source_depths)
            source = AcousticSource(0.0, -depth, frequency_hz)
            values[index, :] = vec(acoustic_field(model, source, receivers; mode=:coherent))
          end
          return values
        end
        """
    )
    solve = getattr(jl, "__sco_multistatic_pekeris")
    response = np.empty(
        (len(parameters), source_count, len(frequencies), len(receiver_depth)),
        dtype=np.complex64,
    )
    start = time.time()
    for system, (water_speed, seabed_speed, density) in enumerate(parameters):
        for frequency_index, frequency in enumerate(frequencies):
            values = solve(
                float(frequency),
                float(water_speed),
                float(seabed_speed),
                float(density),
                source_depth,
                receiver_depth,
                range_m,
            )
            response[system, :, frequency_index] = np.asarray(values, dtype=np.complex64)
    elapsed = time.time() - start
    sources = np.stack(
        (np.zeros_like(source_depth), source_depth / 120.0), axis=1
    ).astype(np.float32)
    receivers = np.stack(
        (np.full_like(receiver_depth, range_m / 4000.0), receiver_depth / 120.0), axis=1
    ).astype(np.float32)
    archive = CrossDomainResponseArchive(
        responses=response,
        sources=sources,
        receivers=receivers,
        frequencies=frequencies.astype(np.float32),
        source_receiver_index=source_receiver_index,
        split=split,
        system_ids=system_ids,
        valid_response_mask=np.ones((source_count, len(receiver_depth)), dtype=bool),
        provenance={
            "domain": "range-invariant ocean acoustics",
            "solver": "UnderwaterAcoustics.jl PekerisModeSolver nmodes=0 ngrid=0",
            "water_depth_m": 120.0,
            "range_m": range_m,
            "receiver_depths_m": receiver_depth.tolist(),
            "source_depths_m": source_depth.tolist(),
            "environment_parameters": [
                "water_speed_mps",
                "seabed_speed_mps",
                "seabed_density_kgm3",
            ],
            "environment_parameter_values": parameters.tolist(),
            "test_policy": "IID held-out environments; no test use for selection",
            "ood_shift": "wider water-speed range and faster, denser seabed",
            "elapsed_seconds": elapsed,
            "smoke": smoke,
        },
    )
    manifest = save_archive(output, archive)
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--domain", choices=("openfwi", "deepwave", "ocean", "the_well"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.domain == "openfwi":
        manifest = prepare_openfwi(config, args.output)
    elif args.domain == "deepwave":
        manifest = prepare_deepwave(config, args.output, args.smoke)
    elif args.domain == "ocean":
        manifest = prepare_ocean(config, args.output, args.smoke)
    else:
        manifest = prepare_the_well(config, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
