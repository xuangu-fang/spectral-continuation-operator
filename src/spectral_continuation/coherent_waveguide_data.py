"""Controlled range-dependent one-way acoustic waveguide, NOT full Helmholtz.

Fixed sine Galerkin basis in depth, pressure-release top and rigid bottom.
The positive/decaying square root of the local vertical Helmholtz matrix drives
midpoint exponential marching. Backscatter and penetrable seabed are omitted.
Stored modal coefficients permit direct continuous depth evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from .coherent_representation import complex_metrics


def environment(seed):
    rng = np.random.default_rng(seed)
    return {
        "seed": int(seed),
        "depth_m": 100.0,
        "range_m": 2000.0,
        "reference_speed_m_s": 1500.0,
        "thermocline_amplitude_m_s": float(rng.uniform(10, 30)),
        "thermocline_depth_m": float(rng.uniform(30, 65)),
        "thermocline_width_m": float(rng.uniform(8, 20)),
        "centers_m": rng.uniform([300, 20], [1700, 80], (3, 2)).tolist(),
        "widths_m": rng.uniform([150, 8], [350, 25], (3, 2)).tolist(),
        "contrasts_m_s": rng.uniform(-35, 35, 3).tolist(),
        "source_depth_m": float(rng.uniform(25, 70)),
        "source_width_m": float(rng.uniform(6, 10)),
        "attenuation_per_m": 0.00002,
    }


def sound_speed(spec, ranges, depths):
    rr, zz = np.broadcast_arrays(np.asarray(ranges), np.asarray(depths))
    speed = spec["reference_speed_m_s"] - spec["thermocline_amplitude_m_s"] * np.tanh(
        (zz - spec["thermocline_depth_m"]) / spec["thermocline_width_m"]
    )
    for center, width, contrast in zip(
        spec["centers_m"], spec["widths_m"], spec["contrasts_m_s"], strict=True
    ):
        speed = speed + contrast * np.exp(
            -0.5 * ((rr - center[0]) / width[0]) ** 2 - 0.5 * ((zz - center[1]) / width[1]) ** 2
        )
    if np.min(speed) <= 1300:
        raise ValueError("unsupported wavespeed")
    return speed


def depth_basis(depths, depth, modes):
    kz = (np.arange(modes) + 0.5) * np.pi / depth
    return np.sqrt(2 / depth) * np.sin(np.asarray(depths)[:, None] * kz[None])


def quadrature(spec, modes, nodes=128):
    abscissa, weights = np.polynomial.legendre.leggauss(nodes)
    z = (abscissa + 1) * spec["depth_m"] / 2
    weights = weights * spec["depth_m"] / 2
    basis = depth_basis(z, spec["depth_m"], modes)
    return z, weights, basis


def local_generator(spec, frequency, radius, z, weights, basis):
    modes = basis.shape[1]
    kz = (np.arange(modes) + 0.5) * np.pi / spec["depth_m"]
    speed = sound_speed(spec, radius, z)
    vertical = basis.T @ (
        (weights * (2 * np.pi * frequency / speed) ** 2)[:, None] * basis
    ) - np.diag(kz**2)
    values, vectors = eigh(vertical, check_finite=False)
    propagation = np.sqrt(values.astype(complex))
    return propagation, vectors


def solve(spec, frequencies, range_points, depth_points, modes=32, quadrature_nodes=128):
    ranges = np.linspace(0, spec["range_m"], range_points)
    depths = np.linspace(0, spec["depth_m"], depth_points)
    z, weights, basis = quadrature(spec, modes, quadrature_nodes)
    source = np.exp(-0.5 * ((z - spec["source_depth_m"]) / spec["source_width_m"]) ** 2)
    initial = basis.T @ (weights * source)
    readout = depth_basis(depths, spec["depth_m"], modes)
    all_coefficients, elapsed = [], []
    for frequency in frequencies:
        start = time.perf_counter()
        coefficients = [initial.astype(complex)]
        for lo, hi in itertools.pairwise(ranges):
            propagation, vectors = local_generator(
                spec, frequency, (lo + hi) / 2, z, weights, basis
            )
            phase = np.exp((1j * propagation - spec["attenuation_per_m"]) * (hi - lo))
            coefficients.append(vectors @ (phase * (vectors.T @ coefficients[-1])))
        all_coefficients.append(np.stack(coefficients))
        elapsed.append(time.perf_counter() - start)
    coefficients = np.stack(all_coefficients)
    fields = np.einsum("zm,frm->fzr", readout, coefficients)
    return fields, coefficients, ranges, depths, elapsed


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    records = []
    seeds = config.get(
        "seeds",
        list(
            range(
                config.get("seed_start", 0),
                config.get("seed_start", 0) + config.get("system_count", 0),
            )
        ),
    )
    for seed in seeds:
        spec = environment(seed)
        completed = {}
        for modes in config["mode_counts"]:
            for count in config["range_counts"]:
                fields, coefficients, ranges, depths, elapsed = solve(
                    spec,
                    config["frequencies"],
                    count,
                    config["depth_points"],
                    modes,
                    config.get("quadrature_nodes", 128),
                )
                completed[(modes, count)] = fields
                path = output / f"system{seed}_m{modes}_r{count}.npz"
                if path.exists():
                    raise FileExistsError(path)
                np.savez_compressed(
                    path,
                    field=fields.astype(np.complex64),
                    modal_coefficients=coefficients.astype(np.complex64),
                    ranges_m=ranges,
                    depths_m=depths,
                    frequencies_hz=config["frequencies"],
                    spec=json.dumps(spec),
                    phase_convention="exp(-i omega t), forward exp(+i k_r r); source initial profile real",
                    evidence="development; one-way wide-angle square-root Galerkin model, NOT full Helmholtz or a penetrable-seabed Ocean simulator",
                )
                record = {
                    "system": seed,
                    "modes": modes,
                    "ranges": count,
                    "solve_seconds": sum(elapsed),
                    "norm_ratio_final_initial_max": float(
                        np.max(
                            np.linalg.norm(coefficients[:, -1], axis=-1)
                            / np.linalg.norm(coefficients[:, 0], axis=-1)
                        )
                    ),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
        reference_modes, reference_count = max(config["mode_counts"]), max(config["range_counts"])
        reference = completed[(reference_modes, reference_count)]
        for (modes, count), fields in completed.items():
            ratio = (reference_count - 1) // (count - 1)
            if ratio * (count - 1) != reference_count - 1:
                raise ValueError("audit grids must nest")
            ranges = np.linspace(0, spec["range_m"], count)
            keep = ranges >= config.get("roi_min_range_m", 50)
            for i, frequency in enumerate(config["frequencies"]):
                record = {
                    "system": seed,
                    "modes": modes,
                    "ranges": count,
                    "reference_modes": reference_modes,
                    "reference_ranges": reference_count,
                    "frequency": frequency,
                    "scope": "convergence within declared one-way model, not its error versus full Helmholtz",
                    **complex_metrics(fields[i][:, keep], reference[i, :, ::ratio][:, keep]),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
        (output / "audit.json").write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
