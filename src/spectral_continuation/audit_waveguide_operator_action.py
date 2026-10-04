"""Frozen generator representations queried by an entire source subspace."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from .audit_waveguide_generator_identification import (
    generator_geometry,
    latent_from_coefficients,
    true_projection,
)
from .coherent_waveguide_data import environment, local_generator, quadrature


def propagate_actions(edges, initial, ranges, spectral_cell, attenuation):
    """Multiple right-hand sides. spectral_cell returns wavenumbers and vectors."""
    current = np.asarray(initial, complex).copy()
    if current.ndim != 2:
        raise ValueError("source columns required")
    if np.any(ranges < edges[0]) or np.any(ranges > edges[-1]):
        raise ValueError("out-of-domain query")
    assignment = np.clip(np.searchsorted(edges, ranges, side="right") - 1, 0, len(edges) - 2)
    result = np.zeros((len(ranges), *current.shape), complex)
    for cell, (lo, hi) in enumerate(itertools.pairwise(edges)):
        root, vectors = spectral_cell(cell)
        rate = 1j * root - attenuation
        modal = vectors.T @ current
        for index in np.flatnonzero(assignment == cell):
            result[index] = vectors @ (np.exp((ranges[index] - lo) * rate)[:, None] * modal)
        current = vectors @ (np.exp((hi - lo) * rate)[:, None] * modal)
    return result


def teacher(spec, config, ranges, cells, modes):
    z, weights, basis = quadrature(spec, modes, config["quadrature_nodes"])
    edges = np.linspace(0, spec["range_m"], cells + 1)
    initial = np.eye(modes, config["source_modes"])

    def spectral(cell):
        return local_generator(
            spec, config["frequency"], (edges[cell] + edges[cell + 1]) / 2, z, weights, basis
        )

    return propagate_actions(edges, initial, ranges, spectral, spec["attenuation_per_m"])


def prediction(beta, spec, config, ranges):
    geometry = generator_geometry(spec, len(beta), beta.shape[1], modes=64)
    initial = np.eye(64, config["source_modes"])
    latent, clipping = latent_from_coefficients(beta, initial, geometry)
    if clipping:
        raise ValueError("no silently clipped operator prediction")
    matrices = (2 * np.pi * config["frequency"]) ** 2 * latent["mass"] - np.diag(latent["kz2"])
    values, vectors = np.linalg.eigh(matrices)
    roots = np.sqrt(values.astype(complex))
    return propagate_actions(
        latent["edges"],
        initial,
        ranges,
        lambda cell: (roots[cell], vectors[cell]),
        latent["attenuation"],
    )


def action_metrics(predicted, truth):
    if predicted.shape[0] != truth.shape[0] or predicted.shape[-1] != truth.shape[-1]:
        raise ValueError("query/source coordinates must agree")
    if predicted.shape[1] > truth.shape[1]:
        raise ValueError("truth must resolve all prediction output modes")
    padded = np.zeros_like(truth)
    padded[:, : predicted.shape[1]] = predicted
    target = truth.reshape(-1, truth.shape[-1])
    difference = (padded - truth).reshape(target.shape)
    source_errors = np.linalg.norm(difference, axis=0) / np.linalg.norm(target, axis=0)
    gram, error_gram = target.conj().T @ target, difference.conj().T @ difference
    spectrum = eigvalsh(gram)
    condition = float(spectrum[-1] / max(spectrum[0], 1e-300))
    well_conditioned = bool(spectrum[0] > spectrum[-1] * 1e-10)
    worst = (
        float(np.sqrt(max(float(eigvalsh(error_gram, gram)[-1]), 0))) if well_conditioned else None
    )
    return {
        "source_macro_raw": float(source_errors.mean()),
        "source_max_raw": float(source_errors.max()),
        "per_source_raw": source_errors.tolist(),
        "frobenius_relative": float(np.linalg.norm(difference) / np.linalg.norm(target)),
        "worst_relative_in_source_subspace": worst,
        "truth_gram_condition": condition,
        "worst_case_valid": well_conditioned,
    }


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(config["data_output"])
    data_root.mkdir(parents=True, exist_ok=False)
    for directory in [output, data_root]:
        (directory / "config.json").write_text(json.dumps(config, indent=2))
    records = []
    for system in config["audit_systems"]:
        spec = environment(system)
        ranges = (np.arange(config["range_count"]) + 0.37) / config["range_count"] * spec["range_m"]
        roi = ranges >= config["roi_min_range_m"]
        lower = teacher(spec, config, ranges, config["teacher_cells"], config["teacher_modes"])
        higher = teacher(spec, config, ranges, config["audit_cells"], config["audit_modes"])
        metric = action_metrics(lower[roi], higher[roi])
        records.append({"system": system, **metric})
        np.savez_compressed(
            data_root / f"audit{system}.npz",
            coarse=lower.astype(np.complex64),
            fine=higher.astype(np.complex64),
            ranges_m=ranges,
            spec=json.dumps(spec),
        )
        (data_root / "convergence.json").write_text(json.dumps(records, indent=2))
        print(
            json.dumps(
                {
                    "audit_system": system,
                    "raw": metric["source_macro_raw"],
                    "worst": metric["worst_relative_in_source_subspace"],
                }
            ),
            flush=True,
        )
        if (
            not metric["worst_case_valid"]
            or metric["worst_relative_in_source_subspace"] >= config["convergence_gate"]
        ):
            raise ValueError(
                "source-subspace numerical convergence gate failed; artifacts preserved"
            )
    with np.load(config["old_coefficients"]) as data:
        old_codes, systems = data["coefficients"], data["systems"].tolist()
    rows, generation = [], []
    for system in config["systems"]:
        spec = environment(system)
        ranges = (np.arange(config["range_count"]) + 0.37) / config["range_count"] * spec["range_m"]
        roi = ranges >= config["roi_min_range_m"]
        start = time.perf_counter()
        truth = teacher(spec, config, ranges, config["teacher_cells"], config["teacher_modes"])
        path = data_root / f"system{system}_operator.npz"
        np.savez_compressed(
            path,
            transfer=truth.astype(np.complex64),
            ranges_m=ranges,
            spec=json.dumps(spec),
            source_mode_indices=np.arange(config["source_modes"]),
            frequency_hz=config["frequency"],
            phase_convention="exp(-i omega t); real orthonormal sin((m+1/2)pi*z/H) source columns",
            evidence="original development systems, new source-subspace action truth, one-way approximation",
        )
        generation.append(
            {
                "system": system,
                "path": str(path),
                "shape": list(truth.shape),
                "seconds": time.perf_counter() - start,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        (data_root / "generation.json").write_text(json.dumps(generation, indent=2))
        methods = {"prior16_cells64": old_codes[systems.index(system)]}
        for terms in [16, 32]:
            with np.load(
                Path(config[f"continuous{terms}"]) / f"system{system}_cells256_step5.npz"
            ) as data:
                methods[f"prior{terms}_cells256"] = data["beta"]
        methods["true_medium32_oracle"] = true_projection(spec, generator_geometry(spec, 256, 32))
        for method, beta in methods.items():
            start = time.perf_counter()
            predicted = prediction(beta, spec, config, ranges)
            seconds = time.perf_counter() - start
            row = {
                "system": system,
                "method": method,
                "query_only_seconds": seconds,
                **action_metrics(predicted[roi], truth[roi]),
            }
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "per_source_raw"}), flush=True)
        (output / "results.json").write_text(json.dumps(rows, indent=2))
    summary = []
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        summary.append(
            {
                "method": method,
                "systems": len(selected),
                **{
                    key: float(np.mean([r[key] for r in selected]))
                    for key in [
                        "source_macro_raw",
                        "source_max_raw",
                        "frobenius_relative",
                        "query_only_seconds",
                    ]
                },
                "worst_relative_system_mean": float(
                    np.mean([r["worst_relative_in_source_subspace"] for r in selected])
                )
                if all(r["worst_case_valid"] for r in selected)
                else None,
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
