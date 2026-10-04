"""Second-domain physical-code compression, with explicit separability caveat."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

from .audit_helmholtz_generator_identification import (
    estimate_slowness,
    known_terms,
    laplacian,
    query,
)
from .audit_waveguide_generator_prior import fit_prior
from .coherent_helmholtz_data import environment, sample_environment
from .coherent_representation import complex_metrics


def tensor_readout(values, axis, target, rank):
    """Per-instance SVD of speed minus background, not of a complex field."""
    left, singular, right = np.linalg.svd(values - 1, full_matrices=False)
    left = CubicSpline(axis, left[:, :rank], axis=0)(target)
    right = CubicSpline(axis, right[:rank].T, axis=0)(target)
    return 1 + (left * singular[:rank]) @ right.T


def spline_readout(values, axis, target):
    """Same tensor-product cubic extrapolation used by separated factors."""
    first = CubicSpline(axis, values, axis=0)(target)
    return CubicSpline(axis, first, axis=1)(target)


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    axis = np.linspace(0, 1, config["context_grid"])
    interior = axis[1:-1]
    training_ids = list(
        range(
            config["training_seed_start"], config["training_seed_start"] + config["training_count"]
        )
    )
    if set(training_ids) & set(config["systems"]):
        raise ValueError("training/development overlap")
    targets, training_manifest = [], []
    for system in training_ids:
        spec = environment(system)
        speed, _, _ = sample_environment(spec, interior)
        targets.append(speed)
        training_manifest.append({"system": system, "spec": spec})
    mean, factor = fit_prior(np.array(targets), config["pca_rank"])
    inverse = np.linalg.pinv(factor.reshape(-1, config["pca_rank"]))
    np.savez_compressed(
        output / "training_prior.npz",
        mean=mean,
        factor=factor,
        targets=np.array(targets),
        systems=training_ids,
    )
    (output / "training_manifest.json").write_text(
        json.dumps(
            {
                "source": "new training medium labels from existing generator; no new training fields",
                "systems": training_manifest,
                "prior_sha256": hashlib.sha256(
                    (output / "training_prior.npz").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as data:
            spec, fields, frequencies, fine, actual_speed = (
                json.loads(str(data["spec"])),
                data["field"],
                data["frequencies"],
                data["axis"],
                data["speed"],
            )
        take = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        stride = (len(fine) - 1) // (len(axis) - 1)
        if not np.array_equal(fine[::stride], axis):
            raise ValueError("nested observation coordinates required")
        start = time.perf_counter()
        q, _ = estimate_slowness(
            fields[take, ::stride, ::stride], frequencies[take], spec, axis, "sine_spectral", 0
        )
        inverse_seconds = time.perf_counter() - start
        # q=1 at the outer boundary is a placeholder, not a measured speed label.
        estimated_speed = 1 / np.sqrt(q[1:-1, 1:-1])
        candidates = [
            (
                "uncompressed_original",
                False,
                RectBivariateSpline(axis, axis, q)(fine, fine),
                q.size,
                0,
                0.0,
            )
        ]
        for oracle, speed in [
            (False, estimated_speed),
            (True, actual_speed[::stride, ::stride][1:-1, 1:-1]),
        ]:
            start = time.perf_counter()
            decoded = spline_readout(speed, interior, fine)
            candidates.append(
                (
                    "uncompressed_interior_speed",
                    oracle,
                    1 / np.clip(decoded, 0.4, 1) ** 2,
                    speed.size,
                    0,
                    time.perf_counter() - start,
                )
            )
            start = time.perf_counter()
            code = inverse @ (speed - mean).ravel()
            reconstruction = mean + np.einsum("ijk,k->ij", factor, code)
            decoded = spline_readout(reconstruction, interior, fine)
            seconds = time.perf_counter() - start
            candidates.append(
                (
                    "pca128_speed",
                    oracle,
                    1 / np.clip(decoded, 0.4, 1) ** 2,
                    len(code),
                    mean.size + factor.size,
                    seconds,
                )
            )
            for rank in config["separable_ranks"]:
                start = time.perf_counter()
                decoded = tensor_readout(speed, interior, fine, rank)
                seconds = time.perf_counter() - start
                candidates.append(
                    (
                        f"separable_speed_rank{rank}",
                        oracle,
                        1 / np.clip(decoded, 0.4, 1) ** 2,
                        2 * len(interior) * rank + rank,
                        0,
                        seconds,
                    )
                )
        loss, source = known_terms(spec, fine)
        lap = laplacian(len(fine))
        keep = (fine >= 0.18) & (fine <= 0.82)
        for method, oracle, reconstructed, per_system, shared, encoding_seconds in candidates:
            for frequency in config["query_frequencies"]:
                fi = int(np.flatnonzero(frequencies == frequency)[0])
                start = time.perf_counter()
                prediction = query(np.clip(reconstructed, 1, 6.25), source, loss, frequency, lap)
                seconds = time.perf_counter() - start
                rows.append(
                    {
                        "system": system,
                        "method": method,
                        "oracle": oracle,
                        "frequency": frequency,
                        "per_system_real_parameters": per_system,
                        "shared_real_parameters": shared,
                        "identification_seconds": inverse_seconds if not oracle else None,
                        "compression_and_readout_seconds": encoding_seconds,
                        "pde_query_seconds": seconds,
                        **complex_metrics(
                            prediction[np.ix_(keep, keep)], fields[fi][np.ix_(keep, keep)]
                        ),
                    }
                )
            print(
                json.dumps(
                    {
                        "system": system,
                        "method": method,
                        "oracle": oracle,
                        "raw": float(
                            np.mean(
                                [
                                    r["complex_rel_l2"]
                                    for r in rows[-len(config["query_frequencies"]) :]
                                ]
                            )
                        ),
                    }
                ),
                flush=True,
            )
            (output / "results.json").write_text(json.dumps(rows, indent=2))
    summary = []
    for method, oracle in sorted({(r["method"], r["oracle"]) for r in rows}):
        selected = [r for r in rows if r["method"] == method and r["oracle"] == oracle]
        summary.append(
            {
                "method": method,
                "oracle": oracle,
                "systems": len(config["systems"]),
                "raw": float(np.mean([r["complex_rel_l2"] for r in selected])),
                "phase_rmse_rad": float(
                    np.mean([r["energy_weighted_phase_rmse_rad"] for r in selected])
                ),
                "per_system_real_parameters": selected[0]["per_system_real_parameters"],
                "shared_real_parameters": selected[0]["shared_real_parameters"],
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
