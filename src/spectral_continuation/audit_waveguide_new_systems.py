"""Frozen 16/32-depth physical-code comparison on new IID waveguide systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import (
    backend_audit,
    modal_raw,
    propagate_code,
)
from .audit_waveguide_generator_identification import (
    generator_geometry,
    normal_statistics,
    true_projection,
)
from .audit_waveguide_generator_prior import conditional_mean, fit_prior
from .audit_waveguide_trajectory_newton import newton_step
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context_data(spec, config):
    ranges = np.linspace(0, spec["range_m"], config["context_profiles"])
    observations = np.array(
        [
            propagate_code(
                spec, f, ranges, 2048, 32, 160, config["eigh_device"], config["eigh_batch"]
            )[:, :, 0]
            for f in config["context_frequencies"]
        ]
    )
    return ranges, observations


def frozen_inference(spec, config, ranges, observed, prior16, prior32):
    mean16, factor16 = prior16
    mean32, factor32 = prior32
    geometry16 = generator_geometry(spec, 64, 16)
    gram, rhs = normal_statistics(spec, observed, config["context_frequencies"], ranges, geometry16)
    beta16 = conditional_mean(gram, rhs, mean16, factor16, config["initial_prior_precision"])
    roi = ranges >= config["roi_min_range_m"]
    options = {
        "ridge_relative": config["ridge_relative"],
        "max_coefficient_step": config["max_coefficient_step"],
        "line_search": config["line_search"],
    }
    initial = observed[0, 0]
    diagnostics = []
    for step in range(config["old_prior_steps"]):
        beta16, row = newton_step(
            beta16,
            factor16,
            initial,
            geometry16,
            config["context_frequencies"],
            observed,
            roi,
            options,
        )
        diagnostics.append({"method": "shared_old16", "step": step + 1, **row})
    # Both branches start from the same context-fitted old16 code.
    old16_alpha = np.linalg.lstsq(
        factor16.reshape(-1, config["rank"]), (beta16 - mean16).ravel(), rcond=None
    )[0]
    np.testing.assert_allclose(
        beta16,
        mean16 + np.einsum("rdk,k->rd", factor16, old16_alpha),
        atol=1e-10,
        rtol=1e-8,
    )
    low32 = factor32[:, :16].reshape(-1, config["rank"])
    new32_alpha = np.linalg.lstsq(low32, (beta16 - mean32[:, :16]).ravel(), rcond=None)[0]
    branches = {}
    for name, mean, factor, alpha in [
        ("prior16", mean16, factor16, old16_alpha),
        ("prior32", mean32, factor32, new32_alpha),
    ]:
        fine_mean = resample_generator(mean, config["decoder_cells"])
        fine_factor = resample_generator(factor, config["decoder_cells"])
        beta = fine_mean + np.einsum("rdk,k->rd", fine_factor, alpha)
        geometry = generator_geometry(spec, config["decoder_cells"], mean.shape[1])
        indices = np.linspace(0, config["decoder_cells"], config["context_profiles"])
        if not np.array_equal(indices, indices.astype(int)):
            raise ValueError("context profile positions must be decoder cell boundaries")
        fine_options = {**options, "observation_indices": indices.astype(int)}
        for step in range(config["continuous_steps"]):
            beta, row = newton_step(
                beta,
                fine_factor,
                initial,
                geometry,
                config["context_frequencies"],
                observed,
                roi,
                fine_options,
            )
            diagnostics.append({"method": name, "step": step + 1, **row})
        branches[name] = beta
    return branches, diagnostics


def preflight(config, prior16, prior32):
    # Matched old-development implementation check; no method choice after this.
    system = 20261332
    path = Path(config["development_data"]) / f"system{system}_m32_r2049.npz"
    with np.load(path) as data:
        spec = json.loads(str(data["spec"]))
        frequencies = data["frequencies_hz"]
        indices = [int(np.flatnonzero(frequencies == f)[0]) for f in config["context_frequencies"]]
        take = np.linspace(0, len(data["ranges_m"]) - 1, config["context_profiles"], dtype=int)
        ranges = data["ranges_m"][take]
        observed = data["modal_coefficients"][indices][:, take]
    branches, _ = frozen_inference(spec, config, ranges, observed, prior16, prior32)
    expected = Path(config["depth32_run"]) / f"system{system}_cells256_step5.npz"
    with np.load(expected) as data:
        reference = data["beta"]
    error = float(np.max(np.abs(branches["prior32"] - reference)))
    if error > 1e-6:
        raise ValueError(f"old-development prior32 implementation mismatch {error}")
    return {
        "system": system,
        "max_coefficient_difference": error,
        "expected_path": str(expected),
        "expected_sha256": digest(expected),
    }


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(config["output_data"])
    data_root.mkdir(parents=True, exist_ok=False)
    if set(config["systems"]) & set(config["audit_systems"]):
        raise ValueError("evaluation and numerical audit systems overlap")
    torch.set_num_threads(2)
    with np.load(Path(config["training_run"]) / "training_statistics.npz") as data:
        prior16 = fit_prior(data["target"].astype(float), config["rank"])
    prior_path = Path(config["depth32_run"]) / "training_prior.npz"
    with np.load(prior_path) as data:
        prior32 = data["mean"], data["factor"]
    before = preflight(config, prior16, prior32)
    backend = backend_audit(
        {
            "audit_systems": config["audit_systems"],
            "eigh_device": config["eigh_device"],
            "eigh_batch": config["eigh_batch"],
        }
    )
    tracked = [
        Path(__file__),
        Path(config["training_run"]) / "training_statistics.npz",
        prior_path,
        Path(config["depth32_run"]) / "config.json",
    ]
    freeze = {
        "registered_utc": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "registration": "Before any new IID or numerical audit truth generation; method fixed on old8",
        "preflight": before,
        "backend": backend,
        "tracked_sha256": {str(p): digest(p) for p in tracked},
        "config": config,
    }
    (output / "freeze_manifest.json").write_text(json.dumps(freeze, indent=2))
    for directory in [output, data_root]:
        (directory / "config.json").write_text(json.dumps(config, indent=2))
    audit = []
    for system in config["audit_systems"]:
        spec = environment(system)
        ranges = (np.arange(513) + 0.37) / 513 * spec["range_m"]
        for frequency in config["query_frequencies"]:
            lower = propagate_code(
                spec,
                frequency,
                ranges,
                config["teacher_cells"],
                config["teacher_modes"],
                config["teacher_quadrature_nodes"],
                config["eigh_device"],
                config["eigh_batch"],
            )
            higher = propagate_code(
                spec,
                frequency,
                ranges,
                config["audit_cells"],
                config["audit_modes"],
                config["audit_quadrature_nodes"],
                config["eigh_device"],
                config["eigh_batch"],
            )
            roi = ranges >= config["roi_min_range_m"]
            errors = modal_raw(lower[roi], higher[roi])
            row = {"system": system, "frequency": frequency, "raw_by_source": errors.tolist()}
            audit.append(row)
            (data_root / "convergence.json").write_text(json.dumps(audit, indent=2))
            print(json.dumps({"audit": row}), flush=True)
            if max(errors) >= config["audit_gate"]:
                raise ValueError("new-system numerical audit gate failed")
    rows, generation, histories, code_records = [], [], [], []
    for system in config["systems"]:
        spec = environment(system)
        started = time.perf_counter()
        context_ranges, observed = context_data(spec, config)
        branches, history = frozen_inference(
            spec, config, context_ranges, observed, prior16, prior32
        )
        histories.extend({"system": system, **r} for r in history)
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        code_path = output / f"system{system}_codes.npz"
        np.savez_compressed(
            code_path,
            observed=observed.astype(np.complex64),
            context_ranges_m=context_ranges,
            beta16=branches["prior16"],
            beta32=branches["prior32"],
            spec=json.dumps(spec),
        )
        code_records.append(
            {
                "system": system,
                "path": str(code_path),
                "sha256": digest(code_path),
                "context_inference_seconds": time.perf_counter() - started,
            }
        )
        (output / "code_manifest.json").write_text(json.dumps(code_records, indent=2))
        ranges = (
            (np.arange(config["query_range_count"]) + 0.37)
            / config["query_range_count"]
            * spec["range_m"]
        )
        depths = np.linspace(0.1, spec["depth_m"] - 0.1, config["query_depth_count"])
        roi = ranges >= config["roi_min_range_m"]
        oracles = {}
        for terms in [32, 64]:
            geometry = generator_geometry(
                spec,
                config["decoder_cells"],
                terms,
                config["decoder_modes"],
                config["teacher_quadrature_nodes"],
            )
            oracles[f"true_medium{terms}_oracle"] = true_projection(spec, geometry)
        methods = {**branches, **oracles, "full_medium_oracle": None}
        for frequency in config["query_frequencies"]:
            start = time.perf_counter()
            truth = propagate_code(
                spec,
                frequency,
                ranges,
                config["teacher_cells"],
                config["teacher_modes"],
                config["teacher_quadrature_nodes"],
                config["eigh_device"],
                config["eigh_batch"],
            )
            path = data_root / f"system{system}_f{frequency}.npz"
            np.savez_compressed(
                path,
                modal_coefficients=truth.astype(np.complex64),
                ranges_m=ranges,
                depths_m=depths,
                frequency_hz=frequency,
                source_variants=config["source_variants"],
                spec=json.dumps(spec),
                phase_convention="exp(-i omega t), real Gaussian source, positive forward phase",
                evidence=config["scope"],
            )
            generation.append(
                {
                    "system": system,
                    "frequency": frequency,
                    "path": str(path),
                    "sha256": digest(path),
                    "seconds": time.perf_counter() - start,
                }
            )
            (data_root / "generation.json").write_text(json.dumps(generation, indent=2))
            target = depth_basis(depths, spec["depth_m"], truth.shape[1]) @ truth.transpose(
                1, 0, 2
            ).reshape(truth.shape[1], -1)
            target = target.reshape(len(depths), len(ranges), len(config["source_variants"]))
            for method, beta in methods.items():
                start = time.perf_counter()
                prediction = propagate_code(
                    spec,
                    frequency,
                    ranges,
                    config["decoder_cells"],
                    config["decoder_modes"],
                    config["teacher_quadrature_nodes"],
                    config["eigh_device"],
                    config["eigh_batch"],
                    beta,
                )
                seconds = time.perf_counter() - start
                field = depth_basis(
                    depths, spec["depth_m"], prediction.shape[1]
                ) @ prediction.transpose(1, 0, 2).reshape(prediction.shape[1], -1)
                field = field.reshape(len(depths), len(ranges), len(config["source_variants"]))
                modal = modal_raw(prediction[roi], truth[roi])
                for source_index, source in enumerate(config["source_variants"]):
                    row = {
                        "system": system,
                        "method": method,
                        "oracle": "oracle" in method,
                        "frequency": frequency,
                        "source": source,
                        "modal_parseval_raw": float(modal[source_index]),
                        "query_only_seconds": seconds,
                        **complex_metrics(
                            field[:, roi, source_index], target[:, roi, source_index]
                        ),
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in ("query_only_seconds", "amplitude_rel_l2")
                            }
                        ),
                        flush=True,
                    )
            (output / "results.json").write_text(json.dumps(rows, indent=2))
    groups = {}
    for row in rows:
        key = (row["method"], row["frequency"], row["source"])
        groups.setdefault(key, []).append(row)
    summary = [
        {
            "method": key[0],
            "frequency": key[1],
            "source": key[2],
            "systems": len(subset),
            **{
                name: float(np.mean([r[name] for r in subset]))
                for name in [
                    "complex_rel_l2",
                    "energy_weighted_phase_rmse_rad",
                    "amplitude_rel_l2",
                    "modal_parseval_raw",
                    "query_only_seconds",
                ]
            },
        }
        for key, subset in groups.items()
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
