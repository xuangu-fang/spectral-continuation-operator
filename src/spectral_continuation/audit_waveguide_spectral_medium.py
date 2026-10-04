"""Frozen richer-medium stress test for a coherent physical-code representation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_frequency_scaling import backend_audit, modal_raw, propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_new_systems import context_data, frozen_inference
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, spectral_environment


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_spec(system, config):
    return spectral_environment(
        system, config["spectral_terms"], config["perturbation_rms_m_s"]
    )


def fit_training_priors(config, output):
    train_ids = list(range(config["train_seed_start"], config["train_seed_start"] + config["train_count"]))
    labels = {16: [], 32: []}
    for system in train_ids:
        spec = make_spec(system, config)
        for terms, values in labels.items():
            geometry = generator_geometry(spec, 64, terms, modes=32, quadrature_nodes=160)
            values.append(true_projection(spec, geometry))
    targets = {terms: np.array(values) for terms, values in labels.items()}
    np.savez_compressed(
        output / "training_labels.npz",
        systems=np.array(train_ids),
        target16=targets[16],
        target32=targets[32],
    )
    priors, statistics = {}, {}
    for terms, values in targets.items():
        mean, factor = fit_prior(values, config["rank"])
        priors[terms] = (mean, factor)
        reconstruction = mean + np.einsum(
            "rdk,bk->brd",
            factor,
            np.linalg.lstsq(
                factor.reshape(-1, config["rank"]),
                (values - mean).reshape(len(values), -1).T,
                rcond=None,
            )[0].T,
        )
        statistics[str(terms)] = {
            "shape": list(values.shape),
            "pca128_variance_fraction": float(
                1 - np.sum((values - reconstruction) ** 2)
                / np.sum((values - mean) ** 2)
            ),
        }
    np.savez_compressed(
        output / "training_priors.npz",
        mean16=priors[16][0],
        factor16=priors[16][1],
        mean32=priors[32][0],
        factor32=priors[32][1],
    )
    (output / "training_manifest.json").write_text(
        json.dumps(
            {
                "train_seed_start": config["train_seed_start"],
                "train_count": config["train_count"],
                "no_validation_overlap": not (set(train_ids) & set(config["systems"])),
                "training_labels_sha256": digest(output / "training_labels.npz"),
                "training_priors_sha256": digest(output / "training_priors.npz"),
                "statistics": statistics,
            },
            indent=2,
        )
    )
    return priors


def numerical_audit(config, data_root):
    rows = []
    for system in config["audit_systems"]:
        spec = make_spec(system, config)
        ranges = (np.arange(513) + 0.37) / 513 * spec["range_m"]
        roi = ranges >= config["roi_min_range_m"]
        for frequency in config["query_frequencies"]:
            lower = propagate_code(
                spec, frequency, ranges, config["teacher_cells"], config["teacher_modes"],
                config["teacher_quadrature_nodes"], config["eigh_device"], config["eigh_batch"]
            )
            upper = propagate_code(
                spec, frequency, ranges, config["audit_cells"], config["audit_modes"],
                config["audit_quadrature_nodes"], config["eigh_device"], config["eigh_batch"]
            )
            error = modal_raw(lower[roi], upper[roi])
            row = {"system": system, "frequency": frequency, "raw_by_source": error.tolist()}
            rows.append(row)
            (data_root / "convergence.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({"audit": row}), flush=True)
            if np.max(error) >= config["audit_gate"]:
                raise ValueError(f"failed numerical audit {row}")
    return rows


def evaluate_system(system, config, output, data_root, priors):
    spec = make_spec(system, config)
    context_ranges, observed = context_data(spec, config)
    started = time.perf_counter()
    branches, diagnostics = frozen_inference(
        spec, config, context_ranges, observed, priors[16], priors[32]
    )
    inference_seconds = time.perf_counter() - started
    code_path = output / f"system{system}_codes.npz"
    np.savez_compressed(
        code_path,
        observed=observed.astype(np.complex64),
        context_ranges_m=context_ranges,
        beta16=branches["prior16"],
        beta32=branches["prior32"],
        spec=json.dumps(spec),
    )
    ranges = (
        (np.arange(config["query_range_count"]) + 0.37)
        / config["query_range_count"]
        * spec["range_m"]
    )
    depths = np.linspace(0.1, spec["depth_m"] - 0.1, config["query_depth_count"])
    roi = ranges >= config["roi_min_range_m"]
    methods = dict(branches)
    for terms in [32, 64]:
        geometry = generator_geometry(
            spec, config["decoder_cells"], terms, modes=config["decoder_modes"],
            quadrature_nodes=config["teacher_quadrature_nodes"]
        )
        methods[f"true_medium{terms}_oracle"] = true_projection(spec, geometry)
    methods["full_medium_oracle"] = None
    rows, generation = [], []
    for frequency in config["query_frequencies"]:
        start = time.perf_counter()
        truth = propagate_code(
            spec, frequency, ranges, config["teacher_cells"], config["teacher_modes"],
            config["teacher_quadrature_nodes"], config["eigh_device"], config["eigh_batch"]
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
        target = depth_basis(depths, spec["depth_m"], truth.shape[1]) @ truth.transpose(
            1, 0, 2
        ).reshape(truth.shape[1], -1)
        target = target.reshape(len(depths), len(ranges), len(config["source_variants"]))
        for method, beta in methods.items():
            start = time.perf_counter()
            prediction = propagate_code(
                spec, frequency, ranges, config["decoder_cells"], config["decoder_modes"],
                config["teacher_quadrature_nodes"], config["eigh_device"], config["eigh_batch"],
                beta
            )
            seconds = time.perf_counter() - start
            field = depth_basis(
                depths, spec["depth_m"], prediction.shape[1]
            ) @ prediction.transpose(1, 0, 2).reshape(prediction.shape[1], -1)
            field = field.reshape(len(depths), len(ranges), len(config["source_variants"]))
            modal = modal_raw(prediction[roi], truth[roi])
            for source_index, source in enumerate(config["source_variants"]):
                rows.append(
                    {
                        "system": system,
                        "method": method,
                        "oracle": "oracle" in method,
                        "frequency": frequency,
                        "source": source,
                        "modal_parseval_raw": float(modal[source_index]),
                        "query_only_seconds": seconds,
                        **complex_metrics(field[:, roi, source_index], target[:, roi, source_index]),
                    }
                )
    return rows, generation, diagnostics, {
        "system": system,
        "path": str(code_path),
        "sha256": digest(code_path),
        "context_inference_seconds": inference_seconds,
    }


def summarize(rows, config):
    summary = []
    for method in ["prior16", "prior32", "true_medium32_oracle", "true_medium64_oracle", "full_medium_oracle"]:
        for frequency in config["query_frequencies"]:
            for source in config["source_variants"]:
                sample = [
                    row for row in rows
                    if row["method"] == method and row["frequency"] == frequency
                    and row["source"] == source
                ]
                summary.append(
                    {
                        "method": method, "frequency": frequency, "source": source,
                        "systems": len(sample),
                        **{
                            metric: float(np.mean([row[metric] for row in sample]))
                            for metric in [
                                "complex_rel_l2", "energy_weighted_phase_rmse_rad",
                                "amplitude_rel_l2", "modal_parseval_raw",
                            ]
                        },
                    }
                )
    return summary


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(config["output_data"])
    data_root.mkdir(parents=True, exist_ok=False)
    train_ids = set(range(config["train_seed_start"], config["train_seed_start"] + config["train_count"]))
    if train_ids & (set(config["systems"]) | set(config["audit_systems"])):
        raise ValueError("train, validation and audit systems must be disjoint")
    if set(config["systems"]) & set(config["audit_systems"]):
        raise ValueError("validation/audit collision")
    torch.set_num_threads(2)
    for directory in [output, data_root]:
        (directory / "config.json").write_text(json.dumps(config, indent=2))
    priors = fit_training_priors(config, output)
    tracked = [
        Path(__file__), Path(__file__).with_name("coherent_waveguide_data.py"),
        Path(__file__).with_name("audit_waveguide_generator_identification.py"),
        Path(__file__).with_name("audit_waveguide_new_systems.py"),
        Path(__file__).with_name("audit_waveguide_spectral_transport.py"),
        Path(__file__).with_name("audit_waveguide_generator_prior.py"),
        Path(__file__).with_name("audit_waveguide_trajectory_newton.py"),
        Path(__file__).with_name("audit_waveguide_frequency_scaling.py"),
        Path(__file__).with_name("coarse_waveguide_operator.py"),
        output / "training_priors.npz",
    ]
    (output / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "registered_utc": datetime.now(timezone.utc).isoformat(),
                "registration": "After training-only prior fit, BEFORE new audit and validation truth",
                "tracked_sha256": {str(path): digest(path) for path in tracked},
                "config": config,
                "backend": backend_audit(config),
            },
            indent=2,
        )
    )
    numerical_audit(config, data_root)
    rows, generation, histories, code_records = [], [], [], []
    for system in config["systems"]:
        evaluated, produced, diagnostics, code = evaluate_system(
            system, config, output, data_root, priors
        )
        rows.extend(evaluated)
        generation.extend(produced)
        histories.extend({"system": system, **item} for item in diagnostics)
        code_records.append(code)
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        (output / "code_manifest.json").write_text(json.dumps(code_records, indent=2))
        (data_root / "generation.json").write_text(json.dumps(generation, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "summary.json").write_text(json.dumps(summarize(rows, config), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
