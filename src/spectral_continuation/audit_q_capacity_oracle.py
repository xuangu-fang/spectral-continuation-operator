"""Posthoc q-PCA rank capacity via TRUE-TEST-MEDIUM projection oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_taper_shift import metric_rows


def run(config: dict, output: Path) -> None:
    if (config["ranks"], config["systems"], config["frequency_hz"],
        config["coarse_cells"], config["fine_decoder_cells"],
        config["depth_terms"], config["projection_quadrature_nodes"]) != (
            [255, 384, 512, 768], list(range(20268000, 20268016)),
            800, 64, 256, 32, 160):
        raise ValueError("frozen capacity diagnostic settings changed")
    source = json.loads(Path(config["source_config"]).read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    labels = []
    for path in config["training_labels"]:
        with np.load(path) as data:
            labels.append(data[config["training_array_key"]].astype(float))
    training = np.concatenate(labels)
    if training.shape != (1024, 64, 32):
        raise ValueError("unexpected q training shape")
    mean = training.mean(axis=0)
    centered = (training - mean).reshape(1024, -1)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    basis = right[:max(config["ranks"])]
    variance = np.cumsum(singular**2) / np.sum(singular**2)
    rows, residuals = [], []
    manifest = {
        "scope": config["scope"], "implementation_sha256": digest(__file__),
        "source_config_sha256": digest(config["source_config"]),
        "training_label_sha256": {str(p): digest(p) for p in config["training_labels"]},
        "truth_sha256": {},
    }
    for system in config["systems"]:
        truth_path = Path(config["truth_dir"]) / f"system{system}_f800.npz"
        manifest["truth_sha256"][str(truth_path)] = digest(truth_path)
        with np.load(truth_path) as data:
            spec = json.loads(str(data["spec"]))
            truth = data["modal_coefficients"]
            ranges, depths = data["ranges_m"], data["depths_m"]
        geometry = generator_geometry(
            spec, config["coarse_cells"], config["depth_terms"],
            modes=32, quadrature_nodes=config["projection_quadrature_nodes"],
        )
        true_q = true_projection(spec, geometry)
        difference = (true_q - mean).ravel()
        all_coordinates = basis @ difference
        for rank in config["ranks"]:
            projected = mean + (all_coordinates[:rank] @ basis[:rank]).reshape(mean.shape)
            residuals.append({
                "system": system, "rank": rank,
                "q_relative_coefficient_projection_residual": float(
                    np.linalg.norm(true_q - projected) / np.linalg.norm(true_q)
                ),
            })
            beta = resample_generator(projected, config["fine_decoder_cells"])
            prediction = propagate_code(
                spec, config["frequency_hz"], ranges,
                config["fine_decoder_cells"], source["decoder_modes"],
                source["teacher_quadrature_nodes"], source["eigh_device"],
                source["eigh_batch"], beta,
            )
            for case in metric_rows(
                prediction, truth, ranges, depths, spec, source, system,
                config["frequency_hz"], f"q_pca_rank{rank}_true_medium_oracle",
            ):
                rows.append({"rank": rank, "information": "TRUE_TEST_MEDIUM_ORACLE", **case})
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "projection_residuals.json").write_text(json.dumps(residuals, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for rank in config["ranks"]:
        for variant in source["source_variants"]:
            sample = [r for r in rows if r["rank"] == rank and r["source"] == variant]
            if len(sample) != len(config["systems"]):
                raise ValueError("incomplete oracle rows")
            summary.append({
                "rank": rank, "source": variant, "systems": len(sample),
                "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                "mean_phase_rad": float(np.mean([
                    r["energy_weighted_phase_rmse_rad"] for r in sample
                ])),
                "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                "mean_medium_projection_residual": float(np.mean([
                    r["q_relative_coefficient_projection_residual"]
                    for r in residuals if r["rank"] == rank
                ])),
                "training_variance_fraction": float(variance[rank - 1]),
                "per_environment_code_float64_values": rank,
                "shared_basis_float64_values": int(mean.size * (rank + 1)),
            })
    (output / "summary.json").write_text(json.dumps({
        "oracle_summary": summary, "scope": config["scope"],
        "training_fields": len(training),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
