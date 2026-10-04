"""Posthoc target-family PCA-prior rescue, not OOD generalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_rank256_inference import infer
from .audit_waveguide_spectral_taper_shift import metric_rows, shifted_environment


def run(config: dict, output: Path) -> None:
    source = json.loads(Path(config["source_config"]).read_text())
    ood = json.loads(Path(config["ood_config"]).read_text())
    ood_rows = json.loads(Path(config["ood_results"]).read_text())
    if config["systems"] != ood["systems"] or len(ood_rows) != 12 * len(config["systems"]):
        raise ValueError("must use all already-seen taper10 systems")
    if (config["rank"], config["terms"], config["taper_scale"],
        config["coarse_steps"], config["fine_steps"]) != (256, 16, 10.0, 5, 5):
        raise ValueError("matched-prior control parameters must stay fixed")
    train_ids = list(range(config["train_seed_start"],
                           config["train_seed_start"] + config["train_count"]))
    if set(train_ids) & set(config["systems"]):
        raise ValueError("training and evaluation systems overlap")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "ood_results_sha256": digest(config["ood_results"]),
        "implementation_sha256": digest(__file__),
        "train_ids": train_ids,
    }, indent=2))
    labels = []
    for system in train_ids:
        spec = shifted_environment(
            system, config["terms"], config["perturbation_rms_m_s"],
            config["taper_scale"],
        )
        geometry = generator_geometry(spec, 64, 32, modes=32, quadrature_nodes=160)
        labels.append(true_projection(spec, geometry))
    training = np.array(labels)
    np.savez_compressed(output / "training_labels.npz",
                        systems=np.array(train_ids), target32=training)
    (output / "training_labels_sha256.json").write_text(json.dumps({
        "sha256": digest(output / "training_labels.npz"),
        "systems": len(train_ids), "target_shape": list(training.shape),
    }, indent=2))
    mean, factor = fit_prior(training, config["rank"])
    fine_mean = resample_generator(mean, source["decoder_cells"])
    fine_factor = resample_generator(factor, source["decoder_cells"])
    rows, histories = [], []
    for system in config["systems"]:
        with np.load(Path(config["ood_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
            observed, context_ranges = data["observed"], data["context_ranges_m"]
        beta, history = infer(
            spec, observed, context_ranges, source, mean, factor,
            config["coarse_steps"], config["fine_steps"],
        )
        histories.extend({"system": system, **item} for item in history)
        coarse = generator_geometry(spec, 64, 32, modes=32, quadrature_nodes=160)
        true_beta = true_projection(spec, coarse)
        coordinates = np.linalg.lstsq(
            factor.reshape(-1, config["rank"]), (true_beta - mean).ravel(), rcond=None
        )[0]
        projected = fine_mean + np.einsum("rdk,k->rd", fine_factor, coordinates)
        np.savez_compressed(output / f"system{system}_code.npz", beta=beta)
        for frequency in source["query_frequencies"]:
            with np.load(Path(ood["output_data"]) /
                         f"system{system}_f{frequency}.npz") as data:
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
            for method, candidate in [
                ("matched_taper10_context", beta),
                ("matched_taper10_true_medium_projection_oracle", projected),
            ]:
                prediction = propagate_code(
                    spec, frequency, ranges, source["decoder_cells"],
                    source["decoder_modes"], source["teacher_quadrature_nodes"],
                    source["eigh_device"], source["eigh_batch"], candidate,
                )
                rows.extend(metric_rows(
                    prediction, truth, ranges, depths, spec, source, system,
                    frequency, method,
                ))
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for method in ["matched_taper10_context",
                   "matched_taper10_true_medium_projection_oracle"]:
        for frequency in source["query_frequencies"]:
            for variant in source["source_variants"]:
                sample = [r for r in rows if r["method"] == method
                          and r["frequency"] == frequency and r["source"] == variant]
                summary.append({
                    "method": method, "frequency": frequency, "source": variant,
                    "systems": len(sample),
                    "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                    "mean_phase_rad": float(np.mean([
                        r["energy_weighted_phase_rmse_rad"] for r in sample
                    ])),
                    "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                })
    old = {
        (frequency, variant): float(np.mean([
            r["complex_rel_l2"] for r in ood_rows
            if r["method"] == "rank256_context" and r["frequency"] == frequency
            and r["source"] == variant
        ]))
        for frequency in source["query_frequencies"]
        for variant in source["source_variants"]
    }
    checked = [r for r in summary if r["method"] == "matched_taper10_context"
               and r["frequency"] == 800]
    gate = config["gate_800hz_both_sources"]
    comparisons = []
    for row in checked:
        previous = old[(800, row["source"])]
        comparisons.append({"source": row["source"],
                            "raw_improvement_vs_fixed_taper5":
                            float(1 - row["mean_raw"] / previous)})
    checks = {
        "mean_raw": all(r["mean_raw"] <= gate["mean_raw_at_most"] for r in checked),
        "mean_phase": all(r["mean_phase_rad"] <= gate["mean_phase_rad_at_most"]
                          for r in checked),
        "improvement": all(
            r["raw_improvement_vs_fixed_taper5"]
            >= gate["relative_raw_improvement_vs_fixed_taper5_at_least"]
            for r in comparisons
        ),
    }
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "comparisons": comparisons,
        "gate_checks": checks, "all_gates_pass": all(checks.values()),
        "scope": config["scope"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
