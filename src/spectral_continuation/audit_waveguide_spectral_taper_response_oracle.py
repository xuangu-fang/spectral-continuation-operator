"""Explicitly target-leaking response oracle for the fixed taper-shift PCA space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import propagate_code, source_columns
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_taper_shift import metric_rows
from .audit_waveguide_trajectory_newton import newton_step
from .coherent_waveguide_data import quadrature


def run(config: dict, output: Path) -> None:
    source = json.loads(Path(config["source_config"]).read_text())
    ood = json.loads(Path(config["ood_config"]).read_text())
    ood_rows = json.loads(Path(config["ood_results"]).read_text())
    if config["systems"] != ood["systems"] or len(ood_rows) != 12 * len(config["systems"]):
        raise ValueError("must evaluate the complete already-used OOD cohort")
    if (config["fit_cells"], config["fit_modes"], config["rank"], config["steps"]) != (
        256, 64, 256, 5
    ):
        raise ValueError("the posthoc oracle settings are frozen")
    if config["target_frequency"] != 800:
        raise ValueError("only the specified target frequency may be used")
    with np.load(config["training_labels"]) as data:
        mean, factor = fit_prior(data["target32"].astype(float), config["rank"])
    fine_mean = resample_generator(mean, config["fit_cells"])
    fine_factor = resample_generator(factor, config["fit_cells"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "ood_results_sha256": digest(config["ood_results"]),
        "training_labels_sha256": digest(config["training_labels"]),
        "implementation_sha256": digest(__file__),
        "systems": config["systems"],
    }, indent=2))
    rows, histories = [], []
    for system in config["systems"]:
        with np.load(Path(config["ood_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
        coarse = generator_geometry(spec, 64, mean.shape[1], modes=32)
        true_beta = true_projection(spec, coarse)
        coordinates = np.linalg.lstsq(
            factor.reshape(-1, config["rank"]), (true_beta - mean).ravel(), rcond=None
        )[0]
        beta = fine_mean + np.einsum("rdk,k->rd", fine_factor, coordinates)
        geometry = generator_geometry(
            spec, config["fit_cells"], mean.shape[1],
            modes=config["fit_modes"], quadrature_nodes=config["fit_quadrature_nodes"],
        )
        z, weights, basis = quadrature(
            spec, config["fit_modes"], config["fit_quadrature_nodes"]
        )
        initial = source_columns(spec, z, weights, basis)[:, 0]
        edges = geometry["edges"]
        observed = propagate_code(
            spec, config["target_frequency"], edges,
            source["teacher_cells"], config["fit_modes"],
            source["teacher_quadrature_nodes"], source["eigh_device"],
            source["eigh_batch"],
        )[None, :, :, 0]
        roi = edges >= source["roi_min_range_m"]
        options = {key: source[key] for key in
                   ["ridge_relative", "max_coefficient_step", "line_search"]}
        candidates = [(0, beta.copy())]
        for step in range(config["steps"]):
            beta, detail = newton_step(
                beta, fine_factor, initial, geometry,
                [config["target_frequency"]], observed, roi, options,
            )
            candidates.append((step + 1, beta.copy()))
            histories.append({"system": system, "step": step + 1, **detail})
        with np.load(Path(ood["output_data"]) /
                     f"system{system}_f{config['target_frequency']}.npz") as data:
            truth = data["modal_coefficients"]
            ranges, depths = data["ranges_m"], data["depths_m"]
        for step, candidate in candidates:
            prediction = propagate_code(
                spec, config["target_frequency"], ranges,
                source["decoder_cells"], source["decoder_modes"],
                source["teacher_quadrature_nodes"], source["eigh_device"],
                source["eigh_batch"], candidate,
            )
            rows.extend({**row, "step": step} for row in metric_rows(
                prediction, truth, ranges, depths, spec, source, system,
                config["target_frequency"], "response_optimized_rank256_oracle",
            ))
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for step in range(config["steps"] + 1):
        for variant in source["source_variants"]:
            sample = [r for r in rows if r["step"] == step and r["source"] == variant]
            summary.append({
                "step": step, "source": variant, "systems": len(sample),
                "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                "mean_phase_rad": float(np.mean([
                    r["energy_weighted_phase_rmse_rad"] for r in sample
                ])),
                "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
            })
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "scope": config["scope"],
        "interpretation": "Posthoc target-800Hz broad-field oracle. It tests a finite local response fit inside the same frozen PCA256 subspace, not its global best capacity and not an inference method."
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
