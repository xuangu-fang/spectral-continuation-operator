"""Isolate two non-learning initialization choices under 1% complex context noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import modal_raw, propagate_code, source_columns
from .audit_waveguide_generator_identification import generator_geometry, normal_statistics
from .audit_waveguide_generator_prior import conditional_mean, fit_prior
from .audit_waveguide_spectral_context_robustness import digest, make_context
from .audit_waveguide_trajectory_newton import newton_step
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def infer_variant(spec, observed, ranges, source, mean, factor, config, variant):
    geometry = generator_geometry(spec, 64, mean.shape[1], modes=32, quadrature_nodes=160)
    if variant["medium_initialization"] == "normal_statistics":
        gram, rhs = normal_statistics(
            spec, observed, source["context_frequencies"], ranges, geometry
        )
        beta = conditional_mean(gram, rhs, mean, factor, source["initial_prior_precision"])
    elif variant["medium_initialization"] == "training_mean":
        beta = mean.copy()
    else:
        raise ValueError("unregistered medium initialization")
    if variant["source_initialization"] == "known_source":
        initial = source_columns(
            spec, geometry["z"], geometry["weights"], geometry["basis"]
        )[:, 0]
    elif variant["source_initialization"] == "noisy_observed":
        initial = observed[0, 0]
    else:
        raise ValueError("unregistered source initialization")
    options = {key: source[key] for key in
               ["ridge_relative", "max_coefficient_step", "line_search"]}
    roi = ranges >= source["roi_min_range_m"]
    history = []
    for step in range(config["coarse_steps"]):
        beta, diagnostic = newton_step(
            beta, factor, initial, geometry, source["context_frequencies"],
            observed, roi, options,
        )
        history.append({"stage": "coarse", "step": step + 1, **diagnostic})
    fine_mean = resample_generator(mean, source["decoder_cells"])
    fine_factor = resample_generator(factor, source["decoder_cells"])
    coordinates = np.linalg.lstsq(
        factor.reshape(-1, factor.shape[-1]), (beta - mean).ravel(), rcond=None
    )[0]
    beta = fine_mean + np.einsum("rdk,k->rd", fine_factor, coordinates)
    fine_geometry = generator_geometry(spec, source["decoder_cells"], mean.shape[1])
    fine_indices = np.linspace(0, source["decoder_cells"], source["context_profiles"])
    if not np.array_equal(fine_indices, fine_indices.astype(int)):
        raise ValueError("original context must align with fine propagation edges")
    fine_options = {**options, "observation_indices": fine_indices.astype(int)}
    for step in range(config["fine_steps"]):
        beta, diagnostic = newton_step(
            beta, fine_factor, initial, fine_geometry, source["context_frequencies"],
            observed, roi, fine_options,
        )
        history.append({"stage": "fine", "step": step + 1, **diagnostic})
    return beta, history


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    if config["rank"] != 256 or config["terms"] != 32 or source["context_profiles"] != 65:
        raise ValueError("this control must hold the original representation fixed")
    previous = json.loads(Path(config["failed_noise_results"]).read_text())
    previous = [r for r in previous if r["condition"] == "profiles65_noise0p01"]
    if len(previous) != 4 * len(config["systems"]):
        raise ValueError("complete original 1% noise failure required")
    with np.load(config["training_labels"]) as data:
        training = data[f"target{config['terms']}"].astype(float)
    mean, factor = fit_prior(training, config["rank"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "source_config_sha256": digest(config["source_config"]),
        "failed_noise_results_sha256": digest(config["failed_noise_results"]),
        "training_labels_sha256": digest(config["training_labels"]),
        "implementation_sha256": digest(__file__),
    }, indent=2))
    rows, histories = [], []
    for system in config["systems"]:
        with np.load(Path(config["source_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
            saved_ranges = data["context_ranges_m"]
            saved_observed = data["observed"]
        rng = np.random.default_rng(config["noise_seed_offset"] + system)
        noise_unit = (rng.standard_normal(saved_observed.shape)
                      + 1j * rng.standard_normal(saved_observed.shape))
        ranges, observed = make_context(
            saved_ranges, saved_observed,
            {"context_profiles": 65,
             "relative_complex_noise": config["relative_complex_noise"]},
            noise_unit,
        )
        targets = {}
        for frequency in source["query_frequencies"]:
            with np.load(Path(source["output_data"]) /
                         f"system{system}_f{frequency}.npz") as data:
                truth, target_ranges, depths = (
                    data["modal_coefficients"], data["ranges_m"], data["depths_m"]
                )
            roi = target_ranges >= source["roi_min_range_m"]
            target = (
                depth_basis(depths, spec["depth_m"], truth.shape[1])
                @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)
            ).reshape(len(depths), len(target_ranges), len(source["source_variants"]))
            targets[frequency] = (truth, target_ranges, depths, roi, target)
        for variant in config["variants"]:
            beta, history = infer_variant(
                spec, observed, ranges, source, mean, factor, config, variant
            )
            histories.extend({"system": system, "variant": variant["name"], **item}
                             for item in history)
            for frequency, (truth, target_ranges, depths, roi, target) in targets.items():
                prediction = propagate_code(
                    spec, frequency, target_ranges, source["decoder_cells"],
                    source["decoder_modes"], source["teacher_quadrature_nodes"],
                    source["eigh_device"], source["eigh_batch"], beta,
                )
                modal = modal_raw(prediction[roi], truth[roi])
                field = (
                    depth_basis(depths, spec["depth_m"], prediction.shape[1])
                    @ prediction.transpose(1, 0, 2).reshape(prediction.shape[1], -1)
                ).reshape(len(depths), len(target_ranges), len(source["source_variants"]))
                for index, source_name in enumerate(source["source_variants"]):
                    rows.append({
                        "system": system, "variant": variant["name"],
                        "frequency": frequency, "source": source_name,
                        "context_last_sq_error": history[-1]["context_after"],
                        "modal_parseval_raw": float(modal[index]),
                        **complex_metrics(field[:, roi, index], target[:, roi, index]),
                    })
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for variant in config["variants"]:
        for frequency in source["query_frequencies"]:
            for source_name in source["source_variants"]:
                sample = [r for r in rows if r["variant"] == variant["name"]
                          and r["frequency"] == frequency and r["source"] == source_name]
                summary.append({
                    "variant": variant["name"], "frequency": frequency,
                    "source": source_name, "systems": len(sample),
                    "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                    "mean_phase_rad": float(np.mean([
                        r["energy_weighted_phase_rmse_rad"] for r in sample
                    ])),
                    "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                })
    threshold = config["diagnostic_gate"]
    checks = {}
    for variant in config["variants"]:
        sample = [r for r in summary if r["variant"] == variant["name"]
                  and r["frequency"] == 800]
        checks[variant["name"]] = (
            len(sample) == len(source["source_variants"])
            and all(r["mean_raw"] <= threshold["any_variant_800hz_both_sources_mean_raw_at_most"]
                    and r["mean_phase_rad"]
                    <= threshold["same_variant_both_sources_mean_phase_rad_at_most"]
                    for r in sample)
        )
    (output / "summary.json").write_text(json.dumps({
        "rows": summary, "predeclared_checks": checks,
        "any_variant_passed": any(checks.values()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
