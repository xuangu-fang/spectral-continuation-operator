"""Previously-used32-system, fixed 1% noise source-initialization cross-check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_spectral_context_robustness import digest, make_context
from .audit_waveguide_spectral_noise_initialization import infer_variant
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    if config["rank"] != 256 or config["terms"] != 32:
        raise ValueError("fixed 256D spectral representation required")
    if config["systems"] != source["systems"]:
        raise ValueError("all previously used32 systems must be reported")
    with np.load(config["training_labels"]) as data:
        training = data[f"target{config['terms']}"].astype(float)
    mean, factor = fit_prior(training, config["rank"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "source_config_sha256": digest(config["source_config"]),
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
        unit = (rng.standard_normal(saved_observed.shape)
                + 1j * rng.standard_normal(saved_observed.shape))
        context_ranges, observed = make_context(
            saved_ranges, saved_observed,
            {"context_profiles": source["context_profiles"],
             "relative_complex_noise": config["relative_complex_noise"]}, unit,
        )
        targets = {}
        for frequency in source["query_frequencies"]:
            with np.load(Path(source["output_data"]) /
                         f"system{system}_f{frequency}.npz") as data:
                truth, ranges, depths = (
                    data["modal_coefficients"], data["ranges_m"], data["depths_m"]
                )
            roi = ranges >= source["roi_min_range_m"]
            target = (
                depth_basis(depths, spec["depth_m"], truth.shape[1])
                @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)
            ).reshape(len(depths), len(ranges), len(source["source_variants"]))
            targets[frequency] = (truth, ranges, depths, roi, target)
        for variant in config["variants"]:
            beta, history = infer_variant(
                spec, observed, context_ranges, source, mean, factor, config, variant
            )
            histories.extend({"system": system, "variant": variant["name"], **item}
                             for item in history)
            for frequency, (truth, ranges, depths, roi, target) in targets.items():
                prediction = propagate_code(
                    spec, frequency, ranges, source["decoder_cells"],
                    source["decoder_modes"], source["teacher_quadrature_nodes"],
                    source["eigh_device"], source["eigh_batch"], beta,
                )
                modal = modal_raw(prediction[roi], truth[roi])
                field = (
                    depth_basis(depths, spec["depth_m"], prediction.shape[1])
                    @ prediction.transpose(1, 0, 2).reshape(prediction.shape[1], -1)
                ).reshape(len(depths), len(ranges), len(source["source_variants"]))
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
    lookup = {(r["variant"], r["system"], r["frequency"], r["source"]): r for r in rows}
    summary = []
    for variant in config["variants"]:
        for frequency in source["query_frequencies"]:
            for source_name in source["source_variants"]:
                sample = [lookup[(variant["name"], system, frequency, source_name)]
                          for system in config["systems"]]
                summary.append({
                    "variant": variant["name"], "frequency": frequency,
                    "source": source_name, "systems": len(sample),
                    "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                    "mean_phase_rad": float(np.mean([
                        r["energy_weighted_phase_rmse_rad"] for r in sample
                    ])),
                    "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                    "wins_over_noisy_source": (
                        None if variant["name"] == "normal_noisy_source" else sum(
                            lookup[(variant["name"], system, frequency, source_name)][
                                "complex_rel_l2"
                            ] < lookup[("normal_noisy_source", system, frequency, source_name)][
                                "complex_rel_l2"
                            ] for system in config["systems"]
                        )
                    ),
                })
    thresholds = config["confirm_gate_800hz_both_sources"]
    checked = [r for r in summary if r["variant"] == "normal_known_source"
               and r["frequency"] == 800]
    checks = {
        "mean_raw": all(r["mean_raw"] <= thresholds["known_source_mean_raw_at_most"]
                        for r in checked),
        "mean_phase": all(r["mean_phase_rad"]
                          <= thresholds["known_source_mean_phase_rad_at_most"]
                          for r in checked),
        "wins": all(r["wins_over_noisy_source"]
                    >= thresholds["known_source_system_wins_vs_noisy_at_least"]
                    for r in checked),
        "max_raw": all(r["max_raw"]
                       <= thresholds["known_source_max_system_raw_at_most"]
                       for r in checked),
    }
    (output / "summary.json").write_text(json.dumps({
        "rows": summary, "predeclared_checks": checks,
        "gate_passed": len(checked) == 2 and all(checks.values()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
