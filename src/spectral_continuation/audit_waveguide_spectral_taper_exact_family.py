"""Exact-known synthetic speed-family control on the already-used taper10 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_exact_basis import analytic_prior
from .audit_waveguide_spectral_exact_nonlinear import infer
from .audit_waveguide_spectral_taper_shift import metric_rows


def run(config: dict, output: Path) -> None:
    source = json.loads(Path(config["source_config"]).read_text())
    ood = json.loads(Path(config["ood_config"]).read_text())
    ood_rows = json.loads(Path(config["ood_results"]).read_text())
    if config["systems"] != ood["systems"] or len(ood_rows) != 12 * len(config["systems"]):
        raise ValueError("must report the full already-evaluated taper10 cohort")
    if (config["range_cosine_terms"], config["depth_cosine_terms"],
        config["depth_terms"], config["taper_scale"],
        config["coarse_steps"], config["fine_steps"],
        config["linear_initial_prior_precision"]) != (16, 16, 32, 10.0, 5, 5, 1e-7):
        raise ValueError("fixed exact-family control settings required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "ood_results_sha256": digest(config["ood_results"]),
        "implementation_sha256": digest(__file__),
        "exact_inverse_sha256": digest(Path(infer.__code__.co_filename)),
        "analytic_prior_sha256": digest(Path(analytic_prior.__code__.co_filename)),
        "systems": config["systems"],
    }, indent=2))
    rows, histories = [], []
    for system in config["systems"]:
        with np.load(Path(config["ood_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
            observed = data["observed"]
            context_ranges = data["context_ranges_m"]
        if spec.get("spectral_taper_scale") != 10.0:
            raise ValueError("target medium does not match declared taper family")
        beta, coordinates, history = infer(
            spec, observed, context_ranges, source, config
        )
        np.savez_compressed(output / f"system{system}_coordinates.npz",
                            coordinates=coordinates)
        histories.extend({"system": system, **entry} for entry in history)
        for frequency in source["query_frequencies"]:
            with np.load(Path(ood["output_data"]) /
                         f"system{system}_f{frequency}.npz") as data:
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
            prediction = propagate_code(
                spec, frequency, ranges, source["decoder_cells"],
                source["decoder_modes"], source["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"], beta,
            )
            rows.extend({**row,
                         "extra_information": "known exact 16x16 speed family and taper10; no test coefficients or target fields",
                         "context_last_sq_error": history[-1]["context_after"]}
                        for row in metric_rows(
                            prediction, truth, ranges, depths, spec, source, system,
                            frequency, "exact_known_speed_family",
                        ))
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for frequency in source["query_frequencies"]:
        for variant in source["source_variants"]:
            sample = [r for r in rows if r["frequency"] == frequency
                      and r["source"] == variant]
            summary.append({
                "frequency": frequency, "source": variant, "systems": len(sample),
                "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                "mean_phase_rad": float(np.mean([
                    r["energy_weighted_phase_rmse_rad"] for r in sample
                ])),
                "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                "mean_context_last_sq_error": float(np.mean([
                    r["context_last_sq_error"] for r in sample
                ])),
            })
    gate = config["gate_800hz_both_sources"]
    checked = [r for r in summary if r["frequency"] == 800]
    checks = {
        "mean_raw": all(r["mean_raw"] <= gate["mean_raw_at_most"] for r in checked),
        "mean_phase": all(r["mean_phase_rad"] <= gate["mean_phase_rad_at_most"]
                          for r in checked),
        "max_raw": all(r["max_raw"] <= gate["max_raw_at_most"] for r in checked),
    }
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "gate_checks": checks,
        "all_gates_pass": all(checks.values()), "scope": config["scope"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
