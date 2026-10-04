"""Frozen-prior confirmation on never-used taper-7.5 synthetic waveguides."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_new_systems import context_data
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_rank256_inference import infer as infer_q
from .audit_waveguide_spectral_taper_shift import (
    metric_rows,
    numerical_audit,
    shifted_environment,
)
from .learned_speed_manifold import infer_speed


def summarize(rows: list[dict], config: dict, audits: list[dict], timings: list[dict]) -> dict:
    methods = ["speed_nonlinear", "q_pca_1e-09", "q_pca_1e-07"]
    summary = []
    for method in methods:
        for frequency in [400, 800]:
            for variant in ["original", "narrow2m"]:
                sample = [r for r in rows if r["method"] == method
                          and r["frequency"] == frequency and r["source"] == variant]
                if len(sample) != len(config["systems"]):
                    raise ValueError("missing evaluation rows")
                summary.append({
                    "method": method, "frequency": frequency, "source": variant,
                    "systems": len(sample),
                    "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                    "mean_phase_rad": float(np.mean([
                        r["energy_weighted_phase_rmse_rad"] for r in sample
                    ])),
                    "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                    "systems_raw_at_most_0p05": sum(
                        r["complex_rel_l2"] <= 0.05 for r in sample
                    ),
                })
    speed = [r for r in summary if r["method"] == "speed_nonlinear"
             and r["frequency"] == 800]
    comparisons = []
    for row in speed:
        best_q = min(r["mean_raw"] for r in summary
                     if r["method"].startswith("q_pca_") and r["frequency"] == 800
                     and r["source"] == row["source"])
        comparisons.append({"source": row["source"], "best_q_mean_raw": best_q,
                            "speed_raw_improvement": float(1 - row["mean_raw"] / best_q)})
    gate = config["gate_800hz_both_sources"]
    checks = {
        "mean_raw": all(r["mean_raw"] <= gate["speed_mean_raw_at_most"] for r in speed),
        "mean_phase": all(r["mean_phase_rad"] <= gate["speed_mean_phase_rad_at_most"]
                          for r in speed),
        "improvement": all(
            r["speed_raw_improvement"] >=
            gate["speed_raw_improvement_vs_best_matched_q_pca_at_least"]
            for r in comparisons
        ),
        "all_systems_raw": all(
            r["systems_raw_at_most_0p05"] >= gate["systems_raw_at_most_0p05_at_least"]
            for r in speed
        ),
        "numerical_audit": max(max(r["raw_by_source"]) for r in audits)
        < gate["numerical_audit_max_raw_below"],
    }
    return {
        "summary": summary, "comparisons": comparisons,
        "gate_checks": checks, "all_confirm_gates_pass": all(checks.values()),
        "mean_inference_seconds_by_method": {
            method: float(np.mean([r["inference_seconds"] for r in timings
                                   if r["method"] == method]))
            for method in methods
        },
        "scope": config["scope"],
    }


def run(config: dict, output: Path) -> None:
    source = json.loads(Path(config["source_config"]).read_text())
    dev = json.loads(Path(config["dev_config"]).read_text())
    if (
        config["test_taper_scale"], config["terms"], config["perturbation_rms_m_s"]
    ) != (7.5, 16, 15.0):
        raise ValueError("confirmation physical setting is frozen")
    if config["systems"] != list(range(20268000, 20268016)):
        raise ValueError("confirmation systems must be never-used 20268000-015")
    if config["audit_systems"] != [20268100, 20268101]:
        raise ValueError("numerical audit systems have changed")
    if set(config["systems"]) & set(dev["dev_systems"] + source["systems"]):
        raise ValueError("development/test systems overlap")
    if config["q_prior_initial_precisions"] != dev["q_prior_initial_precisions"]:
        raise ValueError("q baseline initialization settings changed")
    if digest(config["frozen_priors"]) != config["frozen_priors_sha256"]:
        raise ValueError("frozen prior SHA changed since development")
    with np.load(config["frozen_priors"]) as data:
        mean_speed, factor_speed = data["mean_speed"], data["factor_speed"]
        mean_q, factor_q = data["mean_q"], data["factor_q"]
    if factor_speed.shape != (64, 160, 255) or factor_q.shape != (64, 32, 255):
        raise ValueError("frozen basis shape changed")
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(config["output_data"])
    data_root.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "source_config_sha256": digest(config["source_config"]),
        "dev_config_sha256": digest(config["dev_config"]),
        "frozen_priors_sha256": digest(config["frozen_priors"]),
        "implementation_sha256": digest(__file__),
        "systems": config["systems"], "audit_systems": config["audit_systems"],
    }, indent=2))
    audit_config = {
        "audit_systems": config["audit_systems"],
        "terms": config["terms"],
        "perturbation_rms_m_s": config["perturbation_rms_m_s"],
        "target_taper_scale": config["test_taper_scale"],
        "gate_800hz_both_sources": config["gate_800hz_both_sources"],
    }
    audits = numerical_audit(audit_config, source)
    (output / "numerical_audit.json").write_text(json.dumps(audits, indent=2))
    rows, histories, timings = [], [], []
    for system in config["systems"]:
        spec = shifted_environment(
            system, config["terms"], config["perturbation_rms_m_s"],
            config["test_taper_scale"],
        )
        context_ranges, observed = context_data(spec, source)
        candidates = {}
        started = time.perf_counter()
        speed_beta, coordinates, history = infer_speed(
            spec, observed, context_ranges, source, dev, mean_speed, factor_speed
        )
        timings.append({"system": system, "method": "speed_nonlinear",
                        "inference_seconds": time.perf_counter() - started})
        histories.extend({"system": system, "method": "speed_nonlinear", **entry}
                         for entry in history)
        candidates["speed_nonlinear"] = speed_beta
        np.savez_compressed(output / f"system{system}_codes.npz", spec=json.dumps(spec),
                            context_ranges_m=context_ranges,
                            observed=observed.astype(np.complex64),
                            speed_beta=speed_beta, speed_coordinates=coordinates)
        for precision in config["q_prior_initial_precisions"]:
            q_source = {**source, "initial_prior_precision": precision}
            started = time.perf_counter()
            q_beta, q_history = infer_q(
                spec, observed, context_ranges, q_source, mean_q, factor_q,
                dev["coarse_steps"], dev["fine_steps"],
            )
            method = f"q_pca_{precision:.0e}"
            candidates[method] = q_beta
            timings.append({"system": system, "method": method,
                            "inference_seconds": time.perf_counter() - started})
            histories.extend({"system": system, "method": method, **entry}
                             for entry in q_history)
        ranges = ((np.arange(source["query_range_count"]) + 0.37)
                  / source["query_range_count"] * spec["range_m"])
        depths = np.linspace(0.1, spec["depth_m"] - 0.1, source["query_depth_count"])
        for frequency in source["query_frequencies"]:
            truth = propagate_code(
                spec, frequency, ranges, source["teacher_cells"],
                source["teacher_modes"], source["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"],
            )
            data_file = data_root / f"system{system}_f{frequency}.npz"
            np.savez_compressed(
                data_file, modal_coefficients=truth.astype(np.complex64),
                ranges_m=ranges, depths_m=depths, frequency_hz=frequency,
                source_variants=source["source_variants"], spec=json.dumps(spec),
            )
            for method, beta in candidates.items():
                prediction = propagate_code(
                    spec, frequency, ranges, source["decoder_cells"],
                    source["decoder_modes"], source["teacher_quadrature_nodes"],
                    source["eigh_device"], source["eigh_batch"], beta,
                )
                rows.extend(metric_rows(
                    prediction, truth, ranges, depths, spec, source, system,
                    frequency, method,
                ))
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        (output / "timing.json").write_text(json.dumps(timings, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "summary.json").write_text(json.dumps(
        summarize(rows, config, audits, timings), indent=2,
    ))
    files = sorted(data_root.glob("system*_f*.npz"))
    if len(files) != 2 * len(config["systems"]):
        raise ValueError("missing target truth files")
    (output / "data_sha256.json").write_text(json.dumps(
        {str(path): digest(path) for path in files}, indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
