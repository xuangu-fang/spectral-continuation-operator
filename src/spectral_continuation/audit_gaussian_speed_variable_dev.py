"""Gaussian-medium, same-information speed-vs-q representation development."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_new_systems import context_data
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_rank256_inference import infer as infer_q
from .audit_waveguide_spectral_taper_shift import metric_rows
from .coherent_waveguide_data import environment
from .learned_speed_manifold import fit_speed_prior, infer_speed, speed_delta


def train(config: dict):
    training_config = json.loads(Path(config["gaussian_training_config"]).read_text())
    systems = training_config["train_systems"]
    with np.load(config["q_training_labels"]) as data:
        q = data["target"].astype(float)
        recorded_systems = data["systems"].tolist()
    if len(systems) != 512 or systems != recorded_systems or q.shape != (512, 64, 32):
        raise ValueError("Gaussian training q labels/systems do not match the freeze")
    if set(systems) & set(config["dev_systems"]):
        raise ValueError("training and dev systems overlap")
    speed = []
    for index, system in enumerate(systems):
        spec = environment(system)
        geometry = generator_geometry(
            spec, config["speed_sample_cells"], config["terms"], modes=32,
            quadrature_nodes=config["speed_sample_depth_nodes"],
        )
        if index == 0:
            np.testing.assert_allclose(
                true_projection(spec, geometry), q[index], rtol=1e-11, atol=1e-11
            )
        speed.append(speed_delta(spec, geometry))
    return np.array(speed), q, systems


def run(config: dict, output: Path) -> None:
    source = json.loads(Path(config["source_config"]).read_text())
    if (config["rank"], config["terms"], config["speed_sample_cells"],
        config["speed_sample_depth_nodes"], config["coarse_steps"],
        config["fine_steps"]) != (128, 32, 64, 160, 5, 5):
        raise ValueError("Gaussian one-change development settings changed")
    if config["dev_systems"] != list(range(20261332, 20261340)):
        raise ValueError("only repeatedly used Gaussian development8 allowed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "source_config_sha256": digest(config["source_config"]),
        "gaussian_training_config_sha256": digest(config["gaussian_training_config"]),
        "q_training_labels_sha256": digest(config["q_training_labels"]),
        "implementation_sha256": digest(__file__),
    }, indent=2))
    speed_training, q_training, training_systems = train(config)
    speed_mean, speed_factor, speed_stats = fit_speed_prior(
        speed_training, config["rank"]
    )
    q_mean, q_factor = fit_prior(q_training, config["rank"])
    np.savez_compressed(output / "training_speed_labels.npz",
                        systems=np.array(training_systems),
                        speed_delta=speed_training.astype(np.float32))
    np.savez_compressed(output / "learned_priors.npz",
                        mean_speed=speed_mean, factor_speed=speed_factor,
                        mean_q=q_mean, factor_q=q_factor)
    (output / "training_manifest.json").write_text(json.dumps({
        "training_speed_labels_sha256": digest(output / "training_speed_labels.npz"),
        "learned_priors_sha256": digest(output / "learned_priors.npz"),
        "speed_prior": speed_stats,
        "q_global_basis_float64_values": int(q_mean.size + q_factor.size),
        "training_systems": training_systems,
    }, indent=2))
    rows, histories, timings = [], [], []
    for system in config["dev_systems"]:
        spec = environment(system)
        context_ranges, observed = context_data(spec, source)
        candidates = {}
        started = time.perf_counter()
        speed_beta, coordinates, history = infer_speed(
            spec, observed, context_ranges, source, config, speed_mean, speed_factor
        )
        candidates["speed_nonlinear"] = speed_beta
        timings.append({"system": system, "method": "speed_nonlinear",
                        "inference_seconds": time.perf_counter() - started})
        histories.extend({"system": system, "method": "speed_nonlinear", **entry}
                         for entry in history)
        np.savez_compressed(output / f"system{system}_speed_code.npz",
                            beta=speed_beta, coordinates=coordinates)
        for precision in config["q_prior_initial_precisions"]:
            q_source = {**source, "initial_prior_precision": precision}
            started = time.perf_counter()
            q_beta, q_history = infer_q(
                spec, observed, context_ranges, q_source, q_mean, q_factor,
                config["coarse_steps"], config["fine_steps"],
            )
            method = f"q_pca_{precision:.0e}"
            candidates[method] = q_beta
            timings.append({"system": system, "method": method,
                            "inference_seconds": time.perf_counter() - started})
            histories.extend({"system": system, "method": method, **entry}
                             for entry in q_history)
        for frequency in source["query_frequencies"]:
            with np.load(Path(config["dev_data"]) /
                         f"system{system}_f{frequency}.npz") as data:
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
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
    summary = []
    for method in candidates:
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
    speed800 = [r for r in summary if r["method"] == "speed_nonlinear"
                and r["frequency"] == 800]
    comparison = []
    for row in speed800:
        best_q = min(r["mean_raw"] for r in summary
                     if r["method"].startswith("q_pca_") and r["frequency"] == 800
                     and r["source"] == row["source"])
        comparison.append({"source": row["source"], "best_q_mean_raw": best_q,
                           "speed_raw_improvement": float(1 - row["mean_raw"] / best_q)})
    gate = config["gate_800hz_both_sources"]
    checks = {
        "mean_raw": all(r["mean_raw"] <= gate["speed_mean_raw_at_most"]
                        for r in speed800),
        "mean_phase": all(r["mean_phase_rad"] <= gate["speed_mean_phase_rad_at_most"]
                          for r in speed800),
        "improvement": all(r["speed_raw_improvement"] >=
                           gate["speed_raw_improvement_vs_best_q_pca_at_least"]
                           for r in comparison),
    }
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "comparison": comparison, "gate_checks": checks,
        "all_dev_gates_pass": all(checks.values()),
        "speed_prior": speed_stats,
        "q_global_basis_float64_values": int(q_mean.size + q_factor.size),
        "mean_inference_seconds_by_method": {
            method: float(np.mean([r["inference_seconds"] for r in timings
                                   if r["method"] == method]))
            for method in sorted({r["method"] for r in timings})
        },
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
