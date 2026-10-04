"""One-change speed-manifold vs q-PCA development on already-seen media."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_rank256_inference import infer as infer_q
from .audit_waveguide_spectral_taper_shift import metric_rows, shifted_environment
from .learned_speed_manifold import (
    fit_speed_prior,
    infer_speed,
    speed_delta,
    speed_projection,
)


def training_data(config: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    train_ids = []
    speed_fields = []
    q_labels = []
    with np.load(config["taper5_training_labels"]) as data:
        q5 = data["target32"].astype(float)
        ids5 = data["systems"].tolist()
    with np.load(config["taper10_training_labels"]) as data:
        q10 = data["target32"].astype(float)
        ids10 = data["systems"].tolist()
    for taper, start, q in zip(
        config["train_taper_scales"],
        [config["taper5_train_seed_start"], config["taper10_train_seed_start"]],
        [q5, q10], strict=True,
    ):
        expected = list(range(start, start + config["train_count_per_taper"]))
        found = ids5 if taper == 5.0 else ids10
        if found != expected or len(q) != len(expected):
            raise ValueError("medium training label seeds do not match freeze")
        for index, system in enumerate(expected):
            spec = shifted_environment(system, 16, 15.0, taper)
            geometry = generator_geometry(
                spec, config["speed_sample_cells"], config["terms"],
                modes=32, quadrature_nodes=config["speed_sample_depth_nodes"],
            )
            if index == 0:
                np.testing.assert_allclose(
                    true_projection(spec, geometry), q[index], rtol=1e-11, atol=1e-11
                )
            speed_fields.append(speed_delta(spec, geometry))
            train_ids.append(system)
        q_labels.append(q)
    if len(train_ids) != 2 * config["train_count_per_taper"]:
        raise ValueError("wrong medium label count")
    if set(train_ids) & set(config["dev_systems"]):
        raise ValueError("training/development system overlap")
    return np.array(speed_fields), np.concatenate(q_labels), train_ids


def run(config: dict, output: Path) -> None:
    source = json.loads(Path(config["source_config"]).read_text())
    if (config["rank"], config["terms"], config["speed_sample_cells"],
        config["speed_sample_depth_nodes"], config["coarse_steps"],
        config["fine_steps"]) != (255, 32, 64, 160, 5, 5):
        raise ValueError("one-change speed-manifold POC settings are frozen")
    if config["dev_systems"] != list(range(20267000, 20267016)):
        raise ValueError("development systems must be the already-seen OOD cohort")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "source_config_sha256": digest(config["source_config"]),
        "taper5_training_labels_sha256": digest(config["taper5_training_labels"]),
        "taper10_training_labels_sha256": digest(config["taper10_training_labels"]),
        "implementation_sha256": digest(__file__),
        "rank": config["rank"], "dev_systems": config["dev_systems"],
    }, indent=2))
    speed_train, q_train, train_ids = training_data(config)
    mean_speed, factor_speed, speed_stats = fit_speed_prior(
        speed_train, config["rank"]
    )
    mean_q, factor_q = fit_prior(q_train, config["rank"])
    np.savez_compressed(output / "training_speed_labels.npz",
                        systems=np.array(train_ids), speed_delta=speed_train.astype(np.float32))
    np.savez_compressed(output / "learned_priors.npz",
                        mean_speed=mean_speed, factor_speed=factor_speed,
                        mean_q=mean_q, factor_q=factor_q)
    (output / "training_manifest.json").write_text(json.dumps({
        "training_speed_labels_sha256": digest(output / "training_speed_labels.npz"),
        "learned_priors_sha256": digest(output / "learned_priors.npz"),
        "speed_prior": speed_stats,
        "q_global_basis_float64_values": int(mean_q.size + factor_q.size),
        "train_ids": train_ids,
    }, indent=2))
    rows, histories, timings = [], [], []
    for system in config["dev_systems"]:
        with np.load(Path(config["dev_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
            observed = data["observed"]
            context_ranges = data["context_ranges_m"]
        candidates = {}
        started = time.perf_counter()
        beta_speed, coordinates, history = infer_speed(
            spec, observed, context_ranges, source, config, mean_speed, factor_speed
        )
        timings.append({"system": system, "method": "speed_nonlinear",
                        "inference_seconds": time.perf_counter() - started})
        candidates["speed_nonlinear"] = beta_speed
        np.savez_compressed(output / f"system{system}_speed_code.npz",
                            beta=beta_speed, coordinates=coordinates)
        histories.extend({"system": system, "method": "speed_nonlinear", **entry}
                         for entry in history)
        for precision in config["q_prior_initial_precisions"]:
            q_source = {**source, "initial_prior_precision": precision}
            started = time.perf_counter()
            beta_q, q_history = infer_q(
                spec, observed, context_ranges, q_source, mean_q, factor_q,
                config["coarse_steps"], config["fine_steps"],
            )
            method = f"q_pca_{precision:.0e}"
            timings.append({"system": system, "method": method,
                            "inference_seconds": time.perf_counter() - started})
            candidates[method] = beta_q
            histories.extend({"system": system, "method": method, **entry}
                             for entry in q_history)
        coarse = generator_geometry(
            spec, config["speed_sample_cells"], config["terms"], modes=32,
            quadrature_nodes=config["speed_sample_depth_nodes"],
        )
        true_speed = speed_delta(spec, coarse)
        oracle_coordinates = np.linalg.lstsq(
            factor_speed.reshape(-1, config["rank"]),
            (true_speed - mean_speed).ravel(), rcond=1e-8,
        )[0]
        fine = generator_geometry(
            spec, source["decoder_cells"], config["terms"], modes=32,
            quadrature_nodes=config["speed_sample_depth_nodes"],
        )
        oracle_beta, _ = speed_projection(
            spec, fine,
            resample_generator(mean_speed, source["decoder_cells"]),
            resample_generator(factor_speed, source["decoder_cells"]),
            oracle_coordinates, derivative=False,
        )
        candidates["speed_true_medium_projection_oracle"] = oracle_beta
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
    gate = config["dev_gate_800hz_both_sources"]
    checked = [r for r in summary if r["method"] == "speed_nonlinear"
               and r["frequency"] == 800]
    comparisons = []
    for row in checked:
        best_q = min(r["mean_raw"] for r in summary
                     if r["method"].startswith("q_pca_") and r["frequency"] == 800
                     and r["source"] == row["source"])
        comparisons.append({"source": row["source"], "best_q_mean_raw": best_q,
                            "speed_raw_improvement": float(1 - row["mean_raw"] / best_q)})
    checks = {
        "mean_raw": all(r["mean_raw"] <= gate["speed_mean_raw_at_most"]
                        for r in checked),
        "mean_phase": all(r["mean_phase_rad"] <= gate["speed_mean_phase_rad_at_most"]
                          for r in checked),
        "improvement": all(r["speed_raw_improvement"]
                           >= gate["speed_raw_improvement_vs_best_matched_q_pca_at_least"]
                           for r in comparisons),
    }
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "comparisons": comparisons,
        "gate_checks": checks, "all_dev_gates_pass": all(checks.values()),
        "speed_prior": speed_stats,
        "q_global_basis_float64_values": int(mean_q.size + factor_q.size),
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
