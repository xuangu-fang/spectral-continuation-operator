"""Bounded posthoc q-PCA768 Gaussian-code MAP diagnosis on used systems."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_generator_identification import generator_geometry, normal_statistics
from .audit_waveguide_generator_prior import conditional_mean, fit_prior
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_taper_shift import metric_rows
from .audit_waveguide_trajectory_newton import newton_step


def coordinate_solver(factor: np.ndarray):
    flat = factor.reshape(-1, factor.shape[-1])
    gram = flat.T @ flat
    chol = cho_factor(gram, check_finite=False)

    def solve(beta: np.ndarray, mean: np.ndarray) -> np.ndarray:
        return cho_solve(chol, flat.T @ (beta - mean).ravel(), check_finite=False)

    return solve


def infer_map(spec: dict, observed: np.ndarray, context_ranges: np.ndarray,
              source: dict, mean: np.ndarray, factor: np.ndarray,
              strength: float, coarse_steps: int, fine_steps: int) -> tuple[np.ndarray, list[dict]]:
    coarse = generator_geometry(spec, 64, mean.shape[1], modes=32, quadrature_nodes=160)
    gram, rhs = normal_statistics(
        spec, observed, source["context_frequencies"], context_ranges, coarse
    )
    beta = conditional_mean(gram, rhs, mean, factor, source["initial_prior_precision"])
    coarse_coordinates = coordinate_solver(factor)
    coordinates = coarse_coordinates(beta, mean)
    roi = context_ranges >= source["roi_min_range_m"]
    options = {key: source[key] for key in
               ["ridge_relative", "max_coefficient_step", "line_search"]}
    options["prior_strength"] = strength
    initial = observed[0, 0]
    history = []
    for step in range(coarse_steps):
        beta, diagnostic = newton_step(
            beta, factor, initial, coarse, source["context_frequencies"],
            observed, roi, options, prior_coordinates=coordinates,
        )
        coordinates = coarse_coordinates(beta, mean)
        history.append({"stage": "coarse", "step": step + 1, **diagnostic})
    fine_mean = resample_generator(mean, source["decoder_cells"])
    fine_factor = resample_generator(factor, source["decoder_cells"])
    beta = fine_mean + np.einsum("rdk,k->rd", fine_factor, coordinates)
    fine = generator_geometry(spec, source["decoder_cells"], mean.shape[1])
    indices = np.linspace(0, source["decoder_cells"], source["context_profiles"])
    if not np.array_equal(indices, indices.astype(int)):
        raise ValueError("context positions do not align with fine grid")
    fine_options = {**options, "observation_indices": indices.astype(int)}
    fine_coordinates = coordinate_solver(fine_factor)
    coordinates = fine_coordinates(beta, fine_mean)
    for step in range(fine_steps):
        beta, diagnostic = newton_step(
            beta, fine_factor, initial, fine, source["context_frequencies"],
            observed, roi, fine_options, prior_coordinates=coordinates,
        )
        coordinates = fine_coordinates(beta, fine_mean)
        history.append({"stage": "fine", "step": step + 1, **diagnostic})
    return beta, history


def run(config: dict, output: Path) -> None:
    if (config["rank"], config["systems"], config["prior_strengths"],
        config["coarse_steps"], config["fine_steps"], config["frequency_hz"]) != (
            768, list(range(20268000, 20268008)), [0.0, 1e-6, 1e-4, 1e-2],
            5, 5, 800):
        raise ValueError("fixed q-PCA768 MAP settings changed")
    source = json.loads(Path(config["source_config"]).read_text())
    source["initial_prior_precision"] = config["initial_prior_precision"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    labels = []
    for path in config["training_labels"]:
        with np.load(path) as data:
            labels.append(data["target32"].astype(float))
    training = np.concatenate(labels)
    if training.shape != (1024, 64, 32):
        raise ValueError("unexpected q training shape")
    mean, factor = fit_prior(training, config["rank"])
    previous = json.loads(Path(config["previous_q768_results"]).read_text())
    manifest = {
        "scope": config["scope"], "implementation_sha256": digest(__file__),
        "source_config_sha256": digest(config["source_config"]),
        "previous_q768_results_sha256": digest(config["previous_q768_results"]),
        "training_label_sha256": {str(p): digest(p) for p in config["training_labels"]},
        "context_code_sha256": {}, "truth_sha256": {},
    }
    rows, histories, timings = [], [], []
    for system in config["systems"]:
        code_path = Path(config["context_codes"]) / f"system{system}_codes.npz"
        truth_path = Path(config["truth_dir"]) / f"system{system}_f800.npz"
        manifest["context_code_sha256"][str(code_path)] = digest(code_path)
        manifest["truth_sha256"][str(truth_path)] = digest(truth_path)
        with np.load(code_path) as data:
            spec = json.loads(str(data["spec"]))
            observed = data["observed"]
            context_ranges = data["context_ranges_m"]
        with np.load(truth_path) as data:
            truth = data["modal_coefficients"]
            ranges, depths = data["ranges_m"], data["depths_m"]
        for strength in config["prior_strengths"]:
            method = f"q_pca768_map_{strength:.0e}"
            started = time.perf_counter()
            beta, history = infer_map(
                spec, observed, context_ranges, source, mean, factor,
                strength, config["coarse_steps"], config["fine_steps"],
            )
            timings.append({"system": system, "method": method,
                            "inference_seconds": time.perf_counter() - started})
            histories.extend({"system": system, "method": method, **h} for h in history)
            prediction = propagate_code(
                spec, config["frequency_hz"], ranges, source["decoder_cells"],
                source["decoder_modes"], source["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"], beta,
            )
            cases = metric_rows(
                prediction, truth, ranges, depths, spec, source, system,
                config["frequency_hz"], method,
            )
            if strength == 0.0:
                for case in cases:
                    old = [r for r in previous if r["system"] == system
                           and r["source"] == case["source"]
                           and r["method"] == "q_pca768_1e-07"]
                    if len(old) != 1 or abs(
                        case["complex_rel_l2"] - old[0]["complex_rel_l2"]
                    ) > config["max_unregularized_reproduction_raw_difference"]:
                        raise ValueError("zero-prior MAP failed original q768 reproduction")
            rows.extend({"information": "LOWBAND_DEPLOYMENT_DEV", **case}
                        for case in cases)
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        (output / "timing.json").write_text(json.dumps(timings, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for strength in config["prior_strengths"]:
        method = f"q_pca768_map_{strength:.0e}"
        for variant in source["source_variants"]:
            sample = [r for r in rows if r["method"] == method and r["source"] == variant]
            if len(sample) != len(config["systems"]):
                raise ValueError("incomplete MAP rows")
            summary.append({
                "method": method, "source": variant, "systems": len(sample),
                "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                "mean_phase_rad": float(np.mean([
                    r["energy_weighted_phase_rmse_rad"] for r in sample
                ])),
            })
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "scope": config["scope"],
        "zero_strength_original_reproduction_passed": True,
        "mean_inference_seconds_by_method": {
            f"q_pca768_map_{s:.0e}": float(np.mean([
                r["inference_seconds"] for r in timings
                if r["method"] == f"q_pca768_map_{s:.0e}"
            ])) for s in config["prior_strengths"]
        },
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
