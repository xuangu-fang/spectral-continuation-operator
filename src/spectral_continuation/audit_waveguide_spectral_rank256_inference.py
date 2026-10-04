"""Post-hoc rank128/256 context-only inverse on richer media, first8 development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import generator_geometry, normal_statistics
from .audit_waveguide_generator_prior import conditional_mean, fit_prior
from .audit_waveguide_new_systems import context_data
from .audit_waveguide_trajectory_newton import newton_step
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def infer(spec, observed, ranges, source, mean, factor, coarse_steps, fine_steps):
    geometry = generator_geometry(spec, 64, mean.shape[1], modes=32, quadrature_nodes=160)
    gram, rhs = normal_statistics(
        spec, observed, source["context_frequencies"], ranges, geometry
    )
    beta = conditional_mean(gram, rhs, mean, factor, source["initial_prior_precision"])
    roi = ranges >= source["roi_min_range_m"]
    options = {key: source[key] for key in
               ["ridge_relative", "max_coefficient_step", "line_search"]}
    initial = observed[0, 0]
    history = []
    for step in range(coarse_steps):
        beta, diagnostic = newton_step(
            beta, factor, initial, geometry, source["context_frequencies"], observed,
            roi, options,
        )
        history.append({"stage": "coarse", "step": step + 1, **diagnostic})
    fine_mean = resample_generator(mean, source["decoder_cells"])
    fine_factor = resample_generator(factor, source["decoder_cells"])
    coordinates = np.linalg.lstsq(
        factor.reshape(-1, factor.shape[-1]), (beta - mean).ravel(), rcond=None
    )[0]
    beta = fine_mean + np.einsum("rdk,k->rd", fine_factor, coordinates)
    fine_geometry = generator_geometry(spec, source["decoder_cells"], mean.shape[1])
    indices = np.linspace(0, source["decoder_cells"], source["context_profiles"])
    if not np.array_equal(indices, indices.astype(int)):
        raise ValueError("context positions must coincide with fine-grid edges")
    fine_options = {**options, "observation_indices": indices.astype(int)}
    for step in range(fine_steps):
        beta, diagnostic = newton_step(
            beta, fine_factor, initial, fine_geometry, source["context_frequencies"],
            observed, roi, fine_options,
        )
        history.append({"stage": "fine", "step": step + 1, **diagnostic})
    return beta, history


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    complete = json.loads(Path(config["source_results"]).read_text())
    if len(complete) != 20 * len(source["systems"]):
        raise ValueError("original richer-medium cohort incomplete")
    if not set(config["systems"]) <= set(source["systems"]):
        raise ValueError("development systems outside original cohort")
    with np.load(config["training_labels"]) as data:
        training = data[f"target{config['terms']}"].astype(float)
    mean, full_factor = fit_prior(training, max(config["ranks"]))
    priors = {rank: (mean, full_factor[..., :rank]) for rank in config["ranks"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "supplemental_manifest.json").write_text(
        json.dumps(
            {"registration": config["scope"],
             "original_results_sha256": digest(config["source_results"]),
             "training_labels_sha256": digest(config["training_labels"]),
             "code_sha256": digest(__file__)}, indent=2,
        )
    )
    rows, histories = [], []
    data_root = Path(source["output_data"])
    for system in config["systems"]:
        with np.load(Path(config["source_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
            saved_observed = data["observed"]
            saved_ranges = data["context_ranges_m"]
        context_ranges, observed = context_data(spec, source)
        np.testing.assert_allclose(context_ranges, saved_ranges, rtol=0, atol=0)
        np.testing.assert_allclose(observed, saved_observed, rtol=2e-6, atol=2e-6)
        for rank, (mean, factor) in priors.items():
            beta, history = infer(
                spec, observed, context_ranges, source, mean, factor,
                config["coarse_steps"], config["fine_steps"],
            )
            histories.extend({"system": system, "rank": rank, **row} for row in history)
            np.savez_compressed(output / f"system{system}_rank{rank}_code.npz", beta=beta)
            for frequency in source["query_frequencies"]:
                with np.load(data_root / f"system{system}_f{frequency}.npz") as data:
                    truth = data["modal_coefficients"]
                    ranges, depths = data["ranges_m"], data["depths_m"]
                roi = ranges >= source["roi_min_range_m"]
                prediction = propagate_code(
                    spec, frequency, ranges, source["decoder_cells"],
                    source["decoder_modes"], source["teacher_quadrature_nodes"],
                    source["eigh_device"], source["eigh_batch"], beta,
                )
                raw = modal_raw(prediction[roi], truth[roi])
                target = (
                    depth_basis(depths, spec["depth_m"], truth.shape[1])
                    @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)
                ).reshape(len(depths), len(ranges), len(source["source_variants"]))
                field = (
                    depth_basis(depths, spec["depth_m"], prediction.shape[1])
                    @ prediction.transpose(1, 0, 2).reshape(prediction.shape[1], -1)
                ).reshape(len(depths), len(ranges), len(source["source_variants"]))
                for index, variant in enumerate(source["source_variants"]):
                    rows.append(
                        {"system": system, "rank": rank, "frequency": frequency,
                         "source": variant, "modal_parseval_raw": float(raw[index]),
                         "context_last_sq_error": history[-1]["context_after"],
                         **complex_metrics(field[:, roi, index], target[:, roi, index])}
                    )
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for rank in config["ranks"]:
        for frequency in source["query_frequencies"]:
            for variant in source["source_variants"]:
                sample = [r for r in rows if r["rank"] == rank
                          and r["frequency"] == frequency and r["source"] == variant]
                summary.append(
                    {"rank": rank, "frequency": frequency, "source": variant,
                     "systems": len(sample),
                     "complex_rel_l2": float(np.mean([r["complex_rel_l2"] for r in sample])),
                     "phase_rmse_rad": float(np.mean([
                         r["energy_weighted_phase_rmse_rad"] for r in sample])),
                     "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                     "mean_context_last_sq_error": float(np.mean([
                         r["context_last_sq_error"] for r in sample]))}
                )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
