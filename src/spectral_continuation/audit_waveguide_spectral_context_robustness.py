"""Development-only information-budget check for the fixed spectral 256D codec."""

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
from .audit_waveguide_trajectory_newton import newton_step
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_context(saved_ranges, saved_observed, condition, noise_unit):
    count = condition["context_profiles"]
    if (len(saved_ranges) - 1) % (count - 1):
        raise ValueError("profile subset must divide original observation grid")
    indices = np.arange(count) * ((len(saved_ranges) - 1) // (count - 1))
    ranges = saved_ranges[indices]
    observed = saved_observed[:, indices].astype(np.complex128)
    unit = noise_unit[:, indices]
    level = condition["relative_complex_noise"]
    if level:
        signal_norm = np.linalg.norm(observed, axis=(1, 2))
        unit_norm = np.linalg.norm(unit, axis=(1, 2))
        observed += level * unit * (signal_norm / unit_norm)[:, None, None]
    return ranges, observed


def infer_indexed(spec, observed, ranges, source, mean, factor, coarse_steps, fine_steps):
    """Same fixed GN update with the actual observation indices at both resolutions."""
    geometry = generator_geometry(spec, 64, mean.shape[1], modes=32, quadrature_nodes=160)
    gram, rhs = normal_statistics(
        spec, observed, source["context_frequencies"], ranges, geometry
    )
    beta = conditional_mean(gram, rhs, mean, factor, source["initial_prior_precision"])
    count = source["context_profiles"]
    roi = ranges >= source["roi_min_range_m"]
    options = {key: source[key] for key in
               ["ridge_relative", "max_coefficient_step", "line_search"]}
    initial = observed[0, 0]
    coarse_indices = np.linspace(0, 64, count)
    if not np.array_equal(coarse_indices, coarse_indices.astype(int)):
        raise ValueError("context observations must coincide with coarse-grid edges")
    coarse_options = {**options, "observation_indices": coarse_indices.astype(int)}
    history = []
    for step in range(coarse_steps):
        beta, diagnostic = newton_step(
            beta, factor, initial, geometry, source["context_frequencies"],
            observed, roi, coarse_options,
        )
        history.append({"stage": "coarse", "step": step + 1, **diagnostic})
    fine_mean = resample_generator(mean, source["decoder_cells"])
    fine_factor = resample_generator(factor, source["decoder_cells"])
    coordinates = np.linalg.lstsq(
        factor.reshape(-1, factor.shape[-1]), (beta - mean).ravel(), rcond=None
    )[0]
    beta = fine_mean + np.einsum("rdk,k->rd", fine_factor, coordinates)
    fine_geometry = generator_geometry(spec, source["decoder_cells"], mean.shape[1])
    fine_indices = np.linspace(0, source["decoder_cells"], count)
    if not np.array_equal(fine_indices, fine_indices.astype(int)):
        raise ValueError("context observations must coincide with fine-grid edges")
    fine_options = {**options, "observation_indices": fine_indices.astype(int)}
    for step in range(fine_steps):
        beta, diagnostic = newton_step(
            beta, fine_factor, initial, fine_geometry, source["context_frequencies"],
            observed, roi, fine_options,
        )
        history.append({"stage": "fine", "step": step + 1, **diagnostic})
    return beta, history


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    if source["context_profiles"] != 65:
        raise ValueError("fixed original 65-profile context required")
    if config["rank"] != 256 or config["terms"] != 32:
        raise ValueError("this audit only checks the already-selected representation")
    if not set(config["systems"]) <= set(source["systems"]):
        raise ValueError("development system outside source cohort")
    baseline = json.loads(Path(config["baseline_results"]).read_text())
    if not all(any(row["system"] == system and row["rank"] == 256 for row in baseline)
               for system in config["systems"]):
        raise ValueError("complete clean65 development baseline required")
    with np.load(config["training_labels"]) as data:
        training = data[f"target{config['terms']}"].astype(float)
    mean, factor = fit_prior(training, config["rank"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "source_config_sha256": digest(config["source_config"]),
        "baseline_results_sha256": digest(config["baseline_results"]),
        "training_labels_sha256": digest(config["training_labels"]),
        "implementation_sha256": digest(__file__),
    }, indent=2))
    rows, histories, noise_checks = [], [], []
    for system in config["systems"]:
        with np.load(Path(config["source_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
            saved_ranges = data["context_ranges_m"]
            saved_observed = data["observed"]
        if saved_observed.shape[:2] != (len(source["context_frequencies"]), 65):
            raise ValueError("unexpected context shape")
        rng = np.random.default_rng(config["noise_seed_offset"] + system)
        noise_unit = (rng.standard_normal(saved_observed.shape)
                      + 1j * rng.standard_normal(saved_observed.shape))
        targets = {}
        for frequency in source["query_frequencies"]:
            with np.load(Path(source["output_data"]) /
                         f"system{system}_f{frequency}.npz") as data:
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
            roi = ranges >= source["roi_min_range_m"]
            target = (
                depth_basis(depths, spec["depth_m"], truth.shape[1])
                @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)
            ).reshape(len(depths), len(ranges), len(source["source_variants"]))
            targets[frequency] = (truth, ranges, depths, roi, target)
        for condition in config["conditions"]:
            context_ranges, observed = make_context(
                saved_ranges, saved_observed, condition, noise_unit
            )
            count = condition["context_profiles"]
            clean = saved_observed[:, np.arange(count) * (64 // (count - 1))]
            actual_noise = (
                np.linalg.norm(observed - clean, axis=(1, 2))
                / np.linalg.norm(clean, axis=(1, 2))
            )
            noise_checks.append({"system": system, "condition": condition["name"],
                                 "actual_relative_norm_by_frequency": actual_noise.tolist()})
            method_source = {**source, "context_profiles": count}
            beta, history = infer_indexed(
                spec, observed, context_ranges, method_source, mean, factor,
                config["coarse_steps"], config["fine_steps"],
            )
            histories.extend({"system": system, "condition": condition["name"], **item}
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
                for index, variant in enumerate(source["source_variants"]):
                    rows.append({
                        "system": system, "condition": condition["name"],
                        "context_profiles": count,
                        "relative_complex_noise": condition["relative_complex_noise"],
                        "frequency": frequency, "source": variant,
                        "modal_parseval_raw": float(modal[index]),
                        "context_last_sq_error": history[-1]["context_after"],
                        **complex_metrics(field[:, roi, index], target[:, roi, index]),
                    })
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        (output / "noise_checks.json").write_text(json.dumps(noise_checks, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for condition in config["conditions"]:
        for frequency in source["query_frequencies"]:
            for variant in source["source_variants"]:
                sample = [row for row in rows if row["condition"] == condition["name"]
                          and row["frequency"] == frequency and row["source"] == variant]
                summary.append({
                    "condition": condition["name"], "frequency": frequency,
                    "source": variant, "systems": len(sample),
                    "mean_raw": float(np.mean([row["complex_rel_l2"] for row in sample])),
                    "mean_phase_rad": float(np.mean([
                        row["energy_weighted_phase_rmse_rad"] for row in sample
                    ])),
                    "max_raw": float(np.max([row["complex_rel_l2"] for row in sample])),
                })
    checks = {}
    for name, thresholds in config["gates_800hz_both_sources"].items():
        sample = [row for row in summary if row["condition"] == name
                  and row["frequency"] == 800]
        if len(sample) != len(source["source_variants"]):
            raise ValueError("incomplete gate source set")
        checks[name] = {
            "raw": all(row["mean_raw"] <= thresholds["mean_raw_at_most"] for row in sample),
            "phase": all(row["mean_phase_rad"] <= thresholds["mean_phase_rad_at_most"]
                         for row in sample),
        }
    (output / "summary.json").write_text(json.dumps({
        "rows": summary, "predeclared_checks": checks,
        "all_checks_passed": all(all(item.values()) for item in checks.values()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
