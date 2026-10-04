"""Posthoc capacity-cost diagnostic: low-band deployment of q-PCA768."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_rank256_inference import infer
from .audit_waveguide_spectral_taper_shift import metric_rows


def run(config: dict, output: Path) -> None:
    if (config["rank"], config["systems"], config["frequency_hz"],
        config["coarse_steps"], config["fine_steps"],
        config["initial_prior_precisions"]) != (
            768, list(range(20268000, 20268016)), 800, 5, 5, [1e-9, 1e-7]):
        raise ValueError("frozen high-capacity q deployment diagnosis changed")
    source = json.loads(Path(config["source_config"]).read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    labels = []
    for path in config["training_labels"]:
        with np.load(path) as data:
            labels.append(data[config["training_array_key"]].astype(float))
    training = np.concatenate(labels)
    if training.shape != (1024, 64, 32):
        raise ValueError("unexpected q training shape")
    mean, factor = fit_prior(training, config["rank"])
    manifest = {
        "scope": config["scope"], "implementation_sha256": digest(__file__),
        "source_config_sha256": digest(config["source_config"]),
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
        for precision in config["initial_prior_precisions"]:
            method = f"q_pca768_{precision:.0e}"
            started = time.perf_counter()
            beta, history = infer(
                spec, observed, context_ranges,
                {**source, "initial_prior_precision": precision},
                mean, factor, config["coarse_steps"], config["fine_steps"],
            )
            timings.append({"system": system, "method": method,
                            "inference_seconds": time.perf_counter() - started})
            histories.extend({"system": system, "method": method, **r} for r in history)
            prediction = propagate_code(
                spec, config["frequency_hz"], ranges, source["decoder_cells"],
                source["decoder_modes"], source["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"], beta,
            )
            rows.extend({"information": "LOWBAND_DEPLOYMENT", **r} for r in metric_rows(
                prediction, truth, ranges, depths, spec, source, system,
                config["frequency_hz"], method,
            ))
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        (output / "timing.json").write_text(json.dumps(timings, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for precision in config["initial_prior_precisions"]:
        method = f"q_pca768_{precision:.0e}"
        for source_name in source["source_variants"]:
            sample = [r for r in rows if r["method"] == method
                      and r["source"] == source_name]
            if len(sample) != len(config["systems"]):
                raise ValueError("incomplete low-band deployment rows")
            summary.append({
                "method": method, "source": source_name, "systems": len(sample),
                "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                "mean_phase_rad": float(np.mean([
                    r["energy_weighted_phase_rmse_rad"] for r in sample
                ])),
                "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
            })
    (output / "summary.json").write_text(json.dumps({
        "summary": summary,
        "mean_inference_seconds_by_method": {
            f"q_pca768_{p:.0e}": float(np.mean([
                r["inference_seconds"] for r in timings
                if r["method"] == f"q_pca768_{p:.0e}"
            ])) for p in config["initial_prior_precisions"]
        },
        "rank": config["rank"],
        "per_environment_code_float64_values": config["rank"],
        "shared_basis_float64_values": int(mean.size + factor.size),
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
