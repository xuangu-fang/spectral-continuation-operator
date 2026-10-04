"""Fixed-strength q-PCA768 MAP check on already-used latter eight systems."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .audit_q768_map_dev import infer_map
from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_taper_shift import metric_rows


def run(config: dict, output: Path) -> None:
    if (config["rank"], config["systems"], config["prior_strength"],
        config["coarse_steps"], config["fine_steps"], config["frequency_hz"]) != (
            768, list(range(20268008, 20268016)), 1e-6, 5, 5, 800):
        raise ValueError("fixed conditional check settings changed")
    source = json.loads(Path(config["source_config"]).read_text())
    source["initial_prior_precision"] = config["initial_prior_precision"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    training = []
    for path in config["training_labels"]:
        with np.load(path) as data:
            training.append(data["target32"].astype(float))
    training = np.concatenate(training)
    if training.shape != (1024, 64, 32):
        raise ValueError("unexpected training shape")
    mean, factor = fit_prior(training, config["rank"])
    manifest = {
        "scope": config["scope"], "implementation_sha256": digest(__file__),
        "map_implementation_sha256": digest(infer_map.__code__.co_filename),
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
        started = time.perf_counter()
        beta, history = infer_map(
            spec, observed, context_ranges, source, mean, factor,
            config["prior_strength"], config["coarse_steps"], config["fine_steps"],
        )
        timings.append({"system": system, "inference_seconds": time.perf_counter() - started})
        histories.extend({"system": system, **h} for h in history)
        prediction = propagate_code(
            spec, config["frequency_hz"], ranges, source["decoder_cells"],
            source["decoder_modes"], source["teacher_quadrature_nodes"],
            source["eigh_device"], source["eigh_batch"], beta,
        )
        rows.extend({"information": "LOWBAND_DEPLOYMENT_POSTHOC_CHECK", **r}
                    for r in metric_rows(
                        prediction, truth, ranges, depths, spec, source, system,
                        config["frequency_hz"], "q_pca768_map_1e-06",
                    ))
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        (output / "timing.json").write_text(json.dumps(timings, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for variant in source["source_variants"]:
        sample = [r for r in rows if r["source"] == variant]
        if len(sample) != len(config["systems"]):
            raise ValueError("incomplete check rows")
        summary.append({
            "source": variant, "systems": len(sample),
            "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
            "mean_phase_rad": float(np.mean([
                r["energy_weighted_phase_rmse_rad"] for r in sample
            ])),
            "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
        })
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "mean_inference_seconds": float(np.mean([
            r["inference_seconds"] for r in timings
        ])), "scope": config["scope"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
