"""True-medium projection capacity control for sparse-response local audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .audit_helmholtz_generator_identification import laplacian, query
from .audit_helmholtz_multiscale_fullfield_classic import public_spec
from .audit_helmholtz_sparse_linear_info import randomized_speed_prior
from .audit_waveguide_spectral_context_robustness import digest
from .coherent_helmholtz_data import sample_environment
from .coherent_representation import complex_metrics


def run(config: dict, output: Path, max_systems: int | None = None) -> None:
    if (config["systems"], config["ranks"], config["query_frequencies_hz"]) != (
            list(range(20269600, 20269608)), [64, 128, 256], [11.3, 17.0]):
        raise ValueError("frozen capacity oracle setting changed")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("resume config mismatch")
    config_path.write_text(json.dumps(config, indent=2))
    with np.load(config["training_medium_labels"], allow_pickle=False) as data:
        labels = data["speed"]
    if labels.shape != (512, 65, 65):
        raise ValueError("training label shape changed")
    basis_path = output / "training_prior_basis.npz"
    if basis_path.exists():
        with np.load(basis_path, allow_pickle=False) as data:
            if str(data["training_sha256"]) != digest(config["training_medium_labels"]):
                raise ValueError("cached basis training labels changed")
            mean, basis, singular = data["mean"], data["basis"], data["singular"]
    else:
        mean, scaled, singular = randomized_speed_prior(
            labels, 256, config["randomized_svd_seed"]
        )
        basis = scaled / np.linalg.norm(scaled.reshape(256, -1), axis=1)[:, None, None]
        np.savez_compressed(basis_path, mean=mean, basis=basis, singular=singular,
                            training_sha256=digest(config["training_medium_labels"]))
    axis65 = np.linspace(0, 1, 65)
    fine = np.linspace(0, 1, 257)
    roi = (fine >= config["roi"][0]) & (fine <= config["roi"][1])
    lap = laplacian(257)
    results_path = output / "results.json"
    rows = json.loads(results_path.read_text()) if results_path.exists() else []
    manifest = {"scope": config["scope"], "implementation_sha256": digest(__file__),
                "training_labels_sha256": digest(config["training_medium_labels"]),
                "basis_sha256": digest(basis_path),
                "truth_sha256": {}, "singular_top": singular[:8].tolist()}
    newly_done = 0
    for system in config["systems"]:
        path = Path(config["dev_truth"]) / f"system_{system}_grid257.npz"
        if sum(r["system"] == system for r in rows) == 8:
            manifest["truth_sha256"][str(path)] = digest(path)
            continue
        if any(r["system"] == system for r in rows):
            raise ValueError("partial system rows cannot be resumed safely")
        if max_systems is not None and newly_done >= max_systems:
            break
        manifest["truth_sha256"][str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            speed = data["speed"].astype(np.float64)
            fields = data["field"]
            frequencies = data["frequencies"]
            spec = json.loads(str(data["spec"]))
        visible = public_spec(spec)
        _, loss, source = sample_environment(visible, fine)
        true65 = speed[::4, ::4]
        coefficients = np.einsum("kij,ij->k", basis, true65 - mean)
        candidates = {"true_speed_grid65_interp": true65}
        for rank in config["ranks"]:
            candidates[f"true_speed_pca{rank}"] = mean + np.einsum(
                "k,kij->ij", coefficients[:rank], basis[:rank]
            )
        for method, speed65 in candidates.items():
            speed257 = RectBivariateSpline(axis65, axis65, speed65)(fine, fine)
            clip_fraction = float(np.mean(speed257 < 0.4))
            speed257 = np.maximum(speed257, 0.4)
            for frequency in config["query_frequencies_hz"]:
                target_idx = int(np.flatnonzero(frequencies == frequency)[0])
                prediction = query(1 / speed257**2, source, loss, frequency, lap)
                rows.append({"system": system, "method": method,
                             "frequency": frequency, "clip_fraction": clip_fraction,
                             "information": "TRUE_TEST_MEDIUM_ORACLE_NONDEPLOYABLE",
                             **complex_metrics(prediction[np.ix_(roi, roi)],
                                               fields[target_idx][np.ix_(roi, roi)])})
        results_path.write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "rows": len(rows)}), flush=True)
        newly_done += 1
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if len(rows) != len(config["systems"]) * 8:
        print(json.dumps({"status": "PARTIAL", "rows": len(rows)}), flush=True)
        return
    summary = []
    for method in candidates:
        for frequency in config["query_frequencies_hz"]:
            sample = [r for r in rows if r["method"] == method and r["frequency"] == frequency]
            summary.append({"method": method, "frequency": frequency,
                            "systems": len(sample),
                            "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                            "mean_phase_rad": float(np.mean([
                                r["energy_weighted_phase_rmse_rad"] for r in sample
                            ])),
                            "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                            "max_clip_fraction": float(np.max([r["clip_fraction"] for r in sample]))})
    (output / "summary.json").write_text(json.dumps({"summary": summary,
                                                     "scope": config["scope"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-systems", type=int)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.max_systems)


if __name__ == "__main__":
    main()
