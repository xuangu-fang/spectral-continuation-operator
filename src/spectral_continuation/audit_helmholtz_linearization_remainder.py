"""True-medium extra-information check of a local wave-response Taylor model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .audit_helmholtz_generator_identification import laplacian, query
from .audit_helmholtz_multiscale_fullfield_classic import public_spec
from .audit_helmholtz_sparse_linear_info import tangent_response
from .audit_waveguide_spectral_context_robustness import digest
from .coherent_helmholtz_data import sample_environment
from .coherent_representation import complex_metrics


def run(config: dict, output: Path) -> None:
    if (config["systems"], config["grid"], config["query_frequencies_hz"]) != (
            list(range(20269600, 20269608)), 129, [11.3, 17.0]):
        raise ValueError("frozen local Taylor check changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    with np.load(config["training_medium_labels"], allow_pickle=False) as data:
        labels = data["speed"]
    if labels.shape != (512, 65, 65):
        raise ValueError("training label shape changed")
    coarse = np.linspace(0, 1, 65)
    axis = np.linspace(0, 1, config["grid"])
    mean = RectBivariateSpline(coarse, coarse, labels.mean(axis=0))(axis, axis)
    roi = (axis >= config["roi"][0]) & (axis <= config["roi"][1])
    lap = laplacian(config["grid"])
    rows = []
    manifest = {"scope": config["scope"], "implementation_sha256": digest(__file__),
                "training_labels_sha256": digest(config["training_medium_labels"]),
                "truth_sha256": {}}
    for system in config["systems"]:
        path = Path(config["dev_truth"]) / f"system_{system}_grid257.npz"
        manifest["truth_sha256"][str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            truth_speed = data["speed"][::2, ::2].astype(np.float64)
            fields = data["field"][:, ::2, ::2]
            frequencies = data["frequencies"]
            spec = json.loads(str(data["spec"]))
        _, loss, source = sample_environment(public_spec(spec), axis)
        delta = truth_speed - mean
        for frequency in config["query_frequencies_hz"]:
            target = fields[int(np.flatnonzero(frequencies == frequency)[0])][np.ix_(roi, roi)]
            mean_field, tangent = tangent_response(
                mean, delta[None], loss, source, frequency, lap
            )
            first_order = mean_field + tangent[0]
            true_grid129 = query(1 / truth_speed**2, source, loss, frequency, lap)
            for method, prediction in (("training_mean", mean_field),
                                       ("true_delta_first_order", first_order),
                                       ("direct_true_speed_grid129", true_grid129)):
                rows.append({"system": system, "frequency": frequency,
                             "method": method,
                             "information": "TRUE_TEST_MEDIUM_DIAGNOSTIC_NONDEPLOYABLE",
                             **complex_metrics(prediction[np.ix_(roi, roi)], target)})
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for method in ("training_mean", "true_delta_first_order", "direct_true_speed_grid129"):
        for frequency in config["query_frequencies_hz"]:
            sample = [r for r in rows if r["method"] == method and r["frequency"] == frequency]
            summary.append({"method": method, "frequency": frequency,
                            "systems": len(sample),
                            "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                            "mean_phase_rad": float(np.mean([
                                r["energy_weighted_phase_rmse_rad"] for r in sample
                            ])),
                            "max_raw": float(np.max([r["complex_rel_l2"] for r in sample]))})
    (output / "summary.json").write_text(json.dumps({"summary": summary,
                                                     "scope": config["scope"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
