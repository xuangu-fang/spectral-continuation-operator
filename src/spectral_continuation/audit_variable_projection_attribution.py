"""True-test-medium Euclidean projection ORACLES, never deployment scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_taper_shift import metric_rows
from .learned_speed_manifold import speed_delta, speed_projection


def projected_codes(spec: dict, config: dict, prior: dict) -> tuple[dict, dict]:
    cells, terms, nodes = (
        config["coarse_speed_cells"], config["depth_terms"],
        config["projection_quadrature_nodes"],
    )
    coarse = generator_geometry(spec, cells, terms, modes=32, quadrature_nodes=nodes)
    true_speed = speed_delta(spec, coarse)
    true_q = true_projection(spec, coarse)
    mean_speed, factor_speed = prior["mean_speed"], prior["factor_speed"]
    mean_q, factor_q = prior["mean_q"], prior["factor_q"]
    rank = factor_speed.shape[-1]
    if factor_q.shape[-1] != rank:
        raise ValueError("speed and q per-system coordinates differ")
    coordinates_speed = np.linalg.lstsq(
        factor_speed.reshape(-1, rank), (true_speed - mean_speed).ravel(), rcond=1e-8,
    )[0]
    coordinates_q = np.linalg.lstsq(
        factor_q.reshape(-1, rank), (true_q - mean_q).ravel(), rcond=1e-8,
    )[0]
    speed_coarse = mean_speed + np.einsum(
        "rzk,k->rz", factor_speed, coordinates_speed, optimize=True,
    )
    q_coarse = mean_q + np.einsum("rdk,k->rd", factor_q, coordinates_q, optimize=True)
    fine_cells = config["fine_decoder_cells"]
    fine = generator_geometry(spec, fine_cells, terms, modes=32, quadrature_nodes=nodes)
    speed_beta, _ = speed_projection(
        spec, fine, resample_generator(mean_speed, fine_cells),
        resample_generator(factor_speed, fine_cells), coordinates_speed,
        derivative=False,
    )
    q_beta = resample_generator(q_coarse, fine_cells)
    codes = {"speed_true_medium_euclidean_oracle": speed_beta,
             "q_true_medium_euclidean_oracle": q_beta}
    residuals = {
        "speed_relative_grid_projection_residual": float(
            np.linalg.norm(true_speed - speed_coarse) / np.linalg.norm(true_speed)
        ),
        "q_relative_coefficient_projection_residual": float(
            np.linalg.norm(true_q - q_coarse) / np.linalg.norm(true_q)
        ),
    }
    return codes, residuals


def training_variance(group: dict, prior: dict) -> dict:
    q_arrays = []
    for path in group["q_training_labels"]:
        with np.load(path) as data:
            q_arrays.append(data[group["q_training_array_key"]].astype(float))
    training_q = np.concatenate(q_arrays)
    q_mean, q_factor = prior["mean_q"], prior["factor_q"]
    variance_fraction = (
        (len(training_q) - 1) * np.sum(q_factor**2)
        / np.sum((training_q - q_mean) ** 2)
    )
    speed_manifest = json.loads(Path(group["speed_training_manifest"]).read_text())
    return {
        "training_q_variance_fraction": float(variance_fraction),
        "training_speed_variance_fraction": speed_manifest["speed_prior"][
            "training_speed_variance_fraction"
        ],
        "q_training_fields": len(training_q),
    }


def run(config: dict, output: Path) -> None:
    source = json.loads(Path(config["source_config"]).read_text())
    if (config["frequency_hz"], config["depth_terms"],
        config["coarse_speed_cells"], config["fine_decoder_cells"],
        config["projection_quadrature_nodes"]) != (800, 32, 64, 256, 160):
        raise ValueError("fixed projection diagnostic settings changed")
    if [g["name"] for g in config["groups"]] != [
        "cosine_taper7p5_new16", "gaussian_old8_dev"
    ]:
        raise ValueError("projection groups changed")
    if config["groups"][0]["systems"] != list(range(20268000, 20268016)):
        raise ValueError("cosine confirmation systems changed")
    if config["groups"][1]["systems"] != list(range(20261332, 20261340)):
        raise ValueError("Gaussian development systems changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    manifest = {
        "scope": config["scope"],
        "implementation_sha256": digest(__file__),
        "source_config_sha256": digest(config["source_config"]),
        "groups": {},
    }
    rows, projection_residuals, training_summary = [], [], {}
    for group in config["groups"]:
        with np.load(group["speed_prior"]) as data:
            prior = {key: data[key] for key in data.files}
        if prior["factor_speed"].shape[-1] != group["rank"]:
            raise ValueError("wrong frozen prior rank")
        training_summary[group["name"]] = training_variance(group, prior)
        manifest["groups"][group["name"]] = {
            "speed_prior_sha256": digest(group["speed_prior"]),
            "speed_training_manifest_sha256": digest(group["speed_training_manifest"]),
            "q_training_labels_sha256": {
                path: digest(path) for path in group["q_training_labels"]
            },
            "deployed_results_sha256": digest(group["deployed_results"]),
            "truth_sha256": {},
        }
        for system in group["systems"]:
            path = Path(group["truth_data"]) / f"system{system}_f800.npz"
            manifest["groups"][group["name"]]["truth_sha256"][str(path)] = digest(path)
            with np.load(path) as data:
                spec = json.loads(str(data["spec"]))
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
            codes, residuals = projected_codes(spec, config, prior)
            projection_residuals.append({"family": group["name"], "system": system,
                                         **residuals})
            for method, beta in codes.items():
                prediction = propagate_code(
                    spec, config["frequency_hz"], ranges,
                    config["fine_decoder_cells"], source["decoder_modes"],
                    source["teacher_quadrature_nodes"], source["eigh_device"],
                    source["eigh_batch"], beta,
                )
                cases = metric_rows(
                    prediction, truth, ranges, depths, spec, source, system,
                    config["frequency_hz"], method,
                )
                rows.extend({"family": group["name"], "information": "TRUE_TEST_MEDIUM_ORACLE",
                             **case} for case in cases)
            (output / "results.json").write_text(json.dumps(rows, indent=2))
            (output / "projection_residuals.json").write_text(json.dumps(
                projection_residuals, indent=2,
            ))
            print(json.dumps({"family": group["name"], "system": system,
                              "completed_rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for group in config["groups"]:
        for method in ["speed_true_medium_euclidean_oracle",
                       "q_true_medium_euclidean_oracle"]:
            for source_name in source["source_variants"]:
                sample = [r for r in rows if r["family"] == group["name"]
                          and r["method"] == method and r["source"] == source_name]
                if len(sample) != len(group["systems"]):
                    raise ValueError("missing true-medium projection rows")
                summary.append({
                    "family": group["name"], "method": method,
                    "source": source_name, "systems": len(sample),
                    "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                    "mean_phase_rad": float(np.mean([
                        r["energy_weighted_phase_rmse_rad"] for r in sample
                    ])),
                    "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                })
    checked = {}
    gate = config["mechanism_checks"]
    for family, label in [("cosine_taper7p5_new16", "cosine"),
                          ("gaussian_old8_dev", "gaussian")]:
        for method, short in [("speed_true_medium_euclidean_oracle", "speed"),
                              ("q_true_medium_euclidean_oracle", "q")]:
            values = [r["mean_raw"] for r in summary if r["family"] == family
                      and r["method"] == method]
            key = f"{label}_{short}_projection_mean_raw"
            threshold = gate[f"{key}_{'at_least' if label == 'cosine' and short == 'q' else 'at_most'}"]
            checked[key] = all(x >= threshold for x in values) if (
                label == "cosine" and short == "q"
            ) else all(x <= threshold for x in values)
    (output / "summary.json").write_text(json.dumps({
        "oracle_summary": summary, "mechanism_checks": checked,
        "all_mechanism_checks_pass": all(checked.values()),
        "training_variance": training_summary,
        "mean_test_medium_projection_residual_by_family": {
            group["name"]: {
                key: float(np.mean([r[key] for r in projection_residuals
                                    if r["family"] == group["name"]]))
                for key in ["speed_relative_grid_projection_residual",
                            "q_relative_coefficient_projection_residual"]
            }
            for group in config["groups"]
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
