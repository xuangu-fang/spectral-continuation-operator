"""Post-hoc true-medium PCA-span capacity oracle on all richer-medium systems."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    completed = json.loads(Path(config["source_results"]).read_text())
    if len(completed) != 20 * len(source["systems"]):
        raise ValueError("frozen source cohort incomplete")
    if {row["system"] for row in completed} != set(source["systems"]):
        raise ValueError("unexpected source systems")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    prior_path = Path(config["training_priors"])
    with np.load(prior_path) as data:
        priors = {
            terms: (data[f"mean{terms}"], data[f"factor{terms}"])
            for terms in [16, 32]
        }
    (output / "supplemental_manifest.json").write_text(
        json.dumps(
            {
                "registration": "POSTHOC mechanism oracle after frozen16/32 methods failed; full original32 cohort; test medium read ONLY for explicitly labelled oracle",
                "config": config,
                "source_results_sha256": digest(config["source_results"]),
                "training_priors_sha256": digest(prior_path),
                "code_sha256": digest(__file__),
            },
            indent=2,
        )
    )
    rows, projection_rows = [], []
    data_root = Path(source["output_data"])
    for system in source["systems"]:
        with np.load(data_root / f"system{system}_f{source['query_frequencies'][0]}.npz") as data:
            spec = json.loads(str(data["spec"]))
        codes = {}
        for terms, (mean, factor) in priors.items():
            geometry = generator_geometry(
                spec, 64, terms, modes=32, quadrature_nodes=160
            )
            true_code = true_projection(spec, geometry)
            flat = factor.reshape(-1, source["rank"])
            coordinates = np.linalg.lstsq(flat, (true_code - mean).ravel(), rcond=None)[0]
            reconstructed = mean + np.einsum("rdk,k->rd", factor, coordinates)
            projection_rows.append(
                {
                    "system": system,
                    "terms": terms,
                    "coefficient_rel_l2": float(
                        np.linalg.norm(true_code - reconstructed)
                        / np.linalg.norm(true_code)
                    ),
                }
            )
            codes[terms] = resample_generator(reconstructed, source["decoder_cells"])
        for frequency in source["query_frequencies"]:
            with np.load(data_root / f"system{system}_f{frequency}.npz") as data:
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
            roi = ranges >= source["roi_min_range_m"]
            target = (
                depth_basis(depths, spec["depth_m"], truth.shape[1])
                @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)
            ).reshape(len(depths), len(ranges), len(source["source_variants"]))
            for terms, beta in codes.items():
                prediction = propagate_code(
                    spec, frequency, ranges, source["decoder_cells"],
                    source["decoder_modes"], source["teacher_quadrature_nodes"],
                    source["eigh_device"], source["eigh_batch"], beta
                )
                field = (
                    depth_basis(depths, spec["depth_m"], prediction.shape[1])
                    @ prediction.transpose(1, 0, 2).reshape(prediction.shape[1], -1)
                ).reshape(len(depths), len(ranges), len(source["source_variants"]))
                modal = modal_raw(prediction[roi], truth[roi])
                for source_index, variant in enumerate(source["source_variants"]):
                    row = {
                        "system": system,
                        "method": f"rank128_{terms}_oracle",
                        "oracle": True,
                        "frequency": frequency,
                        "source": variant,
                        "modal_parseval_raw": float(modal[source_index]),
                        **complex_metrics(
                            field[:, roi, source_index], target[:, roi, source_index]
                        ),
                    }
                    rows.append(row)
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "projection.json").write_text(json.dumps(projection_rows, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for terms in [16, 32]:
        for frequency in source["query_frequencies"]:
            for variant in source["source_variants"]:
                sample = [
                    row for row in rows
                    if row["method"] == f"rank128_{terms}_oracle"
                    and row["frequency"] == frequency and row["source"] == variant
                ]
                summary.append(
                    {
                        "method": f"rank128_{terms}_oracle",
                        "frequency": frequency,
                        "source": variant,
                        "systems": len(sample),
                        "complex_rel_l2": float(
                            np.mean([row["complex_rel_l2"] for row in sample])
                        ),
                        "energy_weighted_phase_rmse_rad": float(
                            np.mean([row["energy_weighted_phase_rmse_rad"] for row in sample])
                        ),
                        "max_system_raw": float(
                            np.max([row["complex_rel_l2"] for row in sample])
                        ),
                    }
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
