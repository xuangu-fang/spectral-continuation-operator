"""Post-hoc PCA rank sweep with true-medium coefficients; never a deployable method."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    complete = json.loads(Path(config["source_results"]).read_text())
    if len(complete) != 20 * len(source["systems"]):
        raise ValueError("original source cohort is incomplete")
    with np.load(config["training_labels"]) as labels:
        training = labels[f"target{config['terms']}"].astype(float)
    mean = training.mean(axis=0)
    centered = (training - mean).reshape(len(training), -1)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    if max(config["ranks"]) > len(training) - 1:
        raise ValueError("rank exceeds centered training sample count")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "supplemental_manifest.json").write_text(
        json.dumps(
            {
                "registration": config["scope"],
                "source_results_sha256": digest(config["source_results"]),
                "training_labels_sha256": digest(config["training_labels"]),
                "code_sha256": digest(__file__),
                "training_singular_values": singular.tolist(),
            },
            indent=2,
        )
    )
    rows = []
    data_root = Path(source["output_data"])
    for system in source["systems"]:
        with np.load(data_root / f"system{system}_f{config['query_frequencies'][0]}.npz") as data:
            spec = json.loads(str(data["spec"]))
        geometry = generator_geometry(spec, 64, config["terms"], modes=32, quadrature_nodes=160)
        actual = true_projection(spec, geometry)
        offset = (actual - mean).ravel()
        coefficients = right @ offset
        for rank in config["ranks"]:
            projected = mean + (coefficients[:rank] @ right[:rank]).reshape(mean.shape)
            beta = resample_generator(projected, source["decoder_cells"])
            coefficient_error = float(np.linalg.norm(projected - actual) / np.linalg.norm(actual))
            for frequency in config["query_frequencies"]:
                with np.load(data_root / f"system{system}_f{frequency}.npz") as data:
                    truth = data["modal_coefficients"]
                    ranges = data["ranges_m"]
                roi = ranges >= source["roi_min_range_m"]
                prediction = propagate_code(
                    spec, frequency, ranges, source["decoder_cells"],
                    source["decoder_modes"], source["teacher_quadrature_nodes"],
                    source["eigh_device"], source["eigh_batch"], beta,
                )
                for index, variant in enumerate(source["source_variants"]):
                    rows.append(
                        {
                            "system": system,
                            "rank": rank,
                            "frequency": frequency,
                            "source": variant,
                            "modal_parseval_raw": float(modal_raw(
                                prediction[roi][:, :, index : index + 1],
                                truth[roi][:, :, index : index + 1],
                            )[0]),
                            "coefficient_rel_l2": coefficient_error,
                            "oracle": True,
                        }
                    )
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for rank in config["ranks"]:
        for frequency in config["query_frequencies"]:
            for source_name in source["source_variants"]:
                sample = [
                    row for row in rows
                    if row["rank"] == rank and row["frequency"] == frequency
                    and row["source"] == source_name
                ]
                summary.append(
                    {
                        "rank": rank,
                        "frequency": frequency,
                        "source": source_name,
                        "systems": len(sample),
                        "mean_modal_parseval_raw": float(np.mean([
                            row["modal_parseval_raw"] for row in sample
                        ])),
                        "max_modal_parseval_raw": float(np.max([
                            row["modal_parseval_raw"] for row in sample
                        ])),
                        "mean_coefficient_rel_l2": float(np.mean([
                            row["coefficient_rel_l2"] for row in sample
                        ])),
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
