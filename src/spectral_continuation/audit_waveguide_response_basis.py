"""Linearized response-aware medium basis versus Euclidean medium PCA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import modal_raw, propagate_code, source_columns
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_trajectory_newton import trajectory_jacobian
from .coherent_waveguide_data import spectral_environment


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def training_basis(training):
    mean = training.mean(axis=0)
    _, singular, right = np.linalg.svd(
        (training - mean).reshape(len(training), -1), full_matrices=False
    )
    factor = (right[: len(training) - 1].T * singular[: len(training) - 1]
              / np.sqrt(len(training) - 1)).reshape(*mean.shape, len(training) - 1)
    return mean, factor


def response_rotation(config, source, mean, factor):
    fine_mean = resample_generator(mean, source["decoder_cells"])
    fine_factor = resample_generator(factor, source["decoder_cells"])
    rank = factor.shape[-1]
    gram = np.zeros((rank, rank))
    records = []
    for seed in config["training_background_seeds"]:
        spec = spectral_environment(
            seed, source["spectral_terms"], source["perturbation_rms_m_s"]
        )
        geometry = generator_geometry(
            spec, source["decoder_cells"], config["terms"],
            modes=config["sensitivity_modes"], quadrature_nodes=source["teacher_quadrature_nodes"]
        )
        initials = source_columns(spec, geometry["z"], geometry["weights"], geometry["basis"])
        for variant in config["sensitivity_source_variants"]:
            source_index = source["source_variants"].index(variant)
            fields, derivatives = trajectory_jacobian(
                fine_mean, fine_factor, initials[:, source_index],
                config["sensitivity_frequency_hz"], geometry,
            )
            take = np.arange(0, len(fields), config["sensitivity_stride_cells"])
            take = take[geometry["edges"][take] >= source["roi_min_range_m"]]
            norm = np.linalg.norm(fields[take])
            jacobian = derivatives[take].reshape(-1, rank) / norm
            gram += (jacobian.conj().T @ jacobian).real
            records.append(
                {"training_seed": seed, "source": variant, "linearization_frequency_hz":
                 config["sensitivity_frequency_hz"], "normalized_jacobian_frobenius":
                 float(np.linalg.norm(jacobian))}
            )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order], eigenvalues[order], records


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    complete = json.loads(Path(config["source_results"]).read_text())
    if len(complete) != 20 * len(source["systems"]):
        raise ValueError("original source cohort incomplete")
    if set(config["training_background_seeds"]) & set(source["systems"]):
        raise ValueError("test system used for sensitivity")
    with np.load(config["training_labels"]) as data:
        training = data[f"target{config['terms']}"].astype(float)
    mean, factor = training_basis(training)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "supplemental_manifest.json").write_text(
        json.dumps(
            {
                "registration": config["scope"],
                "source_results_sha256": digest(config["source_results"]),
                "training_labels_sha256": digest(config["training_labels"]),
                "code_sha256": digest(__file__),
            }, indent=2,
        )
    )
    rotation, eigenvalues, sensitivity = response_rotation(config, source, mean, factor)
    np.savez_compressed(
        output / "training_response_basis.npz", mean=mean, factor=factor,
        rotation=rotation[:, : config["rank"]], eigenvalues=eigenvalues,
    )
    (output / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    data_root = Path(source["output_data"])
    rows = []
    flat_factor = factor.reshape(-1, factor.shape[-1])
    column_norm = np.sum(flat_factor**2, axis=0)
    for system in source["systems"]:
        with np.load(data_root / f"system{system}_f{config['query_frequencies'][0]}.npz") as data:
            spec = json.loads(str(data["spec"]))
        geometry = generator_geometry(spec, 64, config["terms"], modes=32, quadrature_nodes=160)
        actual = true_projection(spec, geometry)
        whitened = flat_factor.T @ (actual - mean).ravel() / column_norm
        representations = {
            "pca128": whitened[: config["rank"]],
            "response128": rotation[:, : config["rank"]]
            @ (rotation[:, : config["rank"]].T @ whitened),
        }
        for name, code in representations.items():
            if name == "pca128":
                full_code = np.zeros_like(whitened)
                full_code[: config["rank"]] = code
            else:
                full_code = code
            projected = mean + (flat_factor @ full_code).reshape(mean.shape)
            beta = resample_generator(projected, source["decoder_cells"])
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
                errors = modal_raw(prediction[roi], truth[roi])
                for source_index, variant in enumerate(source["source_variants"]):
                    rows.append(
                        {"system": system, "representation": name,
                         "frequency": frequency, "source": variant,
                         "modal_parseval_raw": float(errors[source_index]),
                         "coefficient_rel_l2": float(
                             np.linalg.norm(actual - projected) / np.linalg.norm(actual)
                         ), "oracle": True}
                    )
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for name in representations:
        for frequency in config["query_frequencies"]:
            for variant in source["source_variants"]:
                sample = [r for r in rows if r["representation"] == name
                          and r["frequency"] == frequency and r["source"] == variant]
                summary.append(
                    {"representation": name, "frequency": frequency, "source": variant,
                     "systems": len(sample), "mean_modal_parseval_raw": float(np.mean(
                         [r["modal_parseval_raw"] for r in sample])),
                     "mean_coefficient_rel_l2": float(np.mean(
                         [r["coefficient_rel_l2"] for r in sample]))}
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
