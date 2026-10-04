"""True-medium projection oracle for the failed first-order exact-family basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_spectral_exact_basis import analytic_prior
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    basis_config = json.loads(Path(config["analytic_basis_config"]).read_text())
    complete = json.loads(Path(config["source_results"]).read_text())
    if len(complete) != 20 * len(source["systems"]):
        raise ValueError("source cohort incomplete")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    data_root = Path(source["output_data"])
    for system in config["systems"]:
        with np.load(data_root / f"system{system}_f{source['query_frequencies'][0]}.npz") as data:
            spec = json.loads(str(data["spec"]))
        mean, factor, _ = analytic_prior(spec, basis_config)
        geometry = generator_geometry(spec, 64, basis_config["depth_terms"], modes=32,
                                      quadrature_nodes=160)
        true = true_projection(spec, geometry)
        coordinates = np.linalg.lstsq(
            factor.reshape(-1, factor.shape[-1]), (true - mean).ravel(), rcond=None
        )[0]
        projected = mean + np.einsum("rdk,k->rd", factor, coordinates)
        coefficient_error = float(np.linalg.norm(projected - true) / np.linalg.norm(true))
        beta = resample_generator(projected, source["decoder_cells"])
        for frequency in source["query_frequencies"]:
            with np.load(data_root / f"system{system}_f{frequency}.npz") as data:
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
            roi = ranges >= source["roi_min_range_m"]
            predicted = propagate_code(
                spec, frequency, ranges, source["decoder_cells"],
                source["decoder_modes"], source["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"], beta,
            )
            modal = modal_raw(predicted[roi], truth[roi])
            target_field = (
                depth_basis(depths, spec["depth_m"], truth.shape[1])
                @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)
            ).reshape(len(depths), len(ranges), len(source["source_variants"]))
            predicted_field = (
                depth_basis(depths, spec["depth_m"], predicted.shape[1])
                @ predicted.transpose(1, 0, 2).reshape(predicted.shape[1], -1)
            ).reshape(len(depths), len(ranges), len(source["source_variants"]))
            for index, variant in enumerate(source["source_variants"]):
                rows.append({
                    "system": system, "frequency": frequency, "source": variant,
                    "oracle": True, "coefficient_rel_l2": coefficient_error,
                    "modal_parseval_raw": float(modal[index]),
                    **complex_metrics(predicted_field[:, roi, index],
                                      target_field[:, roi, index]),
                })
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for frequency in source["query_frequencies"]:
        for variant in source["source_variants"]:
            sample = [r for r in rows if r["frequency"] == frequency and r["source"] == variant]
            summary.append({
                "frequency": frequency, "source": variant, "systems": len(sample),
                "complex_rel_l2": float(np.mean([r["complex_rel_l2"] for r in sample])),
                "phase_rmse_rad": float(np.mean([
                    r["energy_weighted_phase_rmse_rad"] for r in sample])),
                "mean_coefficient_rel_l2": float(np.mean([
                    r["coefficient_rel_l2"] for r in sample])),
            })
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
