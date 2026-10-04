"""Extra-family-knowledge cosine inverse control for synthetic spectral media."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import background_spec, generator_geometry
from .audit_waveguide_new_systems import context_data
from .audit_waveguide_spectral_rank256_inference import infer
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, sound_speed


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def analytic_prior(spec, config):
    """First-order slowness-squared pushforward of the KNOWN 16x16 generator family."""
    r_terms, z_terms = config["range_cosine_terms"], config["depth_cosine_terms"]
    geometry = generator_geometry(spec, 64, config["depth_terms"], modes=32,
                                  quadrature_nodes=160)
    z = geometry["z"]
    background_speed = sound_speed(background_spec(spec), 0, z)
    derivative = -2 * spec["reference_speed_m_s"] ** 2 / background_speed**3
    taper_r = np.exp(-0.5 * (np.arange(r_terms) / 5.0) ** 2)
    taper_z = np.exp(-0.5 * (np.arange(z_terms) / 5.0) ** 2)
    cosine_r_norm = np.where(np.arange(r_terms) == 0, 1.0, 0.5)
    cosine_z_norm = np.where(np.arange(z_terms) == 0, 1.0, 0.5)
    variance = np.outer(taper_r**2 * cosine_r_norm, taper_z**2 * cosine_z_norm)
    variance[0, 0] = 0.0
    speed_scale = spec["spectral_perturbation_rms_m_s"] / np.sqrt(variance.sum())
    range_cos = np.cos(np.pi * geometry["middle"][:, None] /
                       spec["range_m"] * np.arange(r_terms)[None])
    depth_cos = np.cos(np.pi * z[:, None] /
                       spec["depth_m"] * np.arange(z_terms)[None])
    projected_z = ((geometry["weights"] * derivative)[:, None] * depth_cos).T
    projected_z = projected_z @ geometry["psi"] / spec["depth_m"]
    factor = np.einsum("ri,jd,ij->rdij", range_cos, projected_z,
                       speed_scale * np.outer(taper_r, taper_z))
    factor[:, :, 0, 0] = 0.0
    flat = factor.reshape(64, config["depth_terms"], -1)
    keep = [i for i in range(r_terms * z_terms) if i != 0]
    # 255 stochastic coefficients + one harmless zero column would make GN singular.
    # The DC speed coefficient is fixed to zero in data generation; remove it.
    return np.zeros((64, config["depth_terms"])), flat[:, :, keep], speed_scale


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    complete = json.loads(Path(config["source_results"]).read_text())
    if len(complete) != 20 * len(source["systems"]):
        raise ValueError("source cohort incomplete")
    if source["spectral_terms"] != config["range_cosine_terms"] or source[
        "spectral_terms"
    ] != config["depth_cosine_terms"]:
        raise ValueError("known generator family does not match frozen source")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "supplemental_manifest.json").write_text(json.dumps({
        "registration": config["scope"],
        "source_results_sha256": digest(config["source_results"]),
        "code_sha256": digest(__file__),
    }, indent=2))
    rows, histories = [], []
    data_root = Path(source["output_data"])
    for system in config["systems"]:
        with np.load(Path(config["source_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
            saved_observed = data["observed"]
            saved_ranges = data["context_ranges_m"]
        context_ranges, observed = context_data(spec, source)
        np.testing.assert_allclose(context_ranges, saved_ranges, rtol=0, atol=0)
        np.testing.assert_allclose(observed, saved_observed, rtol=2e-6, atol=2e-6)
        mean, factor, speed_scale = analytic_prior(spec, config)
        beta, history = infer(
            spec, observed, context_ranges, source, mean, factor,
            config["coarse_steps"], config["fine_steps"],
        )
        np.savez_compressed(output / f"system{system}_code.npz", beta=beta)
        histories.extend({"system": system, **item} for item in history)
        for frequency in source["query_frequencies"]:
            with np.load(data_root / f"system{system}_f{frequency}.npz") as data:
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
            roi = ranges >= source["roi_min_range_m"]
            prediction = propagate_code(
                spec, frequency, ranges, source["decoder_cells"],
                source["decoder_modes"], source["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"], beta,
            )
            modal = modal_raw(prediction[roi], truth[roi])
            target = (
                depth_basis(depths, spec["depth_m"], truth.shape[1])
                @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)
            ).reshape(len(depths), len(ranges), len(source["source_variants"]))
            field = (
                depth_basis(depths, spec["depth_m"], prediction.shape[1])
                @ prediction.transpose(1, 0, 2).reshape(prediction.shape[1], -1)
            ).reshape(len(depths), len(ranges), len(source["source_variants"]))
            for index, variant in enumerate(source["source_variants"]):
                rows.append({
                    "system": system, "frequency": frequency, "source": variant,
                    "oracle": False,
                    "extra_information": "exact synthetic generator family, not test coefficients",
                    "speed_coefficient_scale_m_s": speed_scale,
                    "latent_dimension": factor.shape[-1],
                    "modal_parseval_raw": float(modal[index]),
                    "context_last_sq_error": history[-1]["context_after"],
                    **complex_metrics(field[:, roi, index], target[:, roi, index]),
                })
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for frequency in source["query_frequencies"]:
        for variant in source["source_variants"]:
            sample = [row for row in rows if row["frequency"] == frequency and
                      row["source"] == variant]
            summary.append({
                "frequency": frequency, "source": variant, "systems": len(sample),
                "complex_rel_l2": float(np.mean([r["complex_rel_l2"] for r in sample])),
                "phase_rmse_rad": float(np.mean([
                    r["energy_weighted_phase_rmse_rad"] for r in sample])),
                "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                "mean_context_last_sq_error": float(np.mean([
                    r["context_last_sq_error"] for r in sample])),
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
