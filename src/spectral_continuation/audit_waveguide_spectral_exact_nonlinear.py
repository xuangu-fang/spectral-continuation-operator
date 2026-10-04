"""Extra-family-information nonlinear speed inverse on old spectral development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import (
    background_spec,
    generator_geometry,
    normal_statistics,
)
from .audit_waveguide_generator_prior import conditional_mean
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_exact_basis import analytic_prior
from .audit_waveguide_trajectory_newton import observation_objective, trajectory_jacobian
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, sound_speed


def exact_family_projection(spec, geometry, coordinates, speed_scale, taper_scale=5.0):
    """Project exact nonlinear c^-2 and differentiate w.r.t. public Fourier coordinates."""
    terms = int(np.sqrt(len(coordinates) + 1))
    if len(coordinates) != terms**2 - 1:
        raise ValueError("wrong exact-family coordinate dimension")
    indices = np.arange(terms)
    if taper_scale <= 0:
        raise ValueError("positive taper scale required")
    taper = np.exp(-0.5 * (indices / taper_scale) ** 2)
    scale = speed_scale * np.outer(taper, taper)
    coefficients = np.r_[0.0, coordinates].reshape(terms, terms) * scale
    range_cos = np.cos(
        np.pi * geometry["middle"][:, None] / spec["range_m"] * indices[None]
    )
    depth_cos = np.cos(
        np.pi * geometry["z"][:, None] / spec["depth_m"] * indices[None]
    )
    background = sound_speed(background_spec(spec), 0, geometry["z"])
    speed = background[None] + range_cos @ coefficients @ depth_cos.T
    if np.min(speed) <= 1300:
        raise ValueError("invalid speed proposal")
    reference = geometry["reference_speed"]
    q = (reference / speed) ** 2 - geometry["background_q"][None]
    beta = (q * geometry["weights"][None]) @ geometry["psi"] / geometry["depth"]
    dq = -2 * reference**2 / speed**3
    projected_depth = np.einsum(
        "rz,zj,zd,z->rjd", dq, depth_cos, geometry["psi"],
        geometry["weights"] / geometry["depth"], optimize=True,
    )
    factor = np.einsum(
        "ri,rjd,ij->rdij", range_cos, projected_depth, scale, optimize=True
    ).reshape(len(beta), beta.shape[1], terms**2)[:, :, 1:]
    return beta, factor


def initial_coordinates(spec, observed, ranges, source, config, geometry):
    mean, factor, speed_scale = analytic_prior(spec, config)
    gram, rhs = normal_statistics(
        spec, observed, source["context_frequencies"], ranges, geometry
    )
    linear_beta = conditional_mean(
        gram, rhs, mean, factor, config["linear_initial_prior_precision"]
    )
    coordinates = np.linalg.lstsq(
        factor.reshape(-1, factor.shape[-1]), linear_beta.ravel(), rcond=1e-8
    )[0]
    return coordinates, speed_scale


def exact_step(coordinates, speed_scale, spec, geometry, observed, context_ranges,
               frequencies, initial, source, indices=None, taper_scale=5.0):
    beta, factor = exact_family_projection(
        spec, geometry, coordinates, speed_scale, taper_scale
    )
    roi = context_ranges >= source["roi_min_range_m"]
    residuals, derivatives = [], []
    for frequency, truth in zip(frequencies, observed, strict=True):
        predicted, jacobian = trajectory_jacobian(
            beta, factor, initial, frequency, geometry
        )
        if indices is not None:
            predicted, jacobian = predicted[indices], jacobian[indices]
        norm = np.linalg.norm(truth[roi])
        residuals.append(((truth - predicted)[roi] / norm).ravel())
        derivatives.append((jacobian[roi] / norm).reshape(-1, len(coordinates)))
    residual, jacobian = np.concatenate(residuals), np.concatenate(derivatives)
    gram = (jacobian.conj().T @ jacobian).real
    rhs = (jacobian.conj().T @ residual).real
    damping = source["ridge_relative"] * np.trace(gram) / len(gram)
    direction = np.linalg.solve(gram + damping * np.eye(len(gram)), rhs)
    projected_step = np.einsum("rdk,k->rd", factor, direction)
    safety = min(
        1.0,
        source["max_coefficient_step"] / max(float(np.max(np.abs(projected_step))), 1e-20),
    )
    before = float(np.vdot(residual, residual).real / len(frequencies))
    for alpha in source["line_search"]:
        proposal = coordinates + alpha * safety * direction
        try:
            beta_proposal, _ = exact_family_projection(
                spec, geometry, proposal, speed_scale, taper_scale
            )
            after = observation_objective(
                beta_proposal, initial, geometry, frequencies, observed, roi, indices
            )
        except ValueError:
            after = float("inf")
        if after < before:
            return proposal, {"context_before": before, "context_after": after,
                              "alpha": alpha * safety, "ridge_absolute": float(damping)}
    return coordinates.copy(), {"context_before": before, "context_after": before,
                                "alpha": 0.0, "ridge_absolute": float(damping)}


def infer(spec, observed, context_ranges, source, config):
    taper_scale = float(config.get("taper_scale", 5.0))
    coarse = generator_geometry(spec, 64, config["depth_terms"], modes=32)
    coordinates, speed_scale = initial_coordinates(
        spec, observed, context_ranges, source, config, coarse
    )
    initial = observed[0, 0]
    history = []
    for step in range(config["coarse_steps"]):
        coordinates, diagnostic = exact_step(
            coordinates, speed_scale, spec, coarse, observed, context_ranges,
            source["context_frequencies"], initial, source,
            taper_scale=taper_scale,
        )
        history.append({"stage": "coarse", "step": step + 1, **diagnostic})
    fine = generator_geometry(spec, source["decoder_cells"], config["depth_terms"], modes=32)
    indices = np.linspace(0, source["decoder_cells"], source["context_profiles"])
    if not np.array_equal(indices, indices.astype(int)):
        raise ValueError("context observations must align with fine propagation edges")
    for step in range(config["fine_steps"]):
        coordinates, diagnostic = exact_step(
            coordinates, speed_scale, spec, fine, observed, context_ranges,
            source["context_frequencies"], initial, source, indices.astype(int),
            taper_scale=taper_scale,
        )
        history.append({"stage": "fine", "step": step + 1, **diagnostic})
    beta, _ = exact_family_projection(spec, fine, coordinates, speed_scale, taper_scale)
    return beta, coordinates, history


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    if source["spectral_terms"] != config["range_cosine_terms"] or source[
        "spectral_terms"
    ] != config["depth_cosine_terms"]:
        raise ValueError("extra-known generator family mismatches data")
    if not set(config["systems"]) <= set(source["systems"]):
        raise ValueError("outside old development cohort")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "source_config_sha256": digest(config["source_config"]),
        "implementation_sha256": digest(__file__),
    }, indent=2))
    rows, histories = [], []
    for system in config["systems"]:
        with np.load(Path(config["source_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
            observed = data["observed"]
            context_ranges = data["context_ranges_m"]
        beta, coordinates, history = infer(spec, observed, context_ranges, source, config)
        np.savez_compressed(output / f"system{system}_coordinates.npz", coordinates=coordinates)
        histories.extend({"system": system, **item} for item in history)
        for frequency in source["query_frequencies"]:
            with np.load(Path(source["output_data"]) / f"system{system}_f{frequency}.npz") as data:
                truth, ranges, depths = (
                    data["modal_coefficients"], data["ranges_m"], data["depths_m"]
                )
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
                    "extra_information": "known exact 16x16 speed family, not test coefficients",
                    "context_last_sq_error": history[-1]["context_after"],
                    "modal_parseval_raw": float(modal[index]),
                    **complex_metrics(field[:, roi, index], target[:, roi, index]),
                })
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for frequency in source["query_frequencies"]:
        for variant in source["source_variants"]:
            sample = [r for r in rows if r["frequency"] == frequency
                      and r["source"] == variant]
            summary.append({
                "frequency": frequency, "source": variant, "systems": len(sample),
                "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                "mean_phase_rad": float(np.mean([
                    r["energy_weighted_phase_rmse_rad"] for r in sample
                ])),
                "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                "context_last_sq_error": float(np.mean([
                    r["context_last_sq_error"] for r in sample
                ])),
            })
    at_800 = [r for r in summary if r["frequency"] == 800]
    gate = (len(at_800) == len(source["source_variants"])
            and all(r["mean_raw"] <= config["gate_800hz_both_sources_mean_raw_at_most"]
                    and r["max_raw"] <= config["gate_800hz_both_sources_max_raw_at_most"]
                    for r in at_800))
    (output / "summary.json").write_text(json.dumps({
        "rows": summary, "predeclared_gate_passed": gate,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
