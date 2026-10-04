"""Exact trajectory sensitivities along a training-only real generator subspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_generator_identification import latent_from_coefficients
from .audit_waveguide_generator_prior import conditional_mean, fit_prior, unpack_statistics
from .audit_waveguide_spectral_transport import encode_field
from .coarse_waveguide_operator import query_coefficients
from .learned_waveguide_generator import GeneratorEncoder, evaluate, prepare


def trajectory_jacobian(beta, factor, initial, frequency, geometry):
    """Derivatives of all boundary states, including inter-cell propagation."""
    total = geometry["background_q"] + beta @ geometry["psi"].T
    if np.any((total <= 0.65) | (total >= 1.5)):
        raise ValueError("clipped generator: unconstrained derivative invalid")
    mass = geometry["mass0"] + np.einsum("rd,dmn->rmn", beta, geometry["matrices"])
    scale = (2 * np.pi * frequency / geometry["reference_speed"]) ** 2
    values, vectors = np.linalg.eigh(scale * mass - np.diag(geometry["kz2"]))
    root = np.sqrt(values.astype(complex))
    denominator = root[:, :, None] + root[:, None, :]
    if np.min(np.abs(denominator)) < 1e-10:
        raise FloatingPointError("near-cutoff derivative; no silent clipping")
    steps = np.diff(geometry["edges"])
    phase = np.exp((1j * root - geometry["attenuation"]) * steps[:, None])
    difference = 1j * steps[:, None, None] * (root[:, :, None] - root[:, None, :])
    ratio = np.empty_like(difference)
    small = np.abs(difference) < 1e-6
    ratio[small] = 1 + difference[small] / 2 + difference[small] ** 2 / 6
    ratio[~small] = np.expm1(difference[~small]) / difference[~small]
    divided = 1j * steps[:, None, None] * phase[:, None, :] * ratio / denominator
    transformed = scale * np.einsum(
        "rmi,dmn,rnj->rdij", vectors, geometry["matrices"], vectors, optimize=True
    )
    current = np.asarray(initial, complex).copy()
    jacobian = np.zeros((len(initial), factor.shape[-1]), complex)
    fields, derivatives = [current.copy()], [jacobian.copy()]
    for cell in range(len(beta)):
        modal = vectors[cell].T @ current
        local = np.einsum("ij,dij,j->id", divided[cell], transformed[cell], modal, optimize=True)
        jacobian = vectors[cell] @ (
            phase[cell, :, None] * (vectors[cell].T @ jacobian) + local @ factor[cell]
        )
        current = vectors[cell] @ (phase[cell] * modal)
        fields.append(current.copy())
        derivatives.append(jacobian.copy())
    return np.array(fields), np.array(derivatives)


def observation_objective(beta, initial, geometry, frequencies, observed, roi, indices=None):
    latent, clipping = latent_from_coefficients(beta, initial, geometry)
    if clipping:
        return float("inf")
    errors = []
    for frequency, truth in zip(frequencies, observed, strict=True):
        predicted = query_coefficients(latent, frequency, geometry["edges"])
        if indices is not None:
            predicted = predicted[indices]
        errors.append(
            np.linalg.norm(predicted[roi] - truth[roi]) ** 2 / np.linalg.norm(truth[roi]) ** 2
        )
    return float(np.mean(errors))


def newton_step(
    beta, factor, initial, geometry, frequencies, observed, roi, options, prior_coordinates=None
):
    residuals, derivatives = [], []
    for frequency, truth in zip(frequencies, observed, strict=True):
        predicted, jacobian = trajectory_jacobian(beta, factor, initial, frequency, geometry)
        indices = options.get("observation_indices")
        if indices is not None:
            predicted, jacobian = predicted[indices], jacobian[indices]
        norm = np.linalg.norm(truth[roi])
        residuals.append(((truth - predicted)[roi] / norm).ravel())
        derivatives.append((jacobian[roi] / norm).reshape(-1, factor.shape[-1]))
    residual, jacobian = np.concatenate(residuals), np.concatenate(derivatives)
    gram = (jacobian.conj().T @ jacobian).real
    rhs = (jacobian.conj().T @ residual).real
    damping = options["ridge_relative"] * np.trace(gram) / len(gram)
    strength = float(options.get("prior_strength", 0.0))
    if strength < 0 or (strength and prior_coordinates is None):
        raise ValueError("nonnegative Gaussian prior requires full coordinates")
    # Gram sums normalized frequency losses; the objective below averages them.
    penalty = strength * len(frequencies) / len(gram)
    adjusted_rhs = rhs if not strength else rhs - penalty * prior_coordinates
    coordinate_step = np.linalg.solve(gram + (damping + penalty) * np.eye(len(gram)), adjusted_rhs)
    delta = np.einsum("rdk,k->rd", factor, coordinate_step)
    safety = min(1.0, options["max_coefficient_step"] / max(float(np.max(np.abs(delta))), 1e-20))
    original = float(np.vdot(residual, residual).real / len(frequencies))
    prior_before = float(np.mean(prior_coordinates**2)) if prior_coordinates is not None else 0.0
    objective_before = original + strength * prior_before
    for alpha in options["line_search"]:
        proposal = beta + alpha * safety * delta
        value = observation_objective(
            proposal,
            initial,
            geometry,
            frequencies,
            observed,
            roi,
            options.get("observation_indices"),
        )
        prior_after = (
            float(np.mean((prior_coordinates + alpha * safety * coordinate_step) ** 2))
            if prior_coordinates is not None
            else 0.0
        )
        objective_after = value + strength * prior_after
        if objective_after < objective_before:
            return proposal, {
                "context_before": original,
                "context_after": value,
                "alpha": alpha * safety,
                "ridge_absolute": float(damping),
                "prior_coordinate_mean_square_before": prior_before,
                "prior_coordinate_mean_square_after": prior_after,
                "objective_before": objective_before,
                "objective_after": objective_after,
            }
    return beta.copy(), {
        "context_before": original,
        "context_after": original,
        "alpha": 0.0,
        "ridge_absolute": float(damping),
        "prior_coordinate_mean_square_before": prior_before,
        "prior_coordinate_mean_square_after": prior_before,
        "objective_before": objective_before,
        "objective_after": objective_before,
    }


def run(options, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise ValueError("existing experiment output")
    directory = Path(options["training_run"])
    config = json.loads((directory / "config.json").read_text())
    if options.get("independent_audit_manifest"):
        manifest_path = Path(options["independent_audit_manifest"])
        manifest = json.loads(manifest_path.read_text())
        config["validation_systems"] = manifest["systems"]
        config["data"] = str(manifest_path.parent / "original")
        generation = json.loads((manifest_path.parent / "generation.json").read_text())
        original = [row for row in generation if row["variant"] == "original"]
        if {row["system"] for row in original} != set(manifest["systems"]) or len(original) != len(
            manifest["systems"]
        ):
            raise ValueError("complete frozen cohort required for supplemental control")
        for row in original:
            if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("changed independent truth")
        paths = [directory / "training_statistics.npz", directory / "config.json", Path(__file__)]
        (output / "supplemental_manifest.json").write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "registration": "POST-original-freeze strengthened classical control; parameters fixed using original8development, NOT an original preregistered candidate",
                    "systems": manifest["systems"],
                    "original_truth_sha256_verified": True,
                    "source_and_prior_sha256": {
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
                    },
                },
                indent=2,
            )
        )
    if "context_range_samples" in options:
        config["context_range_samples"] = options["context_range_samples"]
        indices = np.linspace(0, config["cells"], config["context_range_samples"])
        if not np.array_equal(indices, indices.astype(int)):
            raise ValueError("observation locations must coincide with cell edges")
        options = {**options, "observation_indices": indices.astype(int).tolist()}
    (output / "config.json").write_text(json.dumps({**options, "base": config}, indent=2))
    torch.set_num_threads(2)
    with np.load(directory / "training_statistics.npz") as z:
        mean, factor = fit_prior(z["target"].astype(float), options["rank"])
    factor_inverse = np.linalg.pinv(factor.reshape(-1, factor.shape[-1]))
    val, metadata = prepare(config, config["validation_systems"], False)
    neural = None
    if {"neural", "projected_neural"} & set(options.get("initializations", ["gaussian", "neural"])):
        checkpoint = torch.load(
            Path(options["field_run"]) / "best.pt", map_location="cpu", weights_only=False
        )
        model = GeneratorEncoder(val["features"].shape[1], **config["model"]).eval()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            neural = model(
                torch.tensor((val["features"] - checkpoint["mean"]) / checkpoint["std"]),
                torch.tensor(val["baseline"]),
            ).numpy()
    prior = []
    for features in val["features"]:
        gram, rhs = unpack_statistics(features, config["depth_terms"])
        prior.append(conditional_mean(gram, rhs, mean, factor, options["initial_prior_precision"]))
    rows, history, caches = [], [], {}
    initializations = [("gaussian", np.array(prior)), ("neural", neural)]
    if "prior_mean" in options.get("initializations", []):
        initializations.append(("prior_mean", np.repeat(mean[None], len(metadata), axis=0)))
    if options.get("include_projected_neural"):
        flat = factor.reshape(-1, factor.shape[-1])
        coordinates = np.linalg.lstsq(flat, (neural - mean).reshape(len(neural), -1).T, rcond=None)[
            0
        ].T
        projected = mean + np.einsum("rdk,bk->brd", factor, coordinates)
        initializations.append(("projected_neural", projected))
        (output / "projection_diagnostic.json").write_text(
            json.dumps(
                {
                    "per_system_removed_coefficient_rmse": np.sqrt(
                        np.mean((neural - projected) ** 2, axis=(1, 2))
                    ).tolist(),
                    "source": "training512prior only; no development medium labels used in projection",
                },
                indent=2,
            )
        )
    for method, initial_beta in initializations:
        if method not in options.get("initializations", [name for name, _ in initializations]):
            continue
        caches[method] = {step: [] for step in options["checkpoints"]}
        for si, item in enumerate(metadata):
            geometry = item["geometry"]
            take = np.linspace(
                0, len(item["ranges"]) - 1, config["context_range_samples"], dtype=int
            )
            observed = np.array(
                [
                    encode_field(
                        item["fields"][int(np.flatnonzero(item["frequencies"] == f)[0])][:, take],
                        item["depths"],
                        geometry["depth"],
                        32,
                    )
                    for f in config["observed_frequencies"]
                ]
            )
            observation_ranges = (
                geometry["edges"][options["observation_indices"]]
                if "observation_indices" in options
                else geometry["edges"]
            )
            roi = observation_ranges >= config["roi_min_range_m"]
            beta = initial_beta[si].astype(float)
            if options.get("prior_strength", 0):
                coordinates = factor_inverse @ (beta - mean).ravel()
                reconstructed = mean + np.einsum("rdk,k->rd", factor, coordinates)
                if not np.allclose(reconstructed, beta, atol=1e-10, rtol=1e-8):
                    raise ValueError(
                        "Gaussian MAP requires initialization in its training subspace"
                    )
            seconds = 0.0
            for step in range(max(options["checkpoints"]) + 1):
                if step in options["checkpoints"]:
                    start = time.perf_counter()
                    current = evaluate(beta[None], [item], config)
                    query_seconds = time.perf_counter() - start
                    rows.extend(
                        {
                            **row,
                            "initialization": method,
                            "step": step,
                            "correction_seconds": seconds,
                            "six_query_and_metrics_seconds": query_seconds,
                        }
                        for row in current
                    )
                    caches[method][step].append(beta.copy())
                if step == max(options["checkpoints"]):
                    break
                start = time.perf_counter()
                beta, diagnostic = newton_step(
                    beta,
                    factor,
                    item["initial"],
                    geometry,
                    config["observed_frequencies"],
                    observed,
                    roi,
                    options,
                    prior_coordinates=(
                        factor_inverse @ (beta - mean).ravel()
                        if options.get("prior_strength", 0)
                        else None
                    ),
                )
                seconds += time.perf_counter() - start
                history.append(
                    {
                        "system": item["system"],
                        "initialization": method,
                        "step": step + 1,
                        "cumulative_correction_seconds": seconds,
                        **diagnostic,
                    }
                )
            (output / "results.json").write_text(json.dumps(rows, indent=2))
            (output / "history.json").write_text(json.dumps(history, indent=2))
            print(
                json.dumps(
                    {
                        "system": item["system"],
                        "initialization": method,
                        "last_context": history[-1]["context_after"],
                        "seconds": seconds,
                    }
                ),
                flush=True,
            )
        for step, coefficients in caches[method].items():
            np.savez_compressed(
                output / f"{method}_step{step}.npz",
                coefficients=np.array(coefficients),
                systems=config["validation_systems"],
            )
    summary = []
    for method in caches:
        for step in options["checkpoints"]:
            selected = [
                row for row in rows if row["initialization"] == method and row["step"] == step
            ]
            summary.append(
                {
                    "initialization": method,
                    "step": step,
                    "raw": float(np.mean([r["complex_rel_l2"] for r in selected])),
                    "mean_correction_seconds": float(
                        np.mean([r["correction_seconds"] for r in selected])
                    ),
                }
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-count", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.context_count is not None:
        config["context_range_samples"] = args.context_count
    run(config, args.output)


if __name__ == "__main__":
    main()
