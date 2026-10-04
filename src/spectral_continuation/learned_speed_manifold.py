"""Training-data speed subspace with exact nonlinear speed-to-wave-generator map.

The basis is learned from medium LABELS, not hard-coded as cosine functions.
The physical c^-2 map and one-way wave propagator are known and shared with
the strongest classical controls. This is a bounded representation POC.
"""

from __future__ import annotations

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_generator_identification import (
    background_spec,
    generator_geometry,
    normal_statistics,
)
from .audit_waveguide_generator_prior import conditional_mean
from .audit_waveguide_trajectory_newton import observation_objective, trajectory_jacobian
from .coherent_waveguide_data import sound_speed


def speed_delta(spec: dict, geometry: dict) -> np.ndarray:
    """True medium LABELS only; never call this inside deployment inference."""
    actual = sound_speed(spec, geometry["middle"][:, None], geometry["z"][None])
    background = sound_speed(background_spec(spec), 0, geometry["z"])
    return actual - background[None]


def fit_speed_prior(training: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, dict]:
    values = np.asarray(training, dtype=float)
    if values.ndim != 3 or not (0 < rank < len(values)):
        raise ValueError("expected independent speed fields and lower-dimensional rank")
    mean = values.mean(axis=0)
    centered = (values - mean).reshape(len(values), -1)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    factor = (right[:rank].T * singular[:rank] / np.sqrt(len(values) - 1)).reshape(
        *mean.shape, rank
    )
    variance = singular**2
    summary = {
        "training_fields": len(values),
        "speed_sample_shape": list(mean.shape),
        "rank": rank,
        "training_speed_variance_fraction": float(variance[:rank].sum() / variance.sum()),
        "global_basis_float64_values": int(factor.size + mean.size),
    }
    return mean, factor, summary


def speed_projection(spec: dict, geometry: dict, mean_speed: np.ndarray,
                     factor_speed: np.ndarray, coordinates: np.ndarray,
                     *, derivative: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    if mean_speed.shape != geometry["middle"].shape + geometry["z"].shape:
        raise ValueError("speed mean and physical geometry do not share a grid")
    if factor_speed.shape != (*mean_speed.shape, len(coordinates)):
        raise ValueError("speed factor/code dimensions differ")
    background = sound_speed(background_spec(spec), 0, geometry["z"])
    speed = background[None] + mean_speed + np.einsum(
        "rzk,k->rz", factor_speed, coordinates, optimize=True
    )
    if np.min(speed) <= 1300:
        raise ValueError("invalid nonlinear speed proposal")
    reference = geometry["reference_speed"]
    q = (reference / speed) ** 2 - geometry["background_q"][None]
    beta = (q * geometry["weights"][None]) @ geometry["psi"] / geometry["depth"]
    if not derivative:
        return beta, None
    dqdc = -2 * reference**2 / speed**3
    weighted = dqdc[:, :, None] * factor_speed
    factor = np.einsum(
        "rzk,zd,z->rdk", weighted, geometry["psi"],
        geometry["weights"] / geometry["depth"], optimize=True,
    )
    return beta, factor


def initial_coordinates(spec: dict, observed: np.ndarray, ranges: np.ndarray,
                        source: dict, config: dict, geometry: dict,
                        mean_speed: np.ndarray,
                        factor_speed: np.ndarray) -> np.ndarray:
    rank = factor_speed.shape[-1]
    mean_q, factor_q = speed_projection(
        spec, geometry, mean_speed, factor_speed, np.zeros(rank)
    )
    gram, rhs = normal_statistics(
        spec, observed, source["context_frequencies"], ranges, geometry
    )
    linear_beta = conditional_mean(
        gram, rhs, mean_q, factor_q,
        config["speed_linear_initial_prior_precision"],
    )
    return np.linalg.lstsq(
        factor_q.reshape(-1, rank), (linear_beta - mean_q).ravel(), rcond=1e-8
    )[0]


def speed_step(coordinates: np.ndarray, spec: dict, geometry: dict,
               mean_speed: np.ndarray, factor_speed: np.ndarray,
               observed: np.ndarray, context_ranges: np.ndarray,
               frequencies: list[int], initial: np.ndarray, source: dict,
               indices: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    beta, factor = speed_projection(
        spec, geometry, mean_speed, factor_speed, coordinates
    )
    roi = context_ranges >= source["roi_min_range_m"]
    residuals, derivatives = [], []
    for frequency, truth in zip(frequencies, observed, strict=True):
        prediction, jacobian = trajectory_jacobian(
            beta, factor, initial, frequency, geometry
        )
        if indices is not None:
            prediction, jacobian = prediction[indices], jacobian[indices]
        norm = np.linalg.norm(truth[roi])
        residuals.append(((truth - prediction)[roi] / norm).ravel())
        derivatives.append((jacobian[roi] / norm).reshape(-1, len(coordinates)))
    residual, jacobian = np.concatenate(residuals), np.concatenate(derivatives)
    gram = (jacobian.conj().T @ jacobian).real
    rhs = (jacobian.conj().T @ residual).real
    damping = source["ridge_relative"] * np.trace(gram) / len(gram)
    direction = np.linalg.solve(gram + damping * np.eye(len(gram)), rhs)
    delta = np.einsum("rdk,k->rd", factor, direction)
    safety = min(
        1.0, source["max_coefficient_step"] / max(float(np.max(np.abs(delta))), 1e-20)
    )
    before = float(np.vdot(residual, residual).real / len(frequencies))
    for alpha in source["line_search"]:
        proposal = coordinates + alpha * safety * direction
        try:
            beta_proposal, _ = speed_projection(
                spec, geometry, mean_speed, factor_speed, proposal, derivative=False
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


def infer_speed(spec: dict, observed: np.ndarray, context_ranges: np.ndarray,
                source: dict, config: dict, mean_speed: np.ndarray,
                factor_speed: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    coarse = generator_geometry(
        spec, config["speed_sample_cells"], config["terms"], modes=32,
        quadrature_nodes=config["speed_sample_depth_nodes"],
    )
    coordinates = initial_coordinates(
        spec, observed, context_ranges, source, config, coarse,
        mean_speed, factor_speed,
    )
    initial = observed[0, 0]
    history = []
    for step in range(config["coarse_steps"]):
        coordinates, detail = speed_step(
            coordinates, spec, coarse, mean_speed, factor_speed, observed,
            context_ranges, source["context_frequencies"], initial, source,
        )
        history.append({"stage": "coarse", "step": step + 1, **detail})
    fine = generator_geometry(
        spec, source["decoder_cells"], config["terms"], modes=32,
        quadrature_nodes=config["speed_sample_depth_nodes"],
    )
    fine_mean = resample_generator(mean_speed, source["decoder_cells"])
    fine_factor = resample_generator(factor_speed, source["decoder_cells"])
    indices = np.linspace(0, source["decoder_cells"], source["context_profiles"])
    if not np.array_equal(indices, indices.astype(int)):
        raise ValueError("context profiles do not align with fine cell boundaries")
    for step in range(config["fine_steps"]):
        coordinates, detail = speed_step(
            coordinates, spec, fine, fine_mean, fine_factor, observed,
            context_ranges, source["context_frequencies"], initial, source,
            indices.astype(int),
        )
        history.append({"stage": "fine", "step": step + 1, **detail})
    beta, _ = speed_projection(
        spec, fine, fine_mean, fine_factor, coordinates, derivative=False
    )
    return beta, coordinates, history
