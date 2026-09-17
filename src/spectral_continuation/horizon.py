"""Utilities for instance-conditioned reliable spectral horizons."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def metric_matrix(
    path: Path,
    metric: str = "complex_rel_l2",
    prefixes: Sequence[str] = ("test", "query", "extrapolation"),
) -> tuple[np.ndarray, np.ndarray]:
    """Load one per-environment metric matrix ordered by query frequency."""
    suffix = f"_{metric}"
    with np.load(path, allow_pickle=False) as archive:
        for prefix in prefixes:
            values = []
            pattern = re.compile(rf"^{re.escape(prefix)}_([0-9]+(?:\.[0-9]+)?){re.escape(suffix)}$")
            for key in archive.files:
                match = pattern.match(key)
                if match:
                    values.append((float(match.group(1)), np.asarray(archive[key], dtype=float)))
            if values:
                values.sort(key=lambda item: item[0])
                lengths = {len(item[1]) for item in values}
                if len(lengths) != 1:
                    raise ValueError(f"inconsistent environment counts in {path}")
                frequencies = np.asarray([item[0] for item in values], dtype=float)
                matrix = np.stack([item[1] for item in values], axis=1)
                return frequencies, matrix
    raise ValueError(f"no {metric!r} query vectors found in {path}")


def seed_metric_cube(
    paths: Sequence[Path],
    metric: str = "complex_rel_l2",
) -> tuple[np.ndarray, np.ndarray]:
    """Stack matched metric files as [seed, environment, frequency]."""
    if not paths:
        raise ValueError("at least one metric path is required")
    matrices = []
    reference_frequencies = None
    for path in paths:
        frequencies, matrix = metric_matrix(path, metric=metric)
        if reference_frequencies is None:
            reference_frequencies = frequencies
        elif not np.array_equal(reference_frequencies, frequencies):
            raise ValueError("query frequencies differ across seeds")
        matrices.append(matrix)
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError("environment counts differ across seeds")
    assert reference_frequencies is not None
    return reference_frequencies, np.stack(matrices, axis=0)


def prefix_safe_mask(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Return whether every query up to each frequency remains under threshold."""
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2:
        raise ValueError("errors must have shape [environment, frequency]")
    return np.logical_and.accumulate(errors <= threshold, axis=1)


def reliable_horizon(
    errors: np.ndarray,
    frequencies: np.ndarray,
    reference_frequency: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute conservative prefix horizon, first unsafe frequency, and censoring."""
    errors = np.asarray(errors, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != len(frequencies):
        raise ValueError("errors and frequencies have incompatible shapes")
    if np.any(np.diff(frequencies) <= 0) or frequencies[0] <= reference_frequency:
        raise ValueError("queries must be increasing and above the reference frequency")
    unsafe = errors > threshold
    has_failure = unsafe.any(axis=1)
    first_indices = np.argmax(unsafe, axis=1)
    first_unsafe = np.where(has_failure, frequencies[first_indices], np.nan)
    last_safe = np.full(errors.shape[0], reference_frequency, dtype=float)
    for index in range(errors.shape[0]):
        if has_failure[index]:
            first = int(first_indices[index])
            if first > 0:
                last_safe[index] = frequencies[first - 1]
        else:
            last_safe[index] = frequencies[-1]
    return last_safe - reference_frequency, first_unsafe, ~has_failure


def recovery_after_failure(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Flag curves that become nominally safe after their first threshold failure."""
    errors = np.asarray(errors, dtype=float)
    unsafe = errors > threshold
    recovered = np.zeros(errors.shape[0], dtype=bool)
    for index, row in enumerate(unsafe):
        positions = np.flatnonzero(row)
        if positions.size:
            recovered[index] = np.any(~row[positions[0] + 1 :])
    return recovered


def safe_survival(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Fraction of environments with a safe prefix through each query."""
    return prefix_safe_mask(errors, threshold).mean(axis=0)


def complex_relative_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Complex relative L2 over the final two spatial dimensions."""
    numerator = np.sum(np.abs(prediction - target) ** 2, axis=(-2, -1))
    denominator = np.maximum(np.sum(np.abs(target) ** 2, axis=(-2, -1)), 1e-12)
    return np.sqrt(numerator / denominator)


def target_free_ensemble_features(
    predictions: np.ndarray,
    visible_endpoint: np.ndarray,
    query_frequencies: np.ndarray,
    reference_frequency: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize frozen complex predictions without reading query targets.

    ``predictions`` has shape [seed, environment, frequency, height, width].
    Returned features have shape [environment, frequency, feature].
    """
    predictions = np.asarray(predictions)
    visible_endpoint = np.asarray(visible_endpoint)
    query_frequencies = np.asarray(query_frequencies, dtype=float)
    if predictions.ndim != 5:
        raise ValueError("predictions must have shape [seed, environment, frequency, H, W]")
    if visible_endpoint.shape != predictions.shape[1:2] + predictions.shape[-2:]:
        raise ValueError("visible endpoint shape is incompatible with predictions")
    if len(query_frequencies) != predictions.shape[2]:
        raise ValueError("query frequency count is incompatible with predictions")

    ensemble = predictions.mean(axis=0)
    ensemble_energy = np.maximum(np.sum(np.abs(ensemble) ** 2, axis=(-2, -1)), 1e-12)
    endpoint_energy = np.maximum(np.sum(np.abs(visible_endpoint) ** 2, axis=(-2, -1)), 1e-12)
    disagreement = np.sqrt(
        np.mean(np.sum(np.abs(predictions - ensemble[None]) ** 2, axis=(-2, -1)), axis=0)
        / ensemble_energy
    )
    amplitude_disagreement = np.sqrt(
        np.mean(
            np.sum(
                (np.abs(predictions) - np.abs(ensemble)[None]) ** 2,
                axis=(-2, -1),
            ),
            axis=0,
        )
        / ensemble_energy
    )
    cross = predictions * np.conj(ensemble[None])
    phase_cosine = np.real(cross) / np.maximum(np.abs(cross), 1e-12)
    weights = np.abs(ensemble)
    phase_disagreement = np.mean(
        np.sum(weights[None] * (1.0 - phase_cosine), axis=(-2, -1))
        / np.maximum(np.sum(weights, axis=(-2, -1))[None], 1e-12),
        axis=0,
    )
    norm_ratio = np.sqrt(ensemble_energy / endpoint_energy[:, None])
    rms = np.sqrt(ensemble_energy / np.prod(ensemble.shape[-2:]))
    low_amplitude_fraction = np.mean(np.abs(ensemble) < (0.1 * rms[..., None, None]), axis=(-2, -1))
    gradient_energy = np.zeros_like(ensemble_energy)
    if ensemble.shape[-1] > 1:
        gradient_energy += np.sum(np.abs(np.diff(ensemble, axis=-1)) ** 2, axis=(-2, -1))
    if ensemble.shape[-2] > 1:
        gradient_energy += np.sum(np.abs(np.diff(ensemble, axis=-2)) ** 2, axis=(-2, -1))
    spatial_roughness = np.sqrt(gradient_energy / ensemble_energy)

    previous = np.concatenate((visible_endpoint[:, None], ensemble[:, :-1]), axis=1)
    step_change = complex_relative_error(ensemble, previous)
    curvature = np.zeros_like(step_change)
    if ensemble.shape[1] >= 2:
        first_step = query_frequencies[0] - reference_frequency
        first_slope = (ensemble[:, 0] - visible_endpoint) / first_step
        slopes = [first_slope]
        for index in range(1, len(query_frequencies)):
            delta = query_frequencies[index] - query_frequencies[index - 1]
            slopes.append((ensemble[:, index] - ensemble[:, index - 1]) / delta)
        slopes = np.stack(slopes, axis=1)
        delta_slope = np.diff(slopes, axis=1)
        curvature[:, 1:] = np.sqrt(
            np.sum(np.abs(delta_slope) ** 2, axis=(-2, -1)) / ensemble_energy[:, 1:]
        )

    offset = np.broadcast_to(query_frequencies[None] - reference_frequency, ensemble_energy.shape)
    features = np.stack(
        (
            offset,
            disagreement,
            amplitude_disagreement,
            phase_disagreement,
            norm_ratio,
            low_amplitude_fraction,
            spatial_roughness,
            step_change,
            curvature,
        ),
        axis=-1,
    )
    names = (
        "query_offset_hz",
        "seed_complex_disagreement",
        "seed_amplitude_disagreement",
        "seed_phase_disagreement",
        "predicted_norm_ratio",
        "predicted_low_amplitude_fraction",
        "predicted_spatial_roughness",
        "predicted_step_change",
        "predicted_spectral_curvature",
    )
    return features, names
