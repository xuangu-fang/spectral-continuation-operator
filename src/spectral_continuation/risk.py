"""Small dependency-free utilities for horizon-risk calibration."""

from __future__ import annotations

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


class RidgeLogisticRegression:
    """Deterministic standardized logistic regression for small POCs."""

    def __init__(self, ridge: float = 1e-2, iterations: int = 100):
        self.ridge = ridge
        self.iterations = iterations
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> RidgeLogisticRegression:
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if features.ndim != 2 or labels.shape != (len(features),):
            raise ValueError("features or labels have an invalid shape")
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        standardized = (features - self.mean_) / self.scale_
        design = np.column_stack((np.ones(len(features)), standardized))
        positive_rate = np.clip(labels.mean(), 1e-4, 1 - 1e-4)
        coefficients = np.zeros(design.shape[1], dtype=float)
        coefficients[0] = np.log(positive_rate / (1 - positive_rate))
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        for _ in range(self.iterations):
            probability = sigmoid(design @ coefficients)
            weight = np.maximum(probability * (1.0 - probability), 1e-6)
            gradient = design.T @ (probability - labels) + penalty @ coefficients
            hessian = design.T @ (weight[:, None] * design) + penalty
            update = np.linalg.solve(hessian, gradient)
            coefficients -= update
            if np.max(np.abs(update)) < 1e-8:
                break
        self.coefficients_ = coefficients
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coefficients_ is None:
            raise RuntimeError("model must be fitted before prediction")
        standardized = (np.asarray(features, dtype=float) - self.mean_) / self.scale_
        design = np.column_stack((np.ones(len(standardized)), standardized))
        return sigmoid(design @ self.coefficients_)


class RidgeLinearRegression:
    """Deterministic standardized ridge regression for small grouped POCs."""

    def __init__(self, ridge: float = 1.0):
        self.ridge = ridge
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RidgeLinearRegression:
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if features.ndim != 2 or targets.shape != (len(features),):
            raise ValueError("features or targets have an invalid shape")
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        standardized = (features - self.mean_) / self.scale_
        design = np.column_stack((np.ones(len(features)), standardized))
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ targets,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coefficients_ is None:
            raise RuntimeError("model must be fitted before prediction")
        standardized = (np.asarray(features, dtype=float) - self.mean_) / self.scale_
        design = np.column_stack((np.ones(len(standardized)), standardized))
        return design @ self.coefficients_


def average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with deterministic tie handling."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(np.asarray(left))
    right_rank = average_ranks(np.asarray(right))
    if left_rank.std() < 1e-12 or right_rank.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (
            probabilities < edges[index + 1]
            if index < bins - 1
            else probabilities <= edges[index + 1]
        )
        if selected.any():
            value += selected.mean() * abs(probabilities[selected].mean() - labels[selected].mean())
    return float(value)


def risk_coverage_curve(errors: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative mean error after accepting predictions from low to high risk."""
    errors = np.asarray(errors, dtype=float)
    order = np.argsort(np.asarray(risk), kind="mergesort")
    cumulative = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return coverage, cumulative


def risk_coverage_auc(errors: np.ndarray, risk: np.ndarray) -> float:
    coverage, selective_risk = risk_coverage_curve(errors, risk)
    coverage = np.concatenate(([0.0], coverage))
    selective_risk = np.concatenate(([selective_risk[0]], selective_risk))
    return float(np.trapezoid(selective_risk, coverage))


def prefix_horizon_from_risk(
    probabilities: np.ndarray,
    frequencies: np.ndarray,
    reference_frequency: float,
    failure_probability: float,
) -> np.ndarray:
    """Return the last prefix frequency before predicted risk crosses threshold."""
    probabilities = np.asarray(probabilities, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    unsafe = probabilities > failure_probability
    horizons = np.full(probabilities.shape[0], frequencies[-1] - reference_frequency)
    for index, row in enumerate(unsafe):
        positions = np.flatnonzero(row)
        if positions.size:
            first = int(positions[0])
            horizons[index] = 0.0 if first == 0 else frequencies[first - 1] - reference_frequency
    return horizons
