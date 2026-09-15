"""Per-environment complex-field metrics and paired uncertainty."""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-8


def relative_squared_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction - target).square().flatten(1).sum(1)
    denominator = target.square().flatten(1).sum(1).clamp_min(EPS)
    return (numerator / denominator).mean()


def metric_vectors(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    pred = torch.complex(prediction[:, 0], prediction[:, 1])
    true = torch.complex(target[:, 0], target[:, 1])
    rel = (
        (pred - true).abs().square().flatten(1).sum(1)
        / true.abs().square().flatten(1).sum(1).clamp_min(EPS)
    ).sqrt()
    amp_rel = (
        (pred.abs() - true.abs()).square().flatten(1).sum(1)
        / true.abs().square().flatten(1).sum(1).clamp_min(EPS)
    ).sqrt()
    phase_difference = torch.angle(pred * torch.conj(true))
    weights = true.abs()
    weight_sum = weights.flatten(1).sum(1).clamp_min(EPS)
    phase_mae = (weights * phase_difference.abs()).flatten(1).sum(1) / weight_sum
    coherence = (weights * torch.cos(phase_difference)).flatten(1).sum(1) / weight_sum
    return {
        "complex_rel_l2": rel.cpu().numpy(),
        "amplitude_rel_l2": amp_rel.cpu().numpy(),
        "weighted_phase_mae_rad": phase_mae.cpu().numpy(),
        "phase_coherence": coherence.cpu().numpy(),
    }


def paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int = 20260915,
    replicates: int = 20_000,
) -> dict[str, float]:
    difference = np.asarray(baseline) - np.asarray(candidate)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(replicates, len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        "mean_absolute_improvement": float(difference.mean()),
        "relative_improvement_percent": float(100 * difference.mean() / np.mean(baseline)),
        "fraction_candidate_better": float(np.mean(difference > 0)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }
