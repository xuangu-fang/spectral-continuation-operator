"""Cross-validate lightweight reliable-horizon risk baselines by environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .horizon import reliable_horizon
from .risk import (
    RidgeLogisticRegression,
    average_precision,
    expected_calibration_error,
    prefix_horizon_from_risk,
    risk_coverage_auc,
    risk_coverage_curve,
    roc_auc,
    spearman_correlation,
)

MODEL_FEATURES = {
    "distance_only": ("query_offset_hz",),
    "disagreement_only": ("seed_complex_disagreement",),
    "distance_disagreement": ("query_offset_hz", "seed_complex_disagreement"),
    "physics_combo": (
        "query_offset_hz",
        "seed_complex_disagreement",
        "seed_phase_disagreement",
        "predicted_norm_ratio",
        "predicted_spatial_roughness",
        "predicted_step_change",
        "predicted_spectral_curvature",
    ),
}


def _environment_folds(count: int, folds: int) -> np.ndarray:
    return np.arange(count) % folds


def _within_frequency_metrics(
    labels: np.ndarray, errors: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    aucs, correlations = [], []
    for index in range(labels.shape[1]):
        if 0 < labels[:, index].sum() < labels.shape[0]:
            aucs.append(roc_auc(labels[:, index], probabilities[:, index]))
        correlation = spearman_correlation(errors[:, index], probabilities[:, index])
        if np.isfinite(correlation):
            correlations.append(correlation)
    return {
        "within_frequency_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "within_frequency_auroc_frequency_count": len(aucs),
        "within_frequency_error_spearman": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
    }


def _metrics(
    errors: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    frequencies: np.ndarray,
    reference_frequency: float,
    error_threshold: float,
    safe_probability: float,
) -> dict[str, float]:
    flat_error = errors.reshape(-1)
    flat_label = labels.reshape(-1)
    flat_probability = probabilities.reshape(-1)
    accepted = flat_probability <= safe_probability
    true_horizon = reliable_horizon(errors, frequencies, reference_frequency, error_threshold)[0]
    predicted_horizon = prefix_horizon_from_risk(
        probabilities, frequencies, reference_frequency, safe_probability
    )
    top_count = max(1, int(0.2 * len(flat_label)))
    output = {
        "failure_auroc": roc_auc(flat_label, flat_probability),
        "failure_auprc": average_precision(flat_label, flat_probability),
        "error_spearman": spearman_correlation(flat_error, flat_probability),
        "brier_score": float(np.mean((flat_probability - flat_label) ** 2)),
        "expected_calibration_error": expected_calibration_error(flat_label, flat_probability),
        "risk_coverage_auc": risk_coverage_auc(flat_error, flat_probability),
        "accepted_query_fraction_at_safe_probability": float(accepted.mean()),
        "empirical_safe_rate_at_safe_probability": (
            float(1.0 - flat_label[accepted].mean()) if accepted.any() else float("nan")
        ),
        "horizon_mae_hz": float(np.mean(np.abs(predicted_horizon - true_horizon))),
        "unsafe_overreach_rate": float(np.mean(predicted_horizon > true_horizon)),
        "mean_conservative_horizon_gap_hz": float(np.mean(true_horizon - predicted_horizon)),
        "top_risk_20_percent_failure_recall": float(
            flat_label[np.argsort(flat_probability)[-top_count:]].sum() / max(1, flat_label.sum())
        ),
    }
    output.update(_within_frequency_metrics(labels, errors, probabilities))
    return output


def run(args: argparse.Namespace) -> None:
    with np.load(args.features, allow_pickle=False) as archive:
        feature_names = tuple(archive["feature_names"].astype(str))
        features = np.asarray(archive["features"], dtype=float)
        errors = np.asarray(archive["ensemble_complex_rel_l2"], dtype=float)
        frequencies = np.asarray(archive["query_frequencies_hz"], dtype=float)
        environment_ids = archive["environment_ids"].astype(str)
    labels = errors > args.error_threshold
    fold_ids = _environment_folds(len(environment_ids), args.folds)
    results = {}
    probability_payload = {}

    for model_name, names in MODEL_FEATURES.items():
        indices = [feature_names.index(name) for name in names]
        selected = features[..., indices]
        probabilities = np.empty_like(errors)
        coefficients = []
        for fold in range(args.folds):
            train_environment = fold_ids != fold
            test_environment = fold_ids == fold
            model = RidgeLogisticRegression(ridge=args.ridge)
            model.fit(
                selected[train_environment].reshape(-1, len(indices)),
                labels[train_environment].reshape(-1),
            )
            probabilities[test_environment] = model.predict_proba(
                selected[test_environment].reshape(-1, len(indices))
            ).reshape(test_environment.sum(), len(frequencies))
            coefficients.append(model.coefficients_.tolist())
        current = _metrics(
            errors,
            labels,
            probabilities,
            frequencies,
            args.reference_frequency,
            args.error_threshold,
            args.safe_probability,
        )
        current["features"] = names
        current["fold_coefficients"] = coefficients
        results[model_name] = current
        probability_payload[f"{model_name}_failure_probability"] = probabilities

    baseline_auc = results["distance_only"]["risk_coverage_auc"]
    oracle_auc = risk_coverage_auc(errors.reshape(-1), errors.reshape(-1))
    for current in results.values():
        current["risk_coverage_relative_improvement_over_distance_percent"] = float(
            100 * (baseline_auc - current["risk_coverage_auc"]) / baseline_auc
        )
        current["risk_coverage_oracle_gap_closure_percent"] = float(
            100
            * (baseline_auc - current["risk_coverage_auc"])
            / max(baseline_auc - oracle_auc, 1e-12)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "risk_predictions.npz",
        environment_ids=environment_ids,
        query_frequencies_hz=frequencies,
        errors=errors,
        labels=labels,
        fold_ids=fold_ids,
        **probability_payload,
    )
    output = {
        "schema_version": 1,
        "status": "environment-fold cross-validated target-free risk POC",
        "source_features": str(args.features),
        "environment_count": len(environment_ids),
        "query_count": len(frequencies),
        "folds": args.folds,
        "error_threshold": args.error_threshold,
        "safe_probability": args.safe_probability,
        "reference_frequency_hz": args.reference_frequency,
        "oracle_risk_coverage_auc": oracle_auc,
        "results": results,
    }
    (args.output_dir / "risk_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), constrained_layout=True)
    for model_name in MODEL_FEATURES:
        probability = probability_payload[f"{model_name}_failure_probability"]
        coverage, selective_risk = risk_coverage_curve(errors.ravel(), probability.ravel())
        axes[0].plot(coverage, selective_risk, label=model_name)
        centers, observed = [], []
        for lower in np.linspace(0.0, 0.9, 10):
            chosen = (probability.ravel() >= lower) & (probability.ravel() < lower + 0.1)
            if chosen.any():
                centers.append(probability.ravel()[chosen].mean())
                observed.append(labels.ravel()[chosen].mean())
        axes[1].plot(centers, observed, marker="o", label=model_name)
        true_horizon = reliable_horizon(
            errors, frequencies, args.reference_frequency, args.error_threshold
        )[0]
        predicted_horizon = prefix_horizon_from_risk(
            probability, frequencies, args.reference_frequency, args.safe_probability
        )
        axes[2].scatter(
            true_horizon,
            predicted_horizon,
            s=14,
            alpha=0.55,
            label=model_name,
        )
    axes[0].set(
        xlabel="Accepted query fraction",
        ylabel="Mean accepted RelL2",
        title="Risk--coverage",
    )
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set(
        xlabel="Predicted failure probability",
        ylabel="Observed failure rate",
        title="Reliability",
    )
    maximum = frequencies[-1] - args.reference_frequency
    axes[2].plot([0, maximum], [0, maximum], "k--", linewidth=1)
    axes[2].set(
        xlabel="True horizon (Hz)",
        ylabel="Predicted conservative horizon (Hz)",
        title="Horizon",
    )
    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=7)
    figure.savefig(args.output_dir / "risk_poc.png", dpi=250)
    figure.savefig(args.output_dir / "risk_poc.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--reference-frequency", type=float, required=True)
    parser.add_argument("--error-threshold", type=float, default=0.5)
    parser.add_argument("--safe-probability", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
