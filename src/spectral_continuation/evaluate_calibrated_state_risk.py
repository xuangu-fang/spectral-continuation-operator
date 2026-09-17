"""Cross-validate target-free failure risk after a sparse spectral state reset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .risk import (
    RidgeLogisticRegression,
    average_precision,
    expected_calibration_error,
    roc_auc,
    spearman_correlation,
)

MODEL_FEATURES = {
    "distance_after_reset": ("offset_after_reset_hz",),
    "disagreement_only": ("seed_complex_disagreement",),
    "distance_disagreement": ("offset_after_reset_hz", "seed_complex_disagreement"),
    "disagreement_combo": (
        "offset_after_reset_hz",
        "seed_complex_disagreement",
        "seed_amplitude_disagreement",
        "seed_phase_disagreement",
    ),
}


def _grouped_metrics(
    labels: np.ndarray,
    errors: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float, int]:
    aucs, correlations = [], []
    for group in np.unique(groups):
        selected = groups == group
        current_labels = labels[selected]
        if 0 < current_labels.sum() < len(current_labels):
            aucs.append(roc_auc(current_labels, probabilities[selected]))
        correlation = spearman_correlation(errors[selected], probabilities[selected])
        if np.isfinite(correlation):
            correlations.append(correlation)
    return (
        float(np.mean(aucs)) if aucs else float("nan"),
        float(np.mean(correlations)) if correlations else float("nan"),
        len(aucs),
    )


def run(args: argparse.Namespace) -> None:
    with np.load(args.sequential_errors, allow_pickle=False) as archive:
        environment_ids = archive["environment_ids"].astype(str)
        frequencies = np.asarray(archive["query_frequencies_hz"], dtype=float)
        endpoints = np.asarray(archive["calibration_endpoints_hz"], dtype=float)
        errors = np.asarray(archive["calibrated_error"], dtype=float)
        complex_disagreement = np.asarray(
            archive["calibrated_seed_complex_disagreement"], dtype=float
        )
        amplitude_disagreement = np.asarray(
            archive["calibrated_seed_amplitude_disagreement"], dtype=float
        )
        phase_disagreement = np.asarray(archive["calibrated_seed_phase_disagreement"], dtype=float)

    offset = np.broadcast_to(frequencies[None, None, :] - endpoints[None, :, None], errors.shape)
    valid = offset >= 0.0
    feature_names = (
        "offset_after_reset_hz",
        "seed_complex_disagreement",
        "seed_amplitude_disagreement",
        "seed_phase_disagreement",
    )
    features = np.stack(
        (offset, complex_disagreement, amplitude_disagreement, phase_disagreement), axis=-1
    )
    labels = errors > args.error_threshold
    fold_ids = np.arange(len(environment_ids)) % args.folds
    probability_payload: dict[str, np.ndarray] = {}
    results = {}

    for model_name, selected_names in MODEL_FEATURES.items():
        selected_indices = [feature_names.index(name) for name in selected_names]
        probabilities = np.full(errors.shape, np.nan, dtype=float)
        coefficients = []
        for fold in range(args.folds):
            train_environment = fold_ids != fold
            test_environment = fold_ids == fold
            train_valid = valid & train_environment[:, None, None]
            test_valid = valid & test_environment[:, None, None]
            model = RidgeLogisticRegression(ridge=args.ridge)
            model.fit(features[train_valid][:, selected_indices], labels[train_valid])
            probabilities[test_valid] = model.predict_proba(
                features[test_valid][:, selected_indices]
            )
            coefficients.append(model.coefficients_.tolist())

        flat_label = labels[valid]
        flat_error = errors[valid]
        flat_probability = probabilities[valid]
        accepted = flat_probability <= args.safe_probability
        query_groups = np.broadcast_to(frequencies[None, None, :], errors.shape)[valid]
        offset_groups = offset[valid]
        query_auc, query_spearman, query_count = _grouped_metrics(
            flat_label, flat_error, flat_probability, query_groups
        )
        offset_auc, offset_spearman, offset_count = _grouped_metrics(
            flat_label, flat_error, flat_probability, offset_groups
        )
        results[model_name] = {
            "features": selected_names,
            "failure_auroc": roc_auc(flat_label, flat_probability),
            "failure_auprc": average_precision(flat_label, flat_probability),
            "error_spearman": spearman_correlation(flat_error, flat_probability),
            "brier_score": float(np.mean((flat_probability - flat_label) ** 2)),
            "expected_calibration_error": expected_calibration_error(flat_label, flat_probability),
            "accepted_query_fraction_at_safe_probability": float(accepted.mean()),
            "empirical_safe_rate_at_safe_probability": (
                float(1.0 - flat_label[accepted].mean()) if accepted.any() else float("nan")
            ),
            "within_query_frequency_auroc": query_auc,
            "within_query_frequency_error_spearman": query_spearman,
            "within_query_frequency_count": query_count,
            "within_offset_auroc": offset_auc,
            "within_offset_error_spearman": offset_spearman,
            "within_offset_count": offset_count,
            "fold_coefficients": coefficients,
        }
        probability_payload[f"{model_name}_failure_probability"] = probabilities

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "calibrated_state_risk.npz",
        environment_ids=environment_ids,
        query_frequencies_hz=frequencies,
        calibration_endpoints_hz=endpoints,
        valid=valid,
        errors=errors,
        labels=labels,
        fold_ids=fold_ids,
        **probability_payload,
    )
    output = {
        "schema_version": 1,
        "status": "environment-fold cross-validated post-reset target-free risk",
        "source_sequential_errors": str(args.sequential_errors),
        "environment_count": len(environment_ids),
        "valid_state_query_count": int(valid.sum()),
        "folds": args.folds,
        "error_threshold": args.error_threshold,
        "safe_probability": args.safe_probability,
        "results": results,
    }
    (args.output_dir / "calibrated_state_risk_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    names = list(MODEL_FEATURES)
    axes[0].bar(names, [results[name]["failure_auroc"] for name in names])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set(ylabel="Failure AUROC", title="Post-reset failure ranking", ylim=(0.45, 1.0))
    axes[0].tick_params(axis="x", rotation=24)
    for name in names:
        probability = probability_payload[f"{name}_failure_probability"][valid]
        centers, observed = [], []
        for lower in np.linspace(0.0, 0.9, 10):
            chosen = (probability >= lower) & (probability < lower + 0.1)
            if chosen.any():
                centers.append(float(probability[chosen].mean()))
                observed.append(float(labels[valid][chosen].mean()))
        axes[1].plot(centers, observed, marker="o", label=name)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set(
        xlabel="Predicted failure probability",
        ylabel="Observed failure rate",
        title="Post-reset reliability",
    )
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.6)
    figure.savefig(args.output_dir / "calibrated_state_risk.png", dpi=250)
    figure.savefig(args.output_dir / "calibrated_state_risk.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-errors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--error-threshold", type=float, default=0.5)
    parser.add_argument("--safe-probability", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
