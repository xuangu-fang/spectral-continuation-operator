"""Amplitude/phase diagnostics for validation-selected response-query models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_domain_query_audit import (
    _fit_subspaces,
    _idw_reciprocity_prediction,
    _kernel_probe_prediction,
    _subspace_prediction,
    source_orders,
)
from .cross_domain_response_data import load_cross_domain_archive


def _prediction_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    truth_norm = np.maximum(np.linalg.norm(truth, axis=-1), 1e-12)
    weight = np.abs(truth)
    weight_sum = np.maximum(weight.sum(axis=-1), 1e-12)
    phase_difference = np.angle(prediction * np.conj(truth))
    return {
        "raw_complex_rel_l2": np.linalg.norm(prediction - truth, axis=-1) / truth_norm,
        "amplitude_rel_l2": np.linalg.norm(
            np.abs(prediction) - np.abs(truth), axis=-1
        )
        / truth_norm,
        "weighted_phase_coherence": (weight * np.cos(phase_difference)).sum(axis=-1)
        / weight_sum,
        "weighted_circular_phase_mae_rad": (weight * np.abs(phase_difference)).sum(axis=-1)
        / weight_sum,
    }


def _baseline(archive, response, train_mean, context, target, frequency, name):
    if name == "train_mean":
        return np.broadcast_to(train_mean[target, frequency], (len(response), response.shape[-1]))
    if name == "nearest_probe":
        distance = np.linalg.norm(archive.sources[context] - archive.sources[target], axis=-1)
        return response[:, context[np.argmin(distance)], frequency]
    if name.startswith("kernel_probe_h"):
        return _kernel_probe_prediction(
            archive,
            response,
            context,
            target,
            frequency,
            float(name.split("_h")[-1]),
        )
    if name.startswith("reciprocity_idw_p"):
        return _idw_reciprocity_prediction(
            archive,
            response,
            context,
            target,
            frequency,
            float(name.split("_p")[-1]),
        )
    raise ValueError(f"unsupported baseline {name}")


def run(dataset: Path, audit_root: Path, output: Path) -> dict:
    archive = load_cross_domain_archive(dataset)
    summary = json.loads((audit_root / "audit_summary.json").read_text(encoding="utf-8"))
    systems = np.flatnonzero(archive.split == "test")
    response = archive.responses[systems]
    train_mean = archive.responses[archive.split == "train"].mean(axis=0)
    max_rank = max(
        int(item["learned"].removeprefix("response_subspace_r").split("_l")[0])
        for item in summary["selection"].values()
    )
    subspaces = _fit_subspaces(archive, max_rank)
    orders = source_orders(archive.responses.shape[1], 3, 20260924)
    result = {}
    for budget_text, methods in summary["selection"].items():
        budget = int(budget_text)
        rank_text, ridge_text = methods["learned"].removeprefix(
            "response_subspace_r"
        ).split("_l")
        totals = {
            label: {
                metric: np.zeros(len(systems), dtype=np.float64)
                for metric in (
                    "raw_complex_rel_l2",
                    "amplitude_rel_l2",
                    "weighted_phase_coherence",
                    "weighted_circular_phase_mae_rad",
                )
            }
            for label in ("baseline", "response_subspace")
        }
        count = 0
        for target in range(archive.responses.shape[1]):
            for order in orders[target]:
                context = order[:budget]
                for frequency in range(archive.responses.shape[2]):
                    truth = response[:, target, frequency]
                    baseline = _baseline(
                        archive,
                        response,
                        train_mean,
                        context,
                        target,
                        frequency,
                        methods["baseline"],
                    )
                    learned = _subspace_prediction(
                        subspaces[frequency],
                        response[:, :, frequency],
                        context,
                        target,
                        archive.responses.shape[-1],
                        int(rank_text),
                        float(ridge_text),
                    )
                    for label, prediction in (("baseline", baseline), ("response_subspace", learned)):
                        metrics = _prediction_metrics(prediction, truth)
                        for metric, value in metrics.items():
                            totals[label][metric] += value
                    count += 1
        result[budget_text] = {
            label: {
                metric: float((value / count).mean()) for metric, value in metrics.items()
            }
            for label, metrics in totals.items()
        }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.audit_root, args.output), indent=2))


if __name__ == "__main__":
    main()
