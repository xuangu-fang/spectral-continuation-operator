"""Create manuscript-style cross-domain response-query figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cross_domain_query_audit import (
    _fit_subspaces,
    _idw_reciprocity_prediction,
    _kernel_probe_prediction,
    _subspace_prediction,
    source_orders,
)
from .cross_domain_response_data import load_cross_domain_archive


def _selected_error_system(root: Path, budget: int, method: str) -> str:
    with (root / "test_system_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["budget"]) == budget and row["method"] == method
        ]
    rows.sort(key=lambda row: float(row["raw_complex_relative_l2"]))
    return rows[len(rows) // 2]["system_id"]


def _baseline_prediction(archive, response, context, target, frequency, name):
    train_mean = archive.responses[archive.split == "train"].mean(axis=0)
    if name == "train_mean":
        return train_mean[target, frequency]
    if name.startswith("reciprocity_idw_p"):
        power = float(name.split("_p")[-1])
        return _idw_reciprocity_prediction(
            archive, response[None], context, target, frequency, power
        )[0]
    if name.startswith("kernel_probe_h"):
        bandwidth = float(name.split("_h")[-1])
        return _kernel_probe_prediction(
            archive, response[None], context, target, frequency, bandwidth
        )[0]
    if name == "nearest_probe":
        distance = np.linalg.norm(archive.sources[context] - archive.sources[target], axis=-1)
        return response[context[np.argmin(distance)], frequency]
    raise ValueError(f"unsupported baseline {name}")


def _load_case(dataset: Path, root: Path):
    archive = load_cross_domain_archive(dataset)
    summary = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
    budget = int(summary["measurement_efficiency"]["half_budget"])
    selection = summary["selection"][str(budget)]
    system_id = _selected_error_system(root, budget, selection["learned"])
    system = int(np.flatnonzero(archive.system_ids == system_id)[0])
    target = archive.responses.shape[1] // 2
    frequency = archive.responses.shape[2] // 2
    context = source_orders(archive.responses.shape[1], 3, 20260924)[target][0][:budget]
    truth = archive.responses[system, target, frequency]
    baseline = _baseline_prediction(
        archive, archive.responses[system], context, target, frequency, selection["baseline"]
    )
    rank_text, ridge_text = selection["learned"].removeprefix(
        "response_subspace_r"
    ).split("_l")
    subspace = _fit_subspaces(archive, int(rank_text))[frequency]
    learned = _subspace_prediction(
        subspace,
        archive.responses[system : system + 1, :, frequency],
        context,
        target,
        archive.responses.shape[-1],
        int(rank_text),
        float(ridge_text),
    )[0]
    return archive, summary, budget, truth, baseline, learned


def plot_budget_curves(cases, output: Path) -> None:
    figure, axes = plt.subplots(1, len(cases), figsize=(12.6, 3.6))
    for axis, (title, _, root) in zip(axes, cases, strict=True):
        summary = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
        budgets = sorted(int(value) for value in summary["test"])
        baseline = [summary["test"][str(value)]["baseline_test_mean"] for value in budgets]
        learned = [summary["test"][str(value)]["learned_test_mean"] for value in budgets]
        axis.plot(budgets, baseline, "o-", color="#6b7280", label="best local baseline")
        axis.plot(budgets, learned, "o-", color="#d95f02", label="response subspace")
        axis.set_yscale("log")
        axis.set_xticks(budgets)
        axis.set_title(title)
        axis.grid(alpha=0.25, which="both")
        axis.set_xlabel("context probes K")
    axes[0].set_ylabel("raw complex relative L2 (log scale)")
    axes[-1].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=240)
    plt.close(figure)


def plot_responses(cases, output: Path) -> None:
    figure, axes = plt.subplots(len(cases), 3, figsize=(12.4, 8.0))
    for row, (title, dataset, root) in enumerate(cases):
        _, _, budget, truth, baseline, learned = _load_case(dataset, root)
        x = np.arange(len(truth))
        scale = max(float(np.max(np.abs(truth))), 1e-12)
        for value, label, color, width in (
            (truth, "truth", "#111827", 2.0),
            (baseline, "baseline", "#6b7280", 1.2),
            (learned, "response subspace", "#d95f02", 1.4),
        ):
            axes[row, 0].plot(x, value.real / scale, label=label, color=color, lw=width)
            axes[row, 1].plot(x, np.abs(value) / scale, label=label, color=color, lw=width)
            phase = np.unwrap(np.angle(value))
            axes[row, 2].plot(x, phase, label=label, color=color, lw=width)
        axes[row, 0].set_ylabel(f"{title}\nK={budget}")
        for axis in axes[row]:
            axis.grid(alpha=0.2)
    axes[0, 0].set_title("normalized real response")
    axes[0, 1].set_title("normalized amplitude")
    axes[0, 2].set_title("unwrapped phase (rad)")
    for axis in axes[-1]:
        axis.set_xlabel("receiver index")
    axes[0, 2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    cases = [
        ("OpenFWI", root / "data/openfwi_curvefault_a_v1.npz", root / "openfwi_m0"),
        ("Deepwave", root / "data/deepwave_ring_v1.npz", root / "deepwave_m0"),
        ("Ocean", root / "data/ocean_multistatic_v1.npz", root / "ocean_m0"),
    ]
    plot_budget_curves(cases, root / "main_budget_curves.png")
    plot_responses(cases, root / "representative_responses.png")


if __name__ == "__main__":
    main()
