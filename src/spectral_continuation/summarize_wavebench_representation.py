"""Aggregate the three-seed WaveBench representation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPRESENTATIONS = ("raw", "amplitude", "amplitude_soft_phase_gradient")
LABELS = {
    "raw": "Raw complex",
    "amplitude": "Amplitude only",
    "amplitude_soft_phase_gradient": "Amplitude + phase advance",
}
COLORS = {
    "raw": "#5B6770",
    "amplitude": "#E69F00",
    "amplitude_soft_phase_gradient": "#0072B2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runs(roots: list[Path]) -> dict[str, list[dict]]:
    output = {}
    for representation in REPRESENTATIONS:
        paths = []
        for root in roots:
            paths.extend(root.glob(f"{representation}_seed*.json"))
        paths = sorted(paths)
        if len(paths) != 3:
            raise RuntimeError(f"expected three {representation} runs, found {len(paths)}")
        output[representation] = [json.loads(path.read_text()) for path in paths]
    return output


def _records(roots: list[Path], representation: str) -> np.ndarray:
    paths = []
    for root in roots:
        paths.extend(root.glob(f"{representation}_seed*_records.csv"))
    paths = sorted(paths)
    rows = []
    systems = None
    for path in paths:
        with path.open() as handle:
            current = list(csv.DictReader(handle))
        current_systems = [int(row["system"]) for row in current]
        if systems is None:
            systems = current_systems
        elif current_systems != systems:
            raise RuntimeError("record system ordering differs across seeds")
        rows.append([float(row["full_context_contrast_rel_l2"]) for row in current])
    return np.asarray(rows)


def _bootstrap(roots: list[Path], draws: int = 20_000) -> dict[str, float]:
    raw = _records(roots, "raw")
    phase = _records(roots, "amplitude_soft_phase_gradient")
    rng = np.random.default_rng(20260923)
    reductions = np.empty(draws)
    for draw in range(draws):
        selected = rng.integers(0, raw.shape[1], raw.shape[1])
        raw_mean = raw[:, selected].mean()
        phase_mean = phase[:, selected].mean()
        reductions[draw] = (raw_mean - phase_mean) / raw_mean
    lower, upper = np.quantile(reductions, (0.025, 0.975))
    return {
        "full_context_relative_reduction_vs_raw": float((raw.mean() - phase.mean()) / raw.mean()),
        "system_cluster_bootstrap_95_lower": float(lower),
        "system_cluster_bootstrap_95_upper": float(upper),
    }


def _plot(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    for representation in REPRESENTATIONS:
        selected = [row for row in rows if row["representation"] == representation]
        axes[0].errorbar(
            [row["frequency_budget"] for row in selected],
            [row["contrast_rel_l2_mean"] for row in selected],
            yerr=[row["seed_std"] for row in selected],
            marker="o",
            capsize=3,
            linewidth=2,
            color=COLORS[representation],
            label=LABELS[representation],
        )
    axes[0].set_xlabel("Observed frequencies")
    axes[0].set_ylabel("Wavespeed contrast RelL2")
    axes[0].set_xticks((1, 2, 3, 4))
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Independent WaveBench validation")

    raw = {
        row["frequency_budget"]: row["contrast_rel_l2_mean"]
        for row in rows
        if row["representation"] == "raw"
    }
    phase = {
        row["frequency_budget"]: row["contrast_rel_l2_mean"]
        for row in rows
        if row["representation"] == "amplitude_soft_phase_gradient"
    }
    budgets = (1, 2, 3, 4)
    gains = [100.0 * (raw[budget] - phase[budget]) / raw[budget] for budget in budgets]
    bars = axes[1].bar(budgets, gains, color=COLORS["amplitude_soft_phase_gradient"], width=0.65)
    axes[1].bar_label(bars, labels=[f"{gain:.1f}%" for gain in gains], padding=3, fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Observed frequencies")
    axes[1].set_ylabel("Error reduction vs raw (%)")
    axes[1].set_xticks(budgets)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title("Phase-coordinate gain")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = _runs(args.roots)
    rows = []
    for representation, summaries in runs.items():
        for budget in range(1, 5):
            contrast = np.asarray(
                [
                    summary["frequency_budget"][str(budget)]["contrast_rel_l2_mean"]
                    for summary in summaries
                ]
            )
            physical = np.asarray(
                [
                    summary["frequency_budget"][str(budget)]["physical_rel_l2_mean"]
                    for summary in summaries
                ]
            )
            rows.append(
                {
                    "representation": representation,
                    "frequency_budget": budget,
                    "contrast_rel_l2_mean": float(contrast.mean()),
                    "seed_std": float(contrast.std()),
                    "physical_rel_l2_mean": float(physical.mean()),
                }
            )
    phase_gain = {}
    for budget in range(1, 5):
        raw = next(
            row
            for row in rows
            if row["representation"] == "raw" and row["frequency_budget"] == budget
        )
        phase = next(
            row
            for row in rows
            if row["representation"] == "amplitude_soft_phase_gradient"
            and row["frequency_budget"] == budget
        )
        phase_gain[str(budget)] = float(
            (raw["contrast_rel_l2_mean"] - phase["contrast_rel_l2_mean"])
            / raw["contrast_rel_l2_mean"]
        )
    seed_gains = []
    for raw_summary, phase_summary in zip(
        runs["raw"], runs["amplitude_soft_phase_gradient"], strict=True
    ):
        gains = {}
        for budget in range(1, 5):
            raw_error = raw_summary["frequency_budget"][str(budget)]["contrast_rel_l2_mean"]
            phase_error = phase_summary["frequency_budget"][str(budget)]["contrast_rel_l2_mean"]
            gains[str(budget)] = float((raw_error - phase_error) / raw_error)
        seed_gains.append({"seed": raw_summary["seed"], "gains": gains})
    source_summaries = []
    for root in args.roots:
        for path in sorted(root.glob("*.json")):
            if any(path.name.startswith(f"{name}_seed") for name in REPRESENTATIONS):
                source_summaries.append({"path": str(path), "sha256": _sha256(path)})
    gate_passed = (
        sum(
            phase_gain[str(budget)] >= 0.15
            and all(item["gains"][str(budget)] > 0.0 for item in seed_gains)
            for budget in (1, 2)
        )
        >= 2
    )
    summary = {
        "schema_version": 1,
        "task": "wavebench_variable_frequency_system_identification",
        "seeds": [20260923, 20260924, 20260925],
        "rows": rows,
        "phase_relative_gain_vs_raw": phase_gain,
        "seed_phase_relative_gain_vs_raw": seed_gains,
        "paired_full_context": _bootstrap(args.roots),
        "promotion_gate": {
            "definition": "at least 15% mean gain in two sparse budgets and positive gain in every seed",
            "passed": gate_passed,
        },
        "source_summaries": source_summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "wavebench_representation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    with (args.output / "wavebench_representation_budget.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.output / "wavebench_representation_budget.png")
    print(json.dumps({"phase_gain": phase_gain, **summary["paired_full_context"]}, indent=2))


if __name__ == "__main__":
    main()
