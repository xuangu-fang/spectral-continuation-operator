"""Summarize the formal variable-budget wave set-operator experiment."""

from __future__ import annotations

import argparse
import csv
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


def _load_runs(root: Path) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for representation in REPRESENTATIONS:
        paths = sorted(root.glob(f"{representation}_seed*/set_probe_*.json"))
        if len(paths) != 3:
            raise RuntimeError(f"expected three {representation} runs, found {len(paths)}")
        output[representation] = [json.loads(path.read_text()) for path in paths]
    return output


def _budget_matrix(runs: dict[str, list[dict]]) -> list[dict[str, float | int | str]]:
    rows = []
    for representation, summaries in runs.items():
        for key in summaries[0]["low_group_budgets"]:
            values = np.asarray(
                [summary["low_group_budgets"][key]["contrast_rel_l2_mean"] for summary in summaries]
            )
            source_count, frequency_count = (
                int(part[1:]) for part in key.split("_")
            )
            rows.append(
                {
                    "representation": representation,
                    "source_count": source_count,
                    "frequency_count": frequency_count,
                    "views": source_count * frequency_count,
                    "rel_l2_mean": float(values.mean()),
                    "rel_l2_seed_std": float(values.std()),
                }
            )
    return rows


def _paired_bootstrap(root: Path, samples: int = 20000) -> dict[str, float]:
    records: dict[str, list[dict[int, float]]] = {}
    for representation in REPRESENTATIONS:
        paths = sorted(root.glob(f"{representation}_seed*/set_probe_*_records.csv"))
        records[representation] = []
        for path in paths:
            with path.open() as handle:
                records[representation].append(
                    {
                        int(row["system"]): float(row["contrast_rel_l2"])
                        for row in csv.DictReader(handle)
                    }
                )
    systems = sorted(records["raw"][0])
    raw = np.asarray([[seed[system] for system in systems] for seed in records["raw"]])
    phase = np.asarray(
        [
            [seed[system] for system in systems]
            for seed in records["amplitude_soft_phase_gradient"]
        ]
    )
    generator = np.random.default_rng(20260922)
    reductions = []
    for _ in range(samples):
        selected = generator.integers(0, len(systems), len(systems))
        raw_mean = raw[:, selected].mean()
        phase_mean = phase[:, selected].mean()
        reductions.append((raw_mean - phase_mean) / raw_mean)
    lower, upper = np.quantile(reductions, (0.025, 0.975))
    return {
        "relative_reduction_vs_raw": float((raw.mean() - phase.mean()) / raw.mean()),
        "system_cluster_bootstrap_95_lower": float(lower),
        "system_cluster_bootstrap_95_upper": float(upper),
    }


def _plot(rows: list[dict[str, float | int | str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), constrained_layout=True)
    for representation in REPRESENTATIONS:
        selected = [
            row
            for row in rows
            if row["representation"] == representation and row["source_count"] == 6
        ]
        selected.sort(key=lambda row: int(row["frequency_count"]))
        axes[0].errorbar(
            [row["views"] for row in selected],
            [row["rel_l2_mean"] for row in selected],
            yerr=[row["rel_l2_seed_std"] for row in selected],
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS[representation],
            label=LABELS[representation],
        )
        selected = [
            row
            for row in rows
            if row["representation"] == representation and row["frequency_count"] == 4
        ]
        selected.sort(key=lambda row: int(row["source_count"]))
        axes[1].errorbar(
            [row["views"] for row in selected],
            [row["rel_l2_mean"] for row in selected],
            yerr=[row["rel_l2_seed_std"] for row in selected],
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS[representation],
        )
    axes[0].set_title("More frequencies (6 sources)")
    axes[1].set_title("More sources (4 frequencies)")
    for axis in axes[:2]:
        axis.set_xlabel("Number of response views")
        axis.set_ylabel("Contrast RelL2")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    raw = {
        (int(row["source_count"]), int(row["frequency_count"])): float(row["rel_l2_mean"])
        for row in rows
        if row["representation"] == "raw"
    }
    phase = {
        (int(row["source_count"]), int(row["frequency_count"])): float(row["rel_l2_mean"])
        for row in rows
        if row["representation"] == "amplitude_soft_phase_gradient"
    }
    source_counts = (1, 2, 4, 6)
    frequency_counts = (1, 2, 4)
    gain = np.asarray(
        [
            [100.0 * (raw[(source, frequency)] - phase[(source, frequency)]) / raw[(source, frequency)]
             for frequency in frequency_counts]
            for source in source_counts
        ]
    )
    image = axes[2].imshow(gain, cmap="Blues", vmin=0, vmax=max(40.0, float(gain.max())))
    for row in range(len(source_counts)):
        for column in range(len(frequency_counts)):
            axes[2].text(column, row, f"{gain[row, column]:.0f}%", ha="center", va="center")
    axes[2].set_xticks(range(len(frequency_counts)), frequency_counts)
    axes[2].set_yticks(range(len(source_counts)), source_counts)
    axes[2].set_xlabel("Frequencies")
    axes[2].set_ylabel("Sources")
    axes[2].set_title("Reduction vs raw")
    figure.colorbar(image, ax=axes[2], label="Relative error reduction (%)", fraction=0.046)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = _load_runs(args.root)
    rows = _budget_matrix(runs)
    full = {
        representation: {
            "mean": float(
                np.mean(
                    [run["primary_low_s6_f4"]["contrast_rel_l2_mean"] for run in summaries]
                )
            ),
            "seed_std": float(
                np.std(
                    [run["primary_low_s6_f4"]["contrast_rel_l2_mean"] for run in summaries]
                )
            ),
        }
        for representation, summaries in runs.items()
    }
    summary = {
        "schema_version": 1,
        "task": "variable_budget_wave_set_operator",
        "seeds": [20260922, 20260923, 20260924],
        "full_context": full,
        "phase_vs_raw_paired_bootstrap": _paired_bootstrap(args.root),
        "budget_rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "wave_set_m0_summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output / "wave_set_m0_budget.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.output / "wave_set_budget_scaling.png")
    print(json.dumps({"full_context": full, **summary["phase_vs_raw_paired_bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
