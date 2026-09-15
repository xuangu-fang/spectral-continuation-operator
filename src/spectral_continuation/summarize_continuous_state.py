"""Aggregate continuous-state runs and make a frequency-resolved paper figure."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    "complex_rel_l2",
    "amplitude_rel_l2",
    "weighted_phase_mae_rad",
    "phase_coherence",
)


def _load_specs(specs: list[str]) -> dict[str, list[tuple[Path, dict]]]:
    grouped: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for spec in specs:
        if "=" not in spec:
            raise ValueError("each --run must have the form LABEL=GLOB")
        label, pattern = spec.split("=", 1)
        paths = sorted(Path(path) for path in glob.glob(pattern))
        if not paths:
            raise RuntimeError(f"no summaries matched {pattern!r}")
        for path in paths:
            grouped[label].append((path, json.loads(path.read_text(encoding="utf-8"))))
    return dict(grouped)


def run(args) -> None:
    grouped = _load_specs(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    aggregate = {}
    for label, runs in grouped.items():
        by_frequency: dict[float, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        seeds = []
        for path, summary in runs:
            seed = int(summary["args"]["seed"])
            seeds.append(seed)
            for split_name in ("selection", "extrapolation"):
                frequency_key = (
                    "selection_query_frequencies_hz"
                    if split_name == "selection"
                    else "test_query_frequencies_hz"
                )
                for frequency in summary[frequency_key]:
                    values = summary[split_name][f"{frequency:g}"]
                    row = {
                        "label": label,
                        "seed": seed,
                        "split": split_name,
                        "frequency_hz": frequency,
                        "summary_path": str(path),
                    }
                    for metric in METRICS:
                        value = float(values[metric])
                        row[metric] = value
                        by_frequency[float(frequency)][metric].append(value)
                    rows.append(row)
        aggregate[label] = {
            "seeds": seeds,
            "parameters": [int(summary["parameters"]) for _, summary in runs],
            "frequency_metrics": {
                f"{frequency:g}": {
                    metric: {
                        "mean": float(np.mean(values[metric])),
                        "std": float(np.std(values[metric], ddof=1))
                        if len(values[metric]) > 1
                        else 0.0,
                    }
                    for metric in METRICS
                }
                for frequency, values in sorted(by_frequency.items())
            },
        }

    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )

    first_summary = next(iter(grouped.values()))[0][1]
    visible_max = max(first_summary["visible_frequencies_hz"])
    selection = first_summary["selection_query_frequencies_hz"]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)
    panels = (
        ("complex_rel_l2", "Raw complex RelL2"),
        ("amplitude_rel_l2", "Amplitude RelL2"),
        ("weighted_phase_mae_rad", "Weighted phase MAE (rad)"),
    )
    colors = plt.get_cmap("tab10")
    for label_index, (label, current) in enumerate(aggregate.items()):
        frequency = np.asarray(
            sorted(float(value) for value in current["frequency_metrics"])
        )
        for axis, (metric, title) in zip(axes, panels, strict=True):
            mean = np.asarray(
                [current["frequency_metrics"][f"{value:g}"][metric]["mean"] for value in frequency]
            )
            std = np.asarray(
                [current["frequency_metrics"][f"{value:g}"][metric]["std"] for value in frequency]
            )
            color = colors(label_index)
            axis.plot(frequency, mean, marker="o", markersize=3, label=label, color=color)
            if np.any(std):
                axis.fill_between(frequency, mean - std, mean + std, color=color, alpha=0.16)
            axis.set_title(title)
            axis.set_xlabel("Query frequency (Hz)")
            axis.grid(alpha=0.22, linewidth=0.6)
    for axis in axes:
        axis.axvline(visible_max, color="0.25", linestyle=":", linewidth=1)
        axis.axvspan(min(selection), max(selection), color="0.5", alpha=0.08)
    axes[0].set_ylabel("Error")
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(args.output_dir / "continuous_state.png", dpi=250)
    figure.savefig(args.output_dir / "continuous_state.pdf")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Label and summary glob, for example 'Pole=experiments/runs/pole*/summary.json'",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
