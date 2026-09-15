"""Aggregate frozen-checkpoint evaluations at unseen fractional frequencies."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .summarize_continuous_state import METRICS


def run(args) -> None:
    grouped = defaultdict(list)
    for spec in args.run:
        if "=" not in spec:
            raise ValueError("each --run must have the form LABEL=GLOB")
        label, pattern = spec.split("=", 1)
        paths = sorted(Path(path) for path in glob.glob(pattern))
        if not paths:
            raise RuntimeError(f"no summaries matched {pattern!r}")
        grouped[label].extend(
            (path, json.loads(path.read_text(encoding="utf-8"))) for path in paths
        )

    rows = []
    aggregate = {}
    for label, runs in grouped.items():
        by_frequency = defaultdict(lambda: defaultdict(list))
        macro_values = {metric: [] for metric in METRICS}
        for path, summary in runs:
            seed = int(Path(summary["source_summary"]).parent.name.rsplit("seed", 1)[1])
            for metric in METRICS:
                macro_values[metric].append(
                    float(summary["metrics"]["macro_average"][metric])
                )
            for frequency in summary["queries_hz"]:
                current = summary["metrics"][f"{frequency:g}"]
                row = {
                    "label": label,
                    "seed": seed,
                    "frequency_hz": frequency,
                    "summary_path": str(path),
                }
                for metric in METRICS:
                    value = float(current[metric])
                    row[metric] = value
                    by_frequency[float(frequency)][metric].append(value)
                rows.append(row)
        aggregate[label] = {
            "checkpoints": [
                {
                    "source_summary": summary["source_summary"],
                    "sha256": summary["source_checkpoint_sha256"],
                    "evaluation_summary": str(path),
                }
                for path, summary in runs
            ],
            "macro_average": {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
                for metric, values in macro_values.items()
            },
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
            }
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)
    panels = (
        ("complex_rel_l2", "Raw complex RelL2"),
        ("amplitude_rel_l2", "Amplitude RelL2"),
        ("weighted_phase_mae_rad", "Weighted phase MAE (rad)"),
    )
    colors = plt.get_cmap("tab10")
    for label_index, (label, current) in enumerate(aggregate.items()):
        frequency_metrics = current["frequency_metrics"]
        frequency = np.asarray(sorted(float(value) for value in frequency_metrics))
        for axis, (metric, title) in zip(axes, panels, strict=True):
            mean = np.asarray(
                [frequency_metrics[f"{value:g}"][metric]["mean"] for value in frequency]
            )
            std = np.asarray(
                [frequency_metrics[f"{value:g}"][metric]["std"] for value in frequency]
            )
            color = colors(label_index)
            axis.errorbar(
                frequency,
                mean,
                yerr=std,
                marker="o",
                capsize=2,
                label=label,
                color=color,
            )
            axis.set_title(title)
            axis.set_xlabel("Unseen fractional query (Hz)")
            axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Error")
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(args.output_dir / "fractional_queries.png", dpi=250)
    figure.savefig(args.output_dir / "fractional_queries.pdf")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=GLOB")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
