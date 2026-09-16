"""Aggregate frozen-checkpoint evaluations at separately generated real-valued queries."""

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


def run(args: argparse.Namespace) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for specification in args.evaluation:
        label, pattern = specification.split("=", 1)
        paths = sorted(Path(path) for path in glob.glob(pattern))
        if not paths:
            raise RuntimeError(f"no evaluation summaries matched {pattern!r}")
        grouped[label].extend(json.loads(path.read_text(encoding="utf-8")) for path in paths)

    rows = []
    aggregate = {}
    for label, summaries in grouped.items():
        frequencies = tuple(map(float, summaries[0]["queries_hz"]))
        for summary in summaries[1:]:
            if tuple(map(float, summary["queries_hz"])) != frequencies:
                raise RuntimeError(f"query mismatch within {label!r}")
        frequency_metrics = {}
        for frequency in frequencies:
            current = {}
            for metric in METRICS:
                values = np.asarray(
                    [summary["metrics"][f"{frequency:g}"][metric] for summary in summaries]
                )
                current[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            frequency_metrics[f"{frequency:g}"] = current
            for index, summary in enumerate(summaries):
                row = {"label": label, "run_index": index, "frequency_hz": frequency}
                row.update(summary["metrics"][f"{frequency:g}"])
                rows.append(row)
        aggregate[label] = {
            "runs": len(summaries),
            "queries_hz": frequencies,
            "frequency_metrics": frequency_metrics,
            "macro_complex_rel_l2": {
                "mean": float(
                    np.mean([summary["metrics"]["macro_average"]["complex_rel_l2"] for summary in summaries])
                ),
                "std": float(
                    np.std(
                        [summary["metrics"]["macro_average"]["complex_rel_l2"] for summary in summaries],
                        ddof=1,
                    )
                )
                if len(summaries) > 1
                else 0.0,
            },
            "source_checkpoint_sha256": [
                summary["source_checkpoint_sha256"] for summary in summaries
            ],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fractional_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "fractional_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), constrained_layout=True)
    for label, current in aggregate.items():
        frequencies = np.asarray(current["queries_hz"])
        for axis, metric, title in (
            (axes[0], "complex_rel_l2", "Raw complex RelL2"),
            (axes[1], "phase_coherence", "Phase coherence"),
        ):
            means = np.asarray(
                [current["frequency_metrics"][f"{value:g}"][metric]["mean"] for value in frequencies]
            )
            stds = np.asarray(
                [current["frequency_metrics"][f"{value:g}"][metric]["std"] for value in frequencies]
            )
            axis.plot(frequencies, means, marker="o", label=label)
            axis.fill_between(frequencies, means - stds, means + stds, alpha=0.16)
            axis.set(xlabel="Query frequency (Hz)", title=title)
            axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(args.output_dir / "fractional_queries.png", dpi=250)
    figure.savefig(args.output_dir / "fractional_queries.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation", action="append", required=True, help="LABEL=summary glob"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
