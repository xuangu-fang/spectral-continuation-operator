"""Summarize local spectral-token POC runs and paired 110 Hz statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import paired_bootstrap


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("runs must use LABEL=/path/to/summary.json")
    label, path = value.split("=", 1)
    return label, Path(path)


def run(args: argparse.Namespace) -> None:
    runs = {}
    rows = []
    for item in args.run:
        label, path = _parse_run(item)
        summary = json.loads(path.read_text(encoding="utf-8"))
        runs[label] = {"path": path, "summary": summary}
        for split_name, source in (
            ("selection", summary["selection"]),
            ("extrapolation", summary["extrapolation"]),
        ):
            for frequency, metrics in source.items():
                if frequency in {"macro_average", "phase_equivariance_relative_error"}:
                    continue
                rows.append(
                    {
                        "label": label,
                        "split": split_name,
                        "frequency_hz": float(frequency),
                        **metrics,
                    }
                )

    primary = {}
    for label, block in runs.items():
        summary = block["summary"]
        near = [
            summary["extrapolation"][str(frequency)]["complex_rel_l2"]
            for frequency in range(107, 113)
        ]
        primary[label] = {
            "parameters": summary["parameters"],
            "trainable_parameters": summary.get(
                "trainable_parameters", summary["parameters"]
            ),
            "selected_epoch": summary["selected_epoch"],
            "selection_macro_complex_rel_l2": summary["selection"]["macro_average"][
                "complex_rel_l2"
            ],
            "query_107_112_macro_complex_rel_l2": float(np.mean(near)),
            "query_110_complex_rel_l2": summary["extrapolation"]["110"][
                "complex_rel_l2"
            ],
            "query_110_phase_coherence": summary["extrapolation"]["110"][
                "phase_coherence"
            ],
        }

    paired = None
    if args.candidate_label is not None:
        baseline = runs[args.baseline_label]["path"].parent / "validation_metrics.npz"
        candidate = runs[args.candidate_label]["path"].parent / "validation_metrics.npz"
        with np.load(baseline) as baseline_archive, np.load(candidate) as candidate_archive:
            paired = paired_bootstrap(
                baseline_archive["test_110_complex_rel_l2"],
                candidate_archive["test_110_complex_rel_l2"],
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "token_poc_records.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "status": "development POC; strict environments unopened",
        "primary": primary,
        "paired_110_hz_baseline_vs_candidate": paired,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
    }
    (args.output_dir / "token_poc_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _plot(rows, primary, args)


def _plot(rows: list[dict], primary: dict, args: argparse.Namespace) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), constrained_layout=True)
    labels = list(primary)
    for label in labels:
        selected = sorted(
            (row for row in rows if row["label"] == label),
            key=lambda row: row["frequency_hz"],
        )
        frequencies = [row["frequency_hz"] for row in selected]
        axes[0].plot(
            frequencies,
            [row["complex_rel_l2"] for row in selected],
            marker="o",
            markersize=2.5,
            label=label,
        )
        axes[2].plot(
            frequencies,
            [row["phase_coherence"] for row in selected],
            marker="o",
            markersize=2.5,
            label=label,
        )
    for axis in (axes[0], axes[2]):
        axis.axvline(106.5, color="0.5", linestyle=":", linewidth=1)
        axis.grid(alpha=0.2)
    axes[0].set(
        title="(a) Query horizon",
        xlabel="Frequency (Hz)",
        ylabel="Complex RelL2",
        yscale="log",
    )
    axes[0].legend(frameon=False, fontsize=6)

    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(
        x - width / 2,
        [primary[label]["query_107_112_macro_complex_rel_l2"] for label in labels],
        width,
        label="107–112 macro",
    )
    axes[1].bar(
        x + width / 2,
        [primary[label]["query_110_complex_rel_l2"] for label in labels],
        width,
        label="110 Hz",
    )
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set(title="(b) Primary errors", ylabel="Complex RelL2")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].set(
        title="(c) Phase mechanism",
        xlabel="Frequency (Hz)",
        ylabel="Phase coherence",
    )
    figure.savefig(args.output_dir / "token_poc.png", dpi=300)
    figure.savefig(args.output_dir / "token_poc.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
