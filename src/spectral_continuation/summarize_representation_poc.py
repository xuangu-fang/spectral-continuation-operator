"""Summarize and plot the oscillatory representation POC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def query_values(summary: dict, section: str, queries: tuple[float, ...]) -> np.ndarray:
    return np.asarray(
        [summary[section][f"{frequency:g}"]["complex_rel_l2"] for frequency in queries]
    )


def run(args):
    root = args.run_root
    synthetic_queries = (20.0, 22.0, 24.0)
    synthetic = {
        "Image U-Net": load(root / "representation_poc_synthetic_unet_wideband_seed42/summary.json"),
        "Stable learned state": load(
            root / "representation_poc_synthetic_recurrence_seed42/summary.json"
        ),
        "Prony state probe": load(
            root / "representation_poc_synthetic_prony_order5_fp64_matched/summary.json"
        ),
    }
    ocean_queries = (105.0, 110.0, 115.0, 120.0, 125.0)
    ocean_groups = {}
    for label, stem in (
        ("Image U-Net", "representation_poc_ocean_unet7_seed"),
        ("Stable learned state", "representation_poc_ocean_recurrence_history_seed"),
    ):
        summaries = [load(root / f"{stem}{seed}/summary.json") for seed in (42, 43, 44)]
        ocean_groups[label] = summaries

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, summary in synthetic.items():
        for resolution, section in (
            ("train-grid", "extrapolation"),
            ("2x-grid", "cross_resolution_extrapolation"),
        ):
            for frequency, value in zip(
                synthetic_queries, query_values(summary, section, synthetic_queries), strict=True
            ):
                rows.append(
                    {
                        "domain": "synthetic",
                        "model": label,
                        "frequency": frequency,
                        "resolution": resolution,
                        "mean_complex_rel_l2": value,
                        "std_complex_rel_l2": 0.0,
                    }
                )
    ocean_payload = {}
    for label, summaries in ocean_groups.items():
        values = np.stack(
            [query_values(summary, "extrapolation", ocean_queries) for summary in summaries]
        )
        ocean_payload[label] = {"mean": values.mean(0), "std": values.std(0)}
        for frequency, mean, std in zip(
            ocean_queries,
            values.mean(0),
            values.std(0),
            strict=True,
        ):
            rows.append(
                {
                    "domain": "ocean",
                    "model": label,
                    "frequency": frequency,
                    "resolution": "native",
                    "mean_complex_rel_l2": mean,
                    "std_complex_rel_l2": std,
                }
            )
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {"Image U-Net": "#8f9aa8", "Stable learned state": "#0072b2", "Prony state probe": "#d55e00"}
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for label, summary in synthetic.items():
        axes[0].plot(
            synthetic_queries,
            query_values(summary, "extrapolation", synthetic_queries),
            marker="o",
            linewidth=2,
            label=label,
            color=colors[label],
        )
    axes[0].set_title("Controlled multipath")
    axes[0].set_xlabel("Unseen query frequency")
    axes[0].set_ylabel("Raw complex RelL2")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    width = 0.24
    positions = np.arange(len(synthetic))
    train_grid = [
        summary["extrapolation"]["macro_average"]["complex_rel_l2"]
        for summary in synthetic.values()
    ]
    double_grid = [
        summary["cross_resolution_extrapolation"]["macro_average"]["complex_rel_l2"]
        for summary in synthetic.values()
    ]
    axes[1].bar(positions - width / 2, train_grid, width, label="training grid")
    axes[1].bar(positions + width / 2, double_grid, width, label="2x grid")
    axes[1].set_xticks(positions, ("U-Net", "learned state", "Prony"), rotation=12)
    axes[1].set_title("Zero-shot spatial resolution")
    axes[1].set_ylabel("20–24 Hz macro RelL2")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    for label, payload in ocean_payload.items():
        axes[2].plot(
            ocean_queries,
            payload["mean"],
            marker="o",
            linewidth=2,
            label=label,
            color=colors[label],
        )
        axes[2].fill_between(
            ocean_queries,
            payload["mean"] - payload["std"],
            payload["mean"] + payload["std"],
            alpha=0.18,
            color=colors[label],
        )
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Ocean Pekeris (3 seeds)")
    axes[2].set_xlabel("Unseen query frequency (Hz)")
    axes[2].set_ylabel("Raw complex RelL2")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(args.output_dir / f"representation_poc.{suffix}", dpi=220)
    plt.close(figure)

    result = {
        "synthetic": {
            label: {
                "native_macro": summary["extrapolation"]["macro_average"]["complex_rel_l2"],
                "double_resolution_macro": summary["cross_resolution_extrapolation"][
                    "macro_average"
                ]["complex_rel_l2"],
                "phase_equivariance_error": summary["extrapolation"][
                    "phase_equivariance_relative_error"
                ],
            }
            for label, summary in synthetic.items()
        },
        "ocean": {
            label: {
                "mean_by_frequency": payload["mean"].tolist(),
                "std_by_frequency": payload["std"].tolist(),
                "macro_mean": float(payload["mean"].mean()),
            }
            for label, payload in ocean_payload.items()
        },
        "run_evidence": {
            "synthetic": {
                label: {
                    "parameters": summary.get("parameters", 0),
                    "selected_epoch": summary.get("selected_epoch"),
                    "checkpoint_sha256": summary.get("checkpoint_sha256"),
                    "lineage": summary.get("lineage"),
                }
                for label, summary in synthetic.items()
            },
            "ocean": {
                label: [
                    {
                        "seed": summary["args"]["seed"],
                        "parameters": summary["parameters"],
                        "selected_epoch": summary["selected_epoch"],
                        "checkpoint_sha256": summary["checkpoint_sha256"],
                        "macro_extrapolation": summary["extrapolation"]["macro_average"][
                            "complex_rel_l2"
                        ],
                        "phase_equivariance_error": summary["extrapolation"][
                            "phase_equivariance_relative_error"
                        ],
                    }
                    for summary in summaries
                ]
                for label, summaries in ocean_groups.items()
            },
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/representation_poc")
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
