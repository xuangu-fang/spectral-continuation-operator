"""Create a compact figure from the global/patch subspace diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _error(summary: dict, key: str) -> float:
    return summary["metrics"][key]["complex_rel_l2"]["mean"]


def run(args: argparse.Namespace) -> None:
    ocean = _load(args.ocean / "summary.json")
    synthetic = _load(args.synthetic / "summary.json")
    grids = {
        "2x4": _load(args.grid_2x4 / "summary.json"),
        "4x8": ocean,
        "8x16": _load(args.grid_8x16 / "summary.json"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.15), constrained_layout=True)

    frequencies = ocean["query_frequencies_hz"]
    for representation, method, label, style in (
        ("raw", "low_band_span", "Raw low-band span", "--"),
        ("carrier_gauged", "low_band_span", "Gauged low-band span", "-"),
        ("carrier_gauged", "full_band_oracle", "Gauged full-band oracle", ":"),
    ):
        values = [
            _error(ocean, f"{representation}/{method}/rank11/f{frequency:g}")
            for frequency in frequencies
        ]
        axes[0].plot(frequencies, values, style, marker="o", markersize=2.7, label=label)
    axes[0].set(
        title="(a) Subspace rotation",
        xlabel="Query frequency (Hz)",
        ylabel="Complex RelL2",
        yscale="log",
    )
    axes[0].axvline(100, color="0.6", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=0.2)

    ranks = (1, 2, 4, 8, 11)
    for grid, summary in grids.items():
        patch_count = int(grid.split("x")[0]) * int(grid.split("x")[1])
        values = [
            _error(summary, f"raw/patch_span_{grid}/rank{rank}/f110") for rank in ranks
        ]
        axes[1].plot(
            np.asarray(ranks) * patch_count,
            values,
            marker="o",
            label=f"{grid} patches",
        )
    global_values = [
        _error(ocean, f"raw/low_band_span/rank{rank}/f110") for rank in ranks
    ]
    axes[1].plot(ranks, global_values, "k--", marker="o", label="Global")
    axes[1].set(
        title="(b) Locality–capacity trade-off",
        xlabel="Complex coefficients per field",
        ylabel="Complex RelL2 at 110 Hz",
        xscale="log",
        yscale="log",
    )
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(alpha=0.2)

    labels = ("Low-band\nspan", "Patch\nspan", "Full-band\noracle")
    ocean_values = (
        _error(ocean, "carrier_gauged/low_band_span/rank11/f110"),
        _error(ocean, "raw/patch_span_4x8/rank11/f110"),
        _error(ocean, "carrier_gauged/full_band_oracle/rank11/f110"),
    )
    synthetic_values = (
        _error(synthetic, "raw/low_band_span/rank14/f20"),
        _error(synthetic, "raw/patch_span_4x8/rank14/f20"),
        _error(synthetic, "raw/full_band_oracle/rank14/f20"),
    )
    x = np.arange(3)
    width = 0.36
    axes[2].bar(x - width / 2, ocean_values, width, label="Ocean, 110 Hz")
    axes[2].bar(x + width / 2, synthetic_values, width, label="Synthetic, 20 Hz")
    axes[2].set_xticks(x, labels)
    axes[2].set(
        title="(c) Representation floors",
        ylabel="Complex RelL2",
        yscale="log",
    )
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].grid(axis="y", alpha=0.2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "global_subspace_summary.png", dpi=300)
    figure.savefig(args.output_dir / "global_subspace_summary.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocean", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--grid-2x4", type=Path, required=True)
    parser.add_argument("--grid-8x16", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
