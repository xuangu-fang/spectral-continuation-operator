"""Classical source-space representation evidence, explicitly not neural gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(root, output):
    base = json.loads((root / "coherent_contrast_source_rom_20261003/results.json").read_text())
    fine = json.loads(
        (root / "coherent_contrast_source_continuous_20261003/results.json").read_text()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for method, label, color in [
        ("source_aware_field_pde", "Field-space (6 columns)", "#4477aa"),
        ("source_aware_contrast_source_pde", "Propagated source-space (6)", "#228833"),
        ("source_aware_contrast_source_query_truth_oracle", "Source-space truth oracle", "#aa3377"),
    ]:
        selected = [r for r in base if r["method"] == method]
        frequencies = sorted({r["frequency"] for r in selected})
        values = [
            np.mean([r["complex_rel_l2"] for r in selected if r["frequency"] == f])
            for f in frequencies
        ]
        axes[0].plot(
            frequencies, values, "o--" if "oracle" in method else "o-", label=label, color=color
        )
        for system in sorted({r["system"] for r in selected}):
            axes[0].plot(
                frequencies,
                [
                    next(
                        r["complex_rel_l2"]
                        for r in selected
                        if r["frequency"] == f and r["system"] == system
                    )
                    for f in frequencies
                ],
                color=color,
                alpha=0.15,
                linewidth=1,
            )
    axes[0].set(
        title="Classical representation change\n4 development systems; no neural model",
        xlabel="Query frequency (context: 6 / 10 / 14)",
        ylabel="Raw complex relative L2",
        yscale="log",
    )
    axes[0].legend(fontsize=8)
    labels = ["Field-space", "+ one background\ncorrection", "Source-space"]
    methods = [
        "source_aware_field_pde",
        "source_aware_field_pde_one_background_correction",
        "source_aware_contrast_source_pde",
    ]
    for offset, grid, color in [(-0.18, 257, "#4477aa"), (0.18, 513, "#228833")]:
        values = [
            np.mean(
                [r["complex_rel_l2"] for r in fine if r["grid"] == grid and r["method"] == method]
            )
            for method in methods
        ]
        bars = axes[1].bar(
            np.arange(3) + offset, values, width=0.34, label=f"{grid} grid", color=color
        )
        axes[1].bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    axes[1].set(
        title="Independent noninteger-frequency truth\n2 development systems; 3 queries",
        xticks=np.arange(3),
        xticklabels=labels,
        ylabel="Raw complex relative L2",
        ylim=(0, 0.027),
    )
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Accuracy benefit requires a background inverse; its setup and application are not free",
        fontsize=11,
    )
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "contrast_source_classical.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/assets/coherent_continuous_20261003")
    )
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
