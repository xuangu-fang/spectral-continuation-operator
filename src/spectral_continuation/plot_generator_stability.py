"""Actual perturbation differences, not extrapolated linearized error bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(source, output):
    rows = json.loads((source / "results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    colors = {"broad110": "#2878B5", "broad400": "#C44E52", "narrow400": "#DD9A24"}
    names = {
        "broad110": "Original source, 110 Hz",
        "broad400": "Original source, 400 Hz",
        "narrow400": "Narrow source, 400 Hz",
    }
    systems = sorted({row["system"] for row in rows})
    for query, color in colors.items():
        selected = [row for row in rows if row["query"] == query]
        # Bound-limited requested levels can yield the SAME actual pair: plot once.
        unique = {
            (row["system"], row["actual_context_rms"], row["query_normalized_difference"])
            for row in selected
        }
        axes[0].scatter(
            [r[1] for r in unique], [r[2] for r in unique], c=color, label=names[query], s=44
        )
        magnification = [
            next(row["actual_amplification"] for row in selected if row["system"] == system)
            for system in systems
        ]
        axes[1].plot(np.arange(len(systems)), magnification, "o-", color=color, label=names[query])
    domain = np.geomspace(6e-5, 5e-4, 80)
    for multiplier, style in [(1, ":"), (10, "--")]:
        axes[0].plot(domain, multiplier * domain, style, color="0.6", lw=1)
        axes[0].text(
            domain[-1], multiplier * domain[-1], f" {multiplier}x", va="center", color="0.4"
        )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Actual context difference (relative RMS)",
        ylabel="Actual query difference (relative L2)",
        title="Nearby physical systems: actual differences",
    )
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].set(
        xticks=np.arange(len(systems)),
        xticklabels=[str(s) for s in systems],
        xlabel="Two preselected development systems",
        ylabel="Query / context difference",
        title="High-frequency queries amplify small differences",
        ylim=(0, 14),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Local stability diagnostic — not a non-identifiability theorem", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
