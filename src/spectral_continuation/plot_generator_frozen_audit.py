"""Independent cohort: preserve frozen comparisons and label later strong controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(root, output):
    paths = {
        "frozen_original": root
        / "coherent_waveguide_frozen_system_summary_20261003/all_methods.json",
        "newton_original": root
        / "coherent_waveguide_frozen_newton_supplement_20261003/summary.json",
        "frozen_narrow": root / "coherent_waveguide_frozen_new_queries_20261003/summary.json",
        "newton_narrow": root / "coherent_waveguide_frozen_newton_narrow_20261003/results.json",
    }
    data = {key: json.loads(path.read_text()) for key, path in paths.items()}
    narrow_newton = []
    for method in sorted({r["method"] for r in data["newton_narrow"]}):
        for frequency in [140, 200, 300, 400]:
            rows = [
                r
                for r in data["newton_narrow"]
                if r["method"] == method and r["frequency"] == frequency
            ]
            if len(rows) != 64 or len({r["system"] for r in rows}) != 64:
                raise ValueError("complete independent cohort required")
            narrow_newton.append(
                {
                    "method": method,
                    "frequency": frequency,
                    "systems": 64,
                    "raw": float(np.mean([r["complex_rel_l2"] for r in rows])),
                    "phase_rmse_rad": float(
                        np.mean([r["energy_weighted_phase_rmse_rad"] for r in rows])
                    ),
                }
            )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    original = data["frozen_original"]
    methods = [
        "kernel_generator",
        "field_neural",
        "gaussian_generator_refine_20",
        "field_neural_refine_20",
    ]
    labels = [
        "Kernel",
        "Neural\n3 seeds",
        "Prior +\nL-BFGS20",
        "Neural +\nL-BFGS20",
        "Prior +\nGN1*",
        "Prior +\nGN5*",
    ]
    values = [float(np.mean([r["raw"] for r in original if r["method"] == m])) for m in methods]
    values += [
        next(r["raw"] for r in data["newton_original"] if r["step"] == step) for step in [1, 5]
    ]
    colors = ["#888888", "#2878B5", "#999999", "#6AA7D4", "#D48124", "#A85212"]
    axes[0].bar(range(6), values, color=colors)
    for index, value in enumerate(values):
        axes[0].text(index, value * 1.10, f"{value:.4f}", ha="center", fontsize=8)
    axes[0].set(
        xticks=range(6),
        xticklabels=labels,
        yscale="log",
        ylim=(4e-4, 0.023),
        ylabel="Raw complex relative L2",
        title="Original-source unseen frequencies",
    )
    axes[0].tick_params(axis="x", labelsize=8)
    colors_query = {
        "field_neural": "#2878B5",
        "field_neural_refine_20": "#6AA7D4",
        "gaussian_newton_1": "#D48124",
        "gaussian_newton_5": "#A85212",
        "true_medium_projection": "#333333",
    }
    names = {
        "field_neural": "Neural, 3 seeds",
        "field_neural_refine_20": "Neural + L-BFGS20",
        "gaussian_newton_1": "Prior + GN1*",
        "gaussian_newton_5": "Prior + GN5*",
        "true_medium_projection": "True-medium projection (oracle)",
    }
    query_rows = data["frozen_narrow"] + narrow_newton
    frequencies = [140, 200, 300, 400]
    for ax, metric, ylabel in [
        (axes[1], "raw", "Raw complex relative L2"),
        (axes[2], "phase_rmse_rad", "Energy-weighted phase RMSE [rad]"),
    ]:
        for method, color in colors_query.items():
            grouped = [
                [r[metric] for r in query_rows if r["method"] == method and r["frequency"] == f]
                for f in frequencies
            ]
            if any(not group for group in grouped):
                raise ValueError("incomplete frequency curve")
            mean = [np.mean(group) for group in grouped]
            ax.plot(
                frequencies,
                mean,
                "o--" if "projection" in method else "o-",
                color=color,
                label=names[method],
                ms=4,
            )
            if method.startswith("field_neural"):
                ax.fill_between(
                    frequencies,
                    [min(g) for g in grouped],
                    [max(g) for g in grouped],
                    color=color,
                    alpha=0.15,
                )
        ax.set(
            xlabel="Query frequency [Hz]",
            ylabel=ylabel,
            yscale="log",
            xticks=frequencies,
            title="New narrow source, new coordinates",
        )
    axes[1].legend(fontsize=7.5, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "64 independent waveguides | * GN is a post-freeze strengthened control", fontsize=12
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "scope": "Independent IID cohort, one synthetic one-way domain. Neural shading is seed range, not a CI. Statistical prior is data-learned. GN was added after original candidate freeze.",
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in paths.values()
                },
                "original_bars": dict(zip(labels, values, strict=True)),
                "narrow_newton": narrow_newton,
            },
            indent=2,
        )
    )
    print(json.dumps(narrow_newton, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
