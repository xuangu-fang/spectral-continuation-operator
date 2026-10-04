"""Separate prior regularization, decoder resolution and physical bandwidth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def run(root, output):
    paths = [
        root / "coherent_waveguide_profile5_initialization_20261003/summary.json",
        root / "coherent_waveguide_profile5_map_20261004/summary.json",
        root / "coherent_waveguide_continuous_generator_20261004/summary.json",
    ]
    cold, priors, resolution = [json.loads(path.read_text()) for path in paths]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    options = [
        (0, cold, "Update damping only", "#666666"),
        (1e-8, priors, "Full prior: 1e-8", "#2878B5"),
        (1e-6, priors, "Full prior: 1e-6", "#D48124"),
        (1e-4, priors, "Full prior: 1e-4", "#5A9657"),
    ]
    for strength, pool, label, color in options:
        selected = sorted(
            [
                r
                for r in pool
                if r["initialization"] == "gaussian"
                and (not strength or r["prior_strength"] == strength)
            ],
            key=lambda r: r["step"],
        )
        axes[0].plot(
            [r["step"] for r in selected],
            [r["raw"] for r in selected],
            "o-",
            color=color,
            label=label,
        )
    axes[0].set(
        xlabel="Fixed number of context-only GN steps",
        ylabel="Raw complex relative L2",
        yscale="log",
        title="5 profiles: a prior mismatch, not a proved limit",
    )
    axes[0].legend(fontsize=8)
    options = [
        ("continuous_prior", 16, 0, "Frozen code, finer decoder", "#2878B5", "o-"),
        ("continuous_prior", 16, 5, "Same context, 5 more GN steps", "#D48124", "o-"),
        (
            "true_medium_projection_oracle",
            16,
            -1,
            "True-medium oracle, 16 depth terms",
            "#888888",
            "s--",
        ),
        (
            "true_medium_projection_oracle",
            32,
            -1,
            "True-medium oracle, 32 depth terms",
            "#333333",
            "s--",
        ),
    ]
    for ax, variant, metric, title in [
        (axes[1], "original", "raw", "Original-source, unseen in-band frequencies"),
        (axes[2], "narrow", "narrow400_raw", "New narrow source, 400 Hz"),
    ]:
        for method, terms, step, label, color, style in options:
            selected = sorted(
                [
                    r
                    for r in resolution
                    if r["method"] == method
                    and r["depth_terms"] == terms
                    and r["step"] == step
                    and r["variant"] == variant
                ],
                key=lambda r: r["cells"],
            )
            if [r["cells"] for r in selected] != [64, 128, 256]:
                raise ValueError("complete predeclared resolution curve required")
            ax.plot(
                [r["cells"] for r in selected],
                [r[metric] for r in selected],
                style,
                color=color,
                label=label,
            )
        ax.set(
            xlabel="Range decoder cells",
            ylabel="Raw complex relative L2",
            xscale="log",
            yscale="log",
            xticks=[64, 128, 256],
            xticklabels=["64", "128", "256"],
            title=title,
        )
        ax.xaxis.set_minor_locator(NullLocator())
    axes[1].legend(fontsize=7.1, loc="lower left")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Original 8 development systems | Oracles have extra medium information", fontsize=12
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "scope": "Development mechanism diagnostics, not independent method validation. No query-driven stopping.",
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
                },
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
