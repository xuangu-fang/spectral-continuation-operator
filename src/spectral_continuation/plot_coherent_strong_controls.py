"""Superseding strong controls and classical continuous-codec evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("artifacts")
    output = Path("reports/assets/coherent_continuous_20261003")
    krylov = json.loads((root / "coherent_background_krylov_20261003/results.json").read_text())
    codec = json.loads((root / "coherent_waveguide_curve_codec_20261003/results.json").read_text())
    operator = json.loads(
        (root / "coherent_waveguide_coarse_operator_20261003/results.json").read_text()
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    for label, text, color in [
        ("zero", "GMRES, zero initial", "#4477aa"),
        ("source_aware_field_rom", "GMRES, field-ROM initial", "#228833"),
    ]:
        budgets = [0, 1, 2, 4, 6, 8]
        error = [
            np.mean(
                [
                    r["complex_rel_l2"]
                    for r in krylov
                    if r["initialization"] == label and r["background_applications"] == n
                ]
            )
            for n in budgets
        ]
        axes[0].plot(budgets, error, "o-", label=text, color=color)
    axes[0].scatter(
        [7],
        [0.01352303543238899],
        marker="s",
        color="#cc6677",
        label="6 source columns + background",
    )
    axes[0].set(
        title="Known PDE: strong classical solve\n4 dev systems; same queries",
        xlabel="Background inverse applications",
        ylabel="Raw complex relative L2",
        yscale="log",
    )
    axes[0].legend(fontsize=8)
    for method, text, color in [
        ("raw_modes", "Raw modal curves", "#cc6677"),
        ("background_phase", "Background phase + envelope", "#4477aa"),
        ("diagonal_phase", "Full-medium diagonal phase", "#228833"),
    ]:
        counts = [17, 33, 65, 129, 257, 513]
        error = [
            np.mean(
                [
                    r["complex_rel_l2"]
                    for r in codec
                    if r["method"] == method and r["range_samples"] == n
                ]
            )
            for n in counts
        ]
        axes[1].plot(counts, error, "o-", label=text, color=color)
    axes[1].set(
        title="R0 codec: same-frequency observations\n8 dev systems; held range coordinates",
        xlabel="Observed depth profiles",
        ylabel="Raw complex relative L2",
        xscale="log",
        yscale="log",
    )
    axes[1].legend(fontsize=8)
    axes[1].set_xticks(counts, [str(n) for n in counts])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    cells = [8, 16, 32, 64, 128]
    error = [np.mean([r["complex_rel_l2"] for r in operator if r["cells"] == n]) for n in cells]
    axes[2].plot(cells, error, "o-", color="#4477aa", label="Full coherent mode mixing")
    axes[2].axhline(
        0.2766188725429337, color="#cc6677", linestyle="--", label="Adiabatic, no mode coupling"
    )
    axes[2].set(
        title="Known medium: sparse operator state\n8 dev systems; 2049 output ranges",
        xlabel="Range cells in operator latent",
        ylabel="Raw complex relative L2",
        xscale="log",
        yscale="log",
    )
    axes[2].legend(fontsize=8)
    axes[2].set_xticks(cells, [str(n) for n in cells])
    axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "All three are classical controls, not neural gains; information protocols differ across panels",
        fontsize=11,
    )
    fig.savefig(output / "strong_controls_and_codec.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
