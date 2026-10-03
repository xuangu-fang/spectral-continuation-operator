"""Show classical representation gains separately from actual learned gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .coherent_representation import complex_metrics


def average(rows, **filters):
    selected = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    if not selected:
        raise ValueError(filters)
    return float(np.mean([r["complex_rel_l2"] for r in selected]))


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads((root / "coherent_known_medium_phase_rom_20261003/results.json").read_text())
    base = [r for r in base if len(r["observed_frequencies"]) == 3]
    enriched = json.loads((root / "coherent_phase_rom_envelopes_20261003/results.json").read_text())
    source = json.loads(
        (root / "coherent_fundamental_solution_rom_20261003/results.json").read_text()
    )
    neural = json.loads(
        (root / "coherent_learned_envelope_neural_20261003/validation.json").read_text()
    )
    scalar = json.loads(
        (root / "coherent_learned_envelope_scalar_20261003/validation.json").read_text()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    labels = [
        "Raw\n6 columns",
        "Travel phase\n6 columns",
        "Source-aware\n6 columns",
        "Radial envelope\n18 columns",
    ]
    values = [
        average(base, method="raw_least_residual"),
        average(base, method="straight_ray_medium_phase_least_residual"),
        average(source, reference="gaussian_green_medium_phase", readout="pde"),
        average(enriched, basis="radial_envelope", readout="pde", region="full_roi"),
    ]
    axes[0].bar(np.arange(4), values, color=["#999999", "#4477aa", "#228833", "#66ccee"])
    axes[0].set(
        yscale="log",
        ylim=(0.01, 1.5),
        xticks=np.arange(4),
        xticklabels=labels,
        ylabel="Raw complex relative L2",
        title="Classical representation controls\n4 development environments",
    )
    for x, y in enumerate(values):
        axes[0].text(x, y * 1.12, f"{y:.4f}", ha="center", fontsize=9)
    labels = [
        "Fixed radial\n18 columns",
        "2 global powers\n18 columns",
        "Learned envelope\n18 columns",
    ]
    selectors = [
        (neural, "straight_ray_medium_phase_radial"),
        (scalar, "learned_envelope_scalar"),
        (neural, "learned_envelope_neural"),
    ]
    values = [average(rows, method=name) for rows, name in selectors]
    axes[1].bar(np.arange(3), values, color=["#66ccee", "#aa4499", "#ee7733"], alpha=0.8)
    for x, (rows, name) in enumerate(selectors):
        systems = sorted({r["system"] for r in rows})
        per_system = [average(rows, method=name, system=system) for system in systems]
        axes[1].scatter(
            np.linspace(x - 0.13, x + 0.13, len(systems)),
            per_system,
            s=18,
            color="black",
            alpha=0.6,
            zorder=3,
        )
        axes[1].text(x, max(per_system) + 0.001, f"mean {values[x]:.5f}", ha="center", fontsize=9)
    axes[1].set(
        ylim=(0, 0.038),
        xticks=np.arange(3),
        xticklabels=labels,
        ylabel="Raw complex relative L2",
        title="Learning increment at matched rank\n8 development environments; dots = systems",
    )
    axes[1].text(
        0.5,
        0.97,
        f"Neural improvement: {(1 - values[-1] / values[0]) * 100:.1f}% (gate: 20%)",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Known medium/source/PDE | 3 context frequencies | zero target-frequency observations",
        fontsize=11,
    )
    fig.savefig(output / "classical_vs_learning_rom.png", dpi=180)
    plt.close(fig)

    diagnosis = root / "coherent_rom_grid_sensitivity_20261003/system20261132_diagnosis.npz"
    if diagnosis.exists():
        with np.load(diagnosis) as z:
            data = dict(z)
        target = data["target"]
        failed = data["phase_fine_pde"]
        corrected = data["phase_regularized_fine_pde"]
        roi = data["roi"]
        axis = data["axis"]
        selected = np.flatnonzero(roi.any(axis=0))
        crop = np.ix_(selected, selected)
        extent = [axis[selected[0]], axis[selected[-1]], axis[selected[0]], axis[selected[-1]]]
        rms = np.sqrt(np.mean(abs(target[roi]) ** 2))
        visible = abs(target) >= 0.01 * rms
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
        panels = [
            (np.angle(target), "Target phase", "twilight", -np.pi, np.pi),
            (np.angle(failed), "Unsmooth carrier: predicted phase", "twilight", -np.pi, np.pi),
            (np.angle(corrected), "Source-regularized: predicted phase", "twilight", -np.pi, np.pi),
            (abs(target) / rms, "Target amplitude / ROI RMS", "viridis", 0, 3),
            (
                np.ma.masked_where(~visible, np.angle(failed * target.conj())),
                "Unsmooth: circular phase error",
                "coolwarm",
                -0.2,
                0.2,
            ),
            (
                np.ma.masked_where(~visible, np.angle(corrected * target.conj())),
                "Regularized: circular phase error",
                "coolwarm",
                -0.2,
                0.2,
            ),
        ]
        for ax, (values, title, cmap, lo, hi) in zip(axes.flat, panels, strict=True):
            im = ax.imshow(values[crop], origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
            ax.set(title=title, xlabel="x", ylabel="y")
            fig.colorbar(im, ax=ax, shrink=0.8)
        fail_error = complex_metrics(failed[roi], target[roi])["complex_rel_l2"]
        correct_error = complex_metrics(corrected[roi], target[roi])["complex_rel_l2"]
        fig.suptitle(
            f"Independent 513-grid, f={float(data['frequency']):g} | same 6 columns | raw error {fail_error:.4f} -> {correct_error:.4f}\nClassical source-regularity correction, NOT a learned-method gain",
            fontsize=11,
        )
        fig.savefig(output / "source_regularity_phase.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/assets/coherent_continuous_20261003")
    )
    args = parser.parse_args()
    run(args.artifacts, args.output)


if __name__ == "__main__":
    main()
