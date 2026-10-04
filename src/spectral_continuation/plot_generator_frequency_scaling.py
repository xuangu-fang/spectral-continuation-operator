"""Aggregate all systems and a fixed first-system phase illustration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from .coherent_waveguide_data import depth_basis


def run(directory, output):
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    frequency = config["frequencies"]
    methods = [
        ("frozen_code_cells256_modes32", "Code: 32 propagation modes", "#b55354", "-"),
        ("frozen_code_cells256_modes64", "Code: 64 propagation modes", "#d29829", "-"),
        ("frozen_code_cells256_modes128", "Code: 128 propagation modes", "#287d8e", "-"),
        ("frozen_code_cells512_modes128", "Same code: finer range integration", "#564e93", "-"),
        ("true_medium32_oracle", "True 32-term medium (oracle)", "#737373", "--"),
        ("full_medium_oracle", "Full known medium (oracle)", "#333333", ":"),
    ]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.0), sharex=True)
    for col, source in enumerate(config["source_variants"]):
        for method, label, color, style in methods:
            values = [
                next(
                    row
                    for row in summary
                    if row["method"] == method
                    and row["frequency"] == f
                    and row["source"] == source
                    and row["range_limit_m"] == 2000
                )
                for f in frequency
            ]
            for ax, metric in [
                (axes[0, col], "complex_rel_l2"),
                (axes[1, col], "energy_weighted_phase_rmse_rad"),
            ]:
                ax.semilogy(
                    frequency,
                    [row[metric] for row in values],
                    marker="o",
                    markersize=3,
                    color=color,
                    linestyle=style,
                    label=label,
                )
                ax.grid(alpha=0.2)
                ax.yaxis.set_minor_locator(NullLocator())
                ax.axvline(140, color="gray", alpha=0.4, linewidth=1)
        axes[0, col].set_title("Original broad source" if col == 0 else "New 2 m narrow source")
        axes[1, col].set_xlabel("Query frequency (Hz); observed band ends at 140 Hz")
    axes[0, 0].set_ylabel("Raw complex relative L2")
    axes[1, 0].set_ylabel("Energy-weighted phase RMSE (rad)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=9,
    )
    fig.suptitle(
        "Frozen 128-dimensional environment code; 8 development systems\n"
        "No new-frequency measurements or code refitting",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Predeclared first old-development system, not a best-case selection.
    system = config["systems"][0]
    illustration_frequencies = [140, 400, 800]
    shown = ["frozen_code_cells256_modes32", "frozen_code_cells256_modes128", "full_medium_oracle"]
    fig, axes = plt.subplots(4, 3, figsize=(11.0, 9.0), sharex=True, sharey=True)
    hidden = []
    for col, f in enumerate(illustration_frequencies):
        with np.load(Path(config["data_output"]) / f"system{system}_f{f}.npz") as data:
            spec = json.loads(str(data["spec"]))
            ranges, depths = data["ranges_m"], data["depths_m"]
            mask_r = (ranges >= 1500) & (ranges <= 1600)
            mask_z = (depths >= 20) & (depths <= 80)
            coeff = data["modal_coefficients"][mask_r, :, 1]
            truth = depth_basis(depths[mask_z], spec["depth_m"], coeff.shape[1]) @ coeff.T
        extent = [ranges[mask_r][0], ranges[mask_r][-1], depths[mask_z][-1], depths[mask_z][0]]
        amplitude = np.max(np.abs(truth))
        im_real = axes[0, col].imshow(
            truth.real / amplitude,
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        axes[0, col].set_title(f"{f} Hz")
        with np.load(directory / f"system{system}_f{f}_predictions.npz") as data:
            for row, method in enumerate(shown, 1):
                coeff = data[method][mask_r, :, 1]
                predicted = depth_basis(depths[mask_z], spec["depth_m"], coeff.shape[1]) @ coeff.T
                phase = np.angle(predicted * truth.conj())
                tiny = np.abs(truth) < 1e-3 * amplitude
                hidden.append(
                    {
                        "frequency": f,
                        "method": method,
                        "low_amplitude_display_fraction": float(tiny.mean()),
                    }
                )
                phase[tiny] = np.nan
                im_phase = axes[row, col].imshow(
                    phase,
                    extent=extent,
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-0.05,
                    vmax=0.05,
                    interpolation="nearest",
                )
    labels = ["Truth: Re(u)", "32-mode error", "128-mode error", "Oracle error"]
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label + "\nDepth (m)")
    for ax in axes[-1]:
        ax.set_xlabel("Range (m), fixed illustration crop")
    fig.suptitle(
        f"Fixed system {system}, 2 m query source; absolute phase, no alignment\n"
        "Common phase-error scale ±0.05 rad; metrics use the entire ≥50 m ROI",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.89, bottom=0.07, left=0.11, right=0.84, hspace=0.16, wspace=0.10)
    fig.colorbar(im_real, cax=fig.add_axes([0.87, 0.72, 0.018, 0.14]), label="Re(u) / max |u|")
    fig.colorbar(
        im_phase,
        cax=fig.add_axes([0.87, 0.21, 0.018, 0.38]),
        label="Phase difference (rad)",
        extend="both",
    )
    phase_output = output.with_name(output.name + "_phase")
    for ext in ["png", "pdf"]:
        fig.savefig(phase_output.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(directory),
                "systems": config["systems"],
                "system_illustration": system,
                "illustration_selection": "first system in predeclared development list, not best-performing",
                "phase_display": "Only illustration hides |truth| < 0.001 crop maximum; metrics unmasked",
                "display_details": hidden,
                "scope": "development only; numerical/representation audit, not neural performance claim",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
