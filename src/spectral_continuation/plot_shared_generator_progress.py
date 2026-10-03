"""Recorded error/cost/query figures, with no target phase alignment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit_waveguide_generator_identification import generator_geometry, latent_from_coefficients
from .coarse_waveguide_operator import query_field


def main():
    output = Path("reports/assets/coherent_continuous_20261003")
    cost = json.loads(
        Path(
            "artifacts/coherent_waveguide_generator_cost_cached_quadrature_20261003/summary.json"
        ).read_text()
    )
    transfer = Path("artifacts/coherent_waveguide_generator_query_transfer_20261003")
    rows = json.loads((transfer / "results.json").read_text())
    colors = {
        "field_neural": "#228833",
        "kernel_generator": "#4477aa",
        "regularized_inverse": "#cc6677",
        "nonlinear_context_100": "#222222",
    }
    labels = {
        "field_neural": "Learned generator",
        "kernel_generator": "Kernel generator",
        "regularized_inverse": "Regularized inverse",
        "nonlinear_context_100": "Context inverse, 100 steps",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    classic = sorted(
        [r for r in cost if r["method"].startswith("nonlinear_")],
        key=lambda r: int(r["method"].split("_")[-1]),
    )
    axes[0].plot(
        [r["total_seconds"] for r in classic],
        [r["raw"] for r in classic],
        "o-",
        color="#222222",
        label="Nonlinear context fit",
    )
    for row in classic:
        axes[0].annotate(
            row["method"].split("_")[-1] + " steps",
            (row["total_seconds"], row["raw"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    for row in cost:
        if row["method"] in colors:
            axes[0].scatter(
                row["total_seconds"],
                row["raw"],
                s=60,
                color=colors[row["method"]],
                label=labels[row["method"]],
                zorder=3,
            )
    axes[0].set(
        xscale="log",
        xlabel="Encoding + six dense queries (s)",
        ylabel="Raw complex relative L2",
        title="Accuracy–cost, not accuracy supremacy\n8 dev systems; shared-machine warm pilot",
    )
    axes[0].legend(fontsize=8)
    axes[0].set_xticks([0.2, 0.5, 1, 2, 3], ["0.2", "0.5", "1", "2", "3"])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax, variant, title in zip(
        axes[1:],
        ["original", "shifted"],
        ["Same source, new frequencies", "New source (+15 m), new frequencies"],
        strict=True,
    ):
        frequencies = sorted({r["frequency"] for r in rows})
        for method, color in colors.items():
            values = [
                np.mean(
                    [
                        r["complex_rel_l2"]
                        for r in rows
                        if r["method"] == method
                        and r["source_variant"] == variant
                        and r["frequency"] == f
                    ]
                )
                for f in frequencies
            ]
            ax.plot(frequencies, values, "o-", color=color, label=labels[method], ms=4)
        ax.axvline(140, color="0.6", ls="--", lw=1)
        ax.set(
            xlabel="Query frequency (Hz)",
            ylabel="Raw complex relative L2",
            title=title + "\nFrozen encoder; independent coordinates",
        )
        ax.text(142, ax.get_ylim()[1] * 0.92, "stress", fontsize=8, color="0.4")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Shared physical generator: learned fast inference helps, fully optimized classical inversion remains more accurate",
        fontsize=11,
    )
    fig.savefig(output / "shared_generator_query_transfer.png", dpi=180)
    plt.close(fig)

    # First system, fixed highest queried frequency and shifted source; not best-case selection.
    with np.load(transfer / "frozen_coefficients.npz") as z:
        beta = z["field_neural"][0]
        system = int(z["systems"][0])
    data = Path(
        "/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/coherent_waveguide_generator_query_transfer_20261003"
    )
    with np.load(data / f"system{system}_shifted.npz") as z:
        spec = json.loads(str(z["spec"]))
        target = z["field"][-1]
        ranges, depths = z["ranges_m"], z["depths_m"]
        frequency = float(z["frequencies_hz"][-1])
    geometry = generator_geometry(spec, 64, 16)
    initial = geometry["basis"].T @ (
        geometry["weights"]
        * np.exp(-0.5 * ((geometry["z"] - spec["source_depth_m"]) / spec["source_width_m"]) ** 2)
    )
    latent, _ = latent_from_coefficients(beta, initial, geometry)
    prediction = query_field(latent, frequency, ranges, depths)
    selected = (ranges >= 1750) & (ranges <= 1950)
    phase_error = np.angle(prediction * target.conj())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    extent = [ranges[selected][0], ranges[selected][-1], depths[-1], depths[0]]
    for ax, field, title in [
        (axes[0], target, "Direct solver: wrapped phase"),
        (axes[1], prediction, "Frozen learned generator: phase"),
    ]:
        ax.imshow(
            np.angle(field[:, selected]),
            extent=extent,
            aspect="auto",
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax.set(title=title, xlabel="Range (m)", ylabel="Depth (m)")
    im = axes[2].imshow(
        phase_error[:, selected], extent=extent, aspect="auto", cmap="coolwarm", vmin=-0.1, vmax=0.1
    )
    axes[2].set(title="Circular phase error (no alignment)", xlabel="Range (m)", ylabel="Depth (m)")
    fig.colorbar(im, ax=axes[2], label="rad; color clipped at ±0.1 only")
    fig.suptitle(
        f"First dev system {system}, new source, {frequency:g} Hz: 1024 real coefficients + known background/source",
        fontsize=11,
    )
    fig.savefig(output / "shared_generator_high_frequency_phase.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
