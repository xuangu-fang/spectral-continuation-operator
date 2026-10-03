"""Real-data development figures; no generated or interpolated reference images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .coherent_representation import complex_metrics


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rows = json.loads(
        (root / "coherent_helmholtz_fullfield_budget_20261003/results.json").read_text()
    )
    rows = [r for r in rows if r["region"] == "full_roi"]
    methods = {
        "cubic_spline": "Cubic interpolation",
        "gaussian_e1.25_ridge1e-10": "Tuned Gaussian kernel",
        "learned_plane_wave": "Learned plane-wave chart",
        "learned_curved_kernel": "Learned coherent kernel",
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, label in methods.items():
        x, y = [], []
        for stride in [6, 4, 2]:
            selected = [r for r in rows if r["method"] == method and r["stride"] == stride]
            x.append(np.mean([r["total_observed_points"] for r in selected]))
            y.append(np.mean([r["complex_rel_l2"] for r in selected]))
        ax.loglog(x, y, "o-", label=label)
    ax.set(
        xlabel="Observed complex samples per full field",
        ylabel="Raw complex relative L2",
        title="Heterogeneous Helmholtz | 8 development systems",
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "helmholtz_spatial_budget.png", dpi=180)
    plt.close(fig)

    path = (
        root / "coherent_helmholtz_fine_field_audit_20261003/system20261132_frequency14_stride6.npz"
    )
    with np.load(path) as z:
        data = dict(z)
    target, learned, baseline = (
        data["target"],
        data["learned_curved_kernel"],
        data["gaussian_e1.25_ridge1e-10"],
    )
    scale = np.sqrt(np.mean(abs(target) ** 2))
    extent = [float(data["x"][0]), float(data["x"][-1]), float(data["y"][0]), float(data["y"][-1])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    visible = abs(target) > 0.01 * scale
    views = [
        (abs(target) / scale, "Target amplitude / RMS", "viridis", 0, 3),
        (abs(baseline - target) / scale, "Gaussian: complex error / RMS", "magma", 0, 0.08),
        (abs(learned - target) / scale, "Learned: complex error / RMS", "magma", 0, 0.08),
        (np.ma.masked_where(~visible, np.angle(target)), "Target phase", "twilight", -np.pi, np.pi),
        (
            np.ma.masked_where(~visible, np.angle(baseline * target.conj())),
            "Gaussian: circular phase error (rad)",
            "coolwarm",
            -0.15,
            0.15,
        ),
        (
            np.ma.masked_where(~visible, np.angle(learned * target.conj())),
            "Learned: circular phase error (rad)",
            "coolwarm",
            -0.15,
            0.15,
        ),
    ]
    for ax, (array, title, cmap, vmin, vmax) in zip(axes.flat, views, strict=True):
        im = ax.imshow(
            array, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
        )
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax, shrink=0.78)
    mask = data["query_mask"]
    b = complex_metrics(baseline[mask], target[mask])["complex_rel_l2"]
    l = complex_metrics(learned[mask], target[mask])["complex_rel_l2"]
    fig.suptitle(
        f"Direct fine-grid teacher | system 20261132, f=14 | Gaussian {b:.4f}, learned {l:.4f}",
        fontsize=12,
    )
    fig.savefig(output / "helmholtz_fine_phase.png", dpi=180)
    plt.close(fig)

    spectral = json.loads(
        (root / "coherent_modal_spectral_hankel_20261003/results.json").read_text()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
    for ax, spacing in zip(axes, [5, 10], strict=True):
        systems = sorted({r["system"] for r in spectral})
        for m, marker, label in [
            ("range_carrier_cubic", "o", "Reference-carrier interpolation"),
            ("modal_k_cubic", "s", "Classical modal transport"),
        ]:
            values = [
                np.mean(
                    [
                        r["complex_rel_l2"]
                        for r in spectral
                        if r["system"] == s
                        and r["frequency_spacing"] == spacing
                        and r["method"] == m
                    ]
                )
                for s in systems
            ]
            ax.plot(range(len(systems)), values, marker + "-", label=label)
        ax.set(
            xticks=range(len(systems)),
            xticklabels=systems,
            xlabel="Development system",
            title=f"Observed frequency spacing: {spacing} Hz",
            yscale="log",
        )
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Unobserved-frequency raw complex error")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "modal_spectral_query.png", dpi=180)
    plt.close(fig)
    (output / "README.md").write_text(
        "# Development figures\n\nGenerated from recorded solver/experiment arrays with `plot_coherent_progress.py`. No synthetic illustration.\n\n- Budget: 8 development systems, one learned seed; formal testing pending.\n- Phase map: independently solved 513-grid truth, 257-grid observations; color clipping affects display only. Phase pixels below 1% target RMS are masked for legibility only; raw metrics retain them.\n- Spectral: classical DMD, not a learned method; all four development environments retained.\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument(
        "--output", type=Path, default=Path("reports/assets/coherent_continuous_20261003")
    )
    args = p.parse_args()
    run(args.artifacts, args.output)


if __name__ == "__main__":
    main()
