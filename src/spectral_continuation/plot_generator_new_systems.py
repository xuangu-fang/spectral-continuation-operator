"""System-paired new-cohort error panels and a fixed-system phase illustration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from .audit_waveguide_frequency_scaling import propagate_code
from .coherent_waveguide_data import depth_basis


def paired_summary(rows, systems, frequencies, sources):
    lookup = {(row["system"], row["frequency"], row["source"], row["method"]): row for row in rows}
    rng = np.random.default_rng(20261004)
    records = []
    for frequency in frequencies:
        for source in sources:
            old = np.array(
                [lookup[(s, frequency, source, "prior16")]["complex_rel_l2"] for s in systems]
            )
            new = np.array(
                [lookup[(s, frequency, source, "prior32")]["complex_rel_l2"] for s in systems]
            )
            oracle = np.array(
                [
                    lookup[(s, frequency, source, "true_medium32_oracle")]["complex_rel_l2"]
                    for s in systems
                ]
            )
            sample = rng.integers(len(systems), size=(10000, len(systems)))
            improvement = 1 - new[sample].mean(axis=1) / old[sample].mean(axis=1)
            records.append(
                {
                    "frequency": frequency,
                    "source": source,
                    "systems": len(systems),
                    "prior16_raw": float(old.mean()),
                    "prior32_raw": float(new.mean()),
                    "true_medium32_oracle_raw": float(oracle.mean()),
                    "relative_improvement": float(1 - new.mean() / old.mean()),
                    "paired_bootstrap_95": np.quantile(improvement, [0.025, 0.975]).tolist(),
                    "win_fraction": float(np.mean(new < old)),
                    "max_prior32_system_raw": float(new.max()),
                }
            )
    return records


def plot(config, results, output):
    rows = json.loads(results.read_text())
    if len({r["system"] for r in rows}) != len(config["systems"]):
        raise ValueError("full fixed cohort required before plotting")
    summary = paired_summary(
        rows, config["systems"], config["query_frequencies"], config["source_variants"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(results),
                "selection": "all frozen32 independent IID systems",
                "bootstrap_unit": "physical system, same index for both methods",
                "bootstrap_repetitions": 10000,
                "bootstrap_seed": 20261004,
                "records": summary,
            },
            indent=2,
        )
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharey=True)
    colors = {"prior16": "#b9624a", "prior32": "#287d8e"}
    methods = ["prior16", "prior32"]
    lookup = {(r["system"], r["frequency"], r["source"], r["method"]): r for r in rows}
    for fi, frequency in enumerate(config["query_frequencies"]):
        for si, source in enumerate(config["source_variants"]):
            ax = axes[fi, si]
            for j, system in enumerate(config["systems"]):
                values = [
                    lookup[(system, frequency, source, method)]["complex_rel_l2"]
                    for method in methods
                ]
                ax.plot([0, 1], values, color="gray", alpha=0.23, linewidth=0.7)
                for mi, method in enumerate(methods):
                    ax.scatter(
                        mi + (j % 5 - 2) * 0.012,
                        values[mi],
                        s=12,
                        color=colors[method],
                        alpha=0.7,
                        zorder=3,
                    )
            record = next(
                r for r in summary if r["frequency"] == frequency and r["source"] == source
            )
            ax.scatter(
                [0, 1],
                [record["prior16_raw"], record["prior32_raw"]],
                marker="D",
                s=53,
                color=[colors[m] for m in methods],
                edgecolor="black",
                zorder=4,
            )
            ax.axhline(
                record["true_medium32_oracle_raw"],
                linestyle="--",
                color="#5c5c5c",
                alpha=0.8,
                linewidth=1,
            )
            lo, hi = record["paired_bootstrap_95"]
            ax.set_title(
                f"{frequency} Hz; {'broad' if source == 'original' else 'new 2 m'} source\n"
                f"mean improvement {100 * record['relative_improvement']:.1f}% "
                f"[{100 * lo:.1f}, {100 * hi:.1f}]%"
            )
            ax.set_xticks([0, 1], ["16 terms", "32 terms"])
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.2)
            ax.yaxis.set_minor_locator(NullLocator())
            if si == 0:
                ax.set_ylabel("Raw complex relative L2")
    fig.suptitle(
        "32 new waveguide systems; identical 128-dimensional code budget\n"
        "Each line is one matched environment; diamonds mark system means",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.89))
    for ext in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{ext}"), dpi=190, bbox_inches="tight")
    plt.close(fig)

    # Illustration uses first predeclared new system; no performance-based selection.
    system = config["systems"][0]
    f = max(config["query_frequencies"])
    with np.load(Path(config["output_data"]) / f"system{system}_f{f}.npz") as data:
        spec = json.loads(str(data["spec"]))
        ranges, depths = data["ranges_m"], data["depths_m"]
        take_r = (ranges >= 1500) & (ranges <= 1600)
        take_z = (depths >= 20) & (depths <= 80)
        truth_coeff = data["modal_coefficients"][take_r, :, 1]
        truth = depth_basis(depths[take_z], spec["depth_m"], truth_coeff.shape[1]) @ truth_coeff.T
    with np.load(results.parent / f"system{system}_codes.npz") as data:
        predictions = [("16 terms", data["beta16"]), ("32 terms", data["beta32"])]
    fig, axes = plt.subplots(3, 1, figsize=(7.7, 7.3), sharex=True, sharey=True)
    extent = [ranges[take_r][0], ranges[take_r][-1], depths[take_z][-1], depths[take_z][0]]
    amplitude = np.max(np.abs(truth))
    image_real = axes[0].imshow(
        truth.real / amplitude,
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    mask = np.abs(truth) < 0.001 * amplitude
    images, hidden = [], []
    for ax, (label, beta) in zip(axes[1:], predictions, strict=True):
        prediction = propagate_code(
            spec,
            f,
            ranges[take_r],
            config["decoder_cells"],
            config["decoder_modes"],
            config["teacher_quadrature_nodes"],
            config["eigh_device"],
            config["eigh_batch"],
            beta,
        )
        predicted = (
            depth_basis(depths[take_z], spec["depth_m"], prediction.shape[1])
            @ prediction[:, :, 1].T
        )
        phase = np.angle(predicted * truth.conj())
        phase[mask] = np.nan
        images.append(
            ax.imshow(
                phase,
                extent=extent,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-0.05,
                vmax=0.05,
                interpolation="nearest",
            )
        )
        hidden.append({"method": label, "low_amplitude_display_fraction": float(mask.mean())})
    for ax, label in zip(axes, ["Truth Re(u)", "16-term error", "32-term error"], strict=True):
        ax.set_ylabel(label + "\nDepth (m)")
    axes[-1].set_xlabel("Range (m); fixed 1500–1600 m illustration window")
    fig.suptitle(
        f"Fixed new system {system}, 800 Hz, new 2 m source\n"
        "Pointwise phase difference without alignment; common scale ±0.05 rad",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.15, right=0.83, hspace=0.12)
    fig.colorbar(image_real, cax=fig.add_axes([0.86, 0.68, 0.02, 0.15]), label="Re(u) / max |u|")
    fig.colorbar(
        images[-1],
        cax=fig.add_axes([0.86, 0.18, 0.02, 0.36]),
        label="Phase difference (rad)",
        extend="both",
    )
    phase_path = output.with_name(output.name + "_phase")
    for ext in ["png", "pdf"]:
        fig.savefig(phase_path.with_suffix(f".{ext}"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    provenance = json.loads(output.with_suffix(".json").read_text())
    provenance["illustration"] = {
        "system": system,
        "frequency": f,
        "selection": "first frozen system",
        "display_only_low_amplitude_mask": hidden,
        "primary_metric": "full >=50 m ROI, no amplitude mask or phase alignment",
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.config.read_text()), args.results, args.output)


if __name__ == "__main__":
    main()
