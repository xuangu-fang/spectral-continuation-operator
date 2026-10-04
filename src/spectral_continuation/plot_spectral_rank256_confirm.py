"""Matched-system confirmation figure and fixed-system phase detail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit_waveguide_frequency_scaling import propagate_code
from .coherent_waveguide_data import depth_basis


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(base_rows, new_rows, systems, frequencies, sources):
    base = {(r["system"], r["frequency"], r["source"]): r for r in base_rows
            if r["method"] == "prior32"}
    new = {(r["system"], r["frequency"], r["source"]): r for r in new_rows
           if r["rank"] == 256}
    direct128 = {(r["system"], r["frequency"], r["source"]): r for r in new_rows
                 if r["rank"] == 128}
    rng = np.random.default_rng(20261004)
    records = []
    for frequency in frequencies:
        for source in sources:
            keys = [(system, frequency, source) for system in systems]
            old_raw = np.array([base[key]["complex_rel_l2"] for key in keys])
            direct_raw = np.array([direct128[key]["complex_rel_l2"] for key in keys])
            new_raw = np.array([new[key]["complex_rel_l2"] for key in keys])
            old_phase = np.array([base[key]["energy_weighted_phase_rmse_rad"]
                                  for key in keys])
            new_phase = np.array([new[key]["energy_weighted_phase_rmse_rad"]
                                  for key in keys])
            new_amplitude = np.array([new[key]["amplitude_rel_l2"] for key in keys])
            sample = rng.integers(len(systems), size=(10000, len(systems)))
            improvement = 1 - new_raw[sample].mean(axis=1) / old_raw[sample].mean(axis=1)
            records.append({
                "frequency": frequency, "source": source, "systems": len(systems),
                "original_rank128_raw": float(old_raw.mean()),
                "direct_rank128_raw": float(direct_raw.mean()),
                "rank256_raw": float(new_raw.mean()),
                "rank256_amplitude_rel_l2": float(new_amplitude.mean()),
                "original_rank128_phase_rmse_rad": float(old_phase.mean()),
                "rank256_phase_rmse_rad": float(new_phase.mean()),
                "relative_improvement": float(1 - new_raw.mean() / old_raw.mean()),
                "paired_bootstrap_95": np.quantile(improvement, [0.025, 0.975]).tolist(),
                "system_wins": int(np.sum(new_raw < old_raw)),
                "max_rank256_system_raw": float(new_raw.max()),
                "max_rank256_system_phase_rmse_rad": float(new_phase.max()),
            })
    return records


def phase_illustration(base, new, source, output):
    system = source["systems"][0]
    frequency = max(source["query_frequencies"])
    with np.load(Path(source["output_data"]) / f"system{system}_f{frequency}.npz") as data:
        spec = json.loads(str(data["spec"]))
        ranges, depths = data["ranges_m"], data["depths_m"]
        take_r = (ranges >= 1500) & (ranges <= 1600)
        take_z = (depths >= 20) & (depths <= 80)
        truth_coeff = data["modal_coefficients"][take_r, :, 1]
        truth = depth_basis(depths[take_z], spec["depth_m"], truth_coeff.shape[1]) @ truth_coeff.T
    with np.load(base / f"system{system}_codes.npz") as data:
        beta128 = data["beta32"]
    with np.load(new / f"system{system}_rank256_code.npz") as data:
        beta256 = data["beta"]
    extent = [ranges[take_r][0], ranges[take_r][-1],
              depths[take_z][-1], depths[take_z][0]]
    amplitude = np.max(np.abs(truth))
    mask = np.abs(truth) < amplitude * 0.001
    fig, axes = plt.subplots(3, 1, figsize=(7.7, 7.5), sharex=True, sharey=True)
    real = axes[0].imshow(truth.real / amplitude, extent=extent, aspect="auto",
                          cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    predictions = []
    for beta in [beta128, beta256]:
        modal = propagate_code(
            spec, frequency, ranges[take_r], source["decoder_cells"],
            source["decoder_modes"], source["teacher_quadrature_nodes"],
            source["eigh_device"], source["eigh_batch"], beta,
        )
        predictions.append(depth_basis(depths[take_z], spec["depth_m"], modal.shape[1])
                           @ modal[:, :, 1].T)
    phase_images = []
    for ax, prediction, scale in zip(axes[1:], predictions, [0.75, 0.05], strict=True):
        phase = np.angle(prediction * truth.conj())
        phase[mask] = np.nan
        phase_images.append(ax.imshow(phase, extent=extent, aspect="auto", cmap="RdBu_r",
                                      vmin=-scale, vmax=scale, interpolation="nearest"))
    for ax, label in zip(axes, ["Truth Re(u)", "Original 128D phase error",
                                "Context-fit 256D phase error"], strict=True):
        ax.set_ylabel(label + "\nDepth (m)")
    axes[-1].set_xlabel("Range (m); fixed 1500–1600 m illustration window")
    fig.suptitle(f"Independent fixed system {system}; 800 Hz; unseen new 2 m source\n"
                 "Absolute phase, no alignment; display scales differ and are labelled")
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.16, right=0.79, hspace=0.12)
    fig.colorbar(real, cax=fig.add_axes([0.82, 0.71, 0.02, 0.11]),
                 label="Re(u) / max |u|")
    fig.colorbar(phase_images[0], cax=fig.add_axes([0.82, 0.42, 0.02, 0.17]),
                 label="128D error (rad)", extend="both")
    fig.colorbar(phase_images[1], cax=fig.add_axes([0.82, 0.13, 0.02, 0.17]),
                 label="256D error (rad)", extend="both")
    phase_output = output.with_name(output.name + "_phase")
    for ext in ["png", "pdf"]:
        fig.savefig(phase_output.with_suffix(f".{ext}"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {"system": system, "frequency": frequency, "source": "narrow2m",
            "selection": "first predeclared independent system", "display_mask_fraction": float(mask.mean()),
            "phase_display_limits_rad": [0.75, 0.05],
            "primary_metric": "full >=50 m ROI, no mask or phase alignment"}


def plot(base_path, new_path, output):
    source = json.loads((base_path / "config.json").read_text())
    base_rows = json.loads((base_path / "results.json").read_text())
    new_rows = json.loads((new_path / "results.json").read_text())
    if len(base_rows) != 20 * len(source["systems"]) or len(new_rows) != 8 * len(source["systems"]):
        raise ValueError("complete independent cohort required")
    records = summarize(base_rows, new_rows, source["systems"],
                        source["query_frequencies"], source["source_variants"])
    output.parent.mkdir(parents=True, exist_ok=True)
    lookup_base = {(r["system"], r["frequency"], r["source"]): r for r in base_rows
                   if r["method"] == "prior32"}
    lookup_new = {(r["system"], r["frequency"], r["source"]): r for r in new_rows
                  if r["rank"] == 256}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharey=True)
    for fi, frequency in enumerate(source["query_frequencies"]):
        for si, variant in enumerate(source["source_variants"]):
            ax = axes[fi, si]
            for j, system in enumerate(source["systems"]):
                key = (system, frequency, variant)
                values = [lookup_base[key]["complex_rel_l2"],
                          lookup_new[key]["complex_rel_l2"]]
                ax.plot([0, 1], values, color="#777777", alpha=0.22, lw=0.75)
                ax.scatter([0 + (j % 5 - 2) * 0.01, 1 + (j % 5 - 2) * 0.01], values,
                           s=11, color=["#b9624a", "#287d8e"], alpha=0.7, zorder=3)
            item = next(r for r in records if r["frequency"] == frequency
                        and r["source"] == variant)
            ax.scatter([0, 1], [item["original_rank128_raw"], item["rank256_raw"]],
                       marker="D", s=55, color=["#b9624a", "#287d8e"],
                       edgecolor="black", zorder=5)
            lo, hi = item["paired_bootstrap_95"]
            ax.set_title(f"{frequency} Hz; {'broad' if variant == 'original' else 'new 2 m'} source\n"
                         f"mean raw reduction {item['relative_improvement'] * 100:.1f}% "
                         f"[{lo * 100:.1f}, {hi * 100:.1f}]%")
            ax.set_xticks([0, 1], ["frozen 128D", "context-fit 256D"])
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.2)
            if si == 0:
                ax.set_ylabel("Raw complex relative L2")
    fig.suptitle("32 newly seeded correlated-spectral systems; method fixed before new truth\n"
                 "Same 60/100/140 Hz context; each line is one matched system")
    fig.tight_layout(rect=(0, 0.02, 1, 0.89))
    for ext in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{ext}"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    illustration = phase_illustration(base_path, new_path, source, output)
    output.with_suffix(".json").write_text(json.dumps({
        "base_results_sha256": digest(base_path / "results.json"),
        "rank256_results_sha256": digest(new_path / "results.json"),
        "paired_bootstrap_unit": "physical system", "paired_bootstrap_repetitions": 10000,
        "paired_bootstrap_seed": 20261004, "records": records,
        "illustration": illustration,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rank256", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.base, args.rank256, args.output)


if __name__ == "__main__":
    main()
