"""Preselected OOD-system phase maps with consistent circular scales and masking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_generator_identification import generator_geometry
from .audit_waveguide_spectral_exact_basis import analytic_prior
from .audit_waveguide_spectral_exact_nonlinear import exact_family_projection
from .coherent_waveguide_data import depth_basis


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(modal: np.ndarray, depths: np.ndarray, depth_m: float, source: int) -> np.ndarray:
    basis = depth_basis(depths, depth_m, modal.shape[1])
    return basis @ modal[:, :, source].T


def run(config: dict, output: Path) -> None:
    if (config["system"], config["frequency"], config["source_index"],
        config["range_min_m"], config["range_max_m"]) != (20267000, 800, 1, 1500, 1600):
        raise ValueError("preselected representative visualization must remain fixed")
    source = json.loads(Path(config["source_config"]).read_text())
    exact_config = json.loads(Path(config["exact_config"]).read_text())
    with np.load(config["truth_npz"]) as data:
        truth_modal = data["modal_coefficients"]
        ranges, depths = data["ranges_m"], data["depths_m"]
        spec = json.loads(str(data["spec"]))
    with np.load(config["frozen_code_npz"]) as data:
        frozen_beta = data["beta"]
    with np.load(config["exact_coordinates_npz"]) as data:
        coordinates = data["coordinates"]
    _, _, speed_scale = analytic_prior(spec, exact_config)
    geometry = generator_geometry(
        spec, source["decoder_cells"], exact_config["depth_terms"], modes=32
    )
    exact_beta, _ = exact_family_projection(
        spec, geometry, coordinates, speed_scale,
        taper_scale=exact_config["taper_scale"],
    )
    predictions = {}
    for name, beta in [("fixed", frozen_beta), ("exact", exact_beta)]:
        predictions[name] = propagate_code(
            spec, config["frequency"], ranges, source["decoder_cells"],
            source["decoder_modes"], source["teacher_quadrature_nodes"],
            source["eigh_device"], source["eigh_batch"], beta,
        )
    take_r = (ranges >= config["range_min_m"]) & (ranges <= config["range_max_m"])
    take_z = (depths >= config["depth_min_m"]) & (depths <= config["depth_max_m"])
    x, z = ranges[take_r], depths[take_z]
    target = field(truth_modal[take_r], z, spec["depth_m"], config["source_index"])
    fixed = field(predictions["fixed"][take_r], z, spec["depth_m"],
                  config["source_index"])
    exact = field(predictions["exact"][take_r], z, spec["depth_m"],
                  config["source_index"])
    amplitude = np.abs(target)
    cutoff = float(np.quantile(amplitude, config["amplitude_mask_quantile"]))
    mask = amplitude < cutoff
    phase = [np.ma.array(np.angle(value), mask=mask) for value in (target, fixed, exact)]
    differences = [
        np.ma.array(np.angle(value * np.conj(target)), mask=mask)
        for value in (fixed, exact)
    ]
    phase_cmap = plt.get_cmap("twilight_shifted").copy()
    phase_cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), sharex=True, sharey=True)
    extent = (float(x.min()), float(x.max()), float(z.max()), float(z.min()))
    top_titles = ["Numerical truth phase", "Frozen PCA256 phase", "Exact-family inverse phase"]
    for axis, values, title in zip(axes[0], phase, top_titles, strict=True):
        image_phase = axis.imshow(values, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi,
                                  extent=extent, aspect="auto", interpolation="nearest")
        axis.set_title(title)
    db = 20 * np.log10(np.maximum(amplitude / np.max(amplitude), 1e-4))
    image_amplitude = axes[1, 0].imshow(
        db, cmap="viridis", vmin=-40, vmax=0, extent=extent,
        aspect="auto", interpolation="nearest",
    )
    axes[1, 0].set_title("Truth amplitude (dB relative to local max)")
    for axis, values, title in zip(
        axes[1, 1:], differences,
        ["Frozen minus truth phase", "Exact-family minus truth phase"], strict=True,
    ):
        axis.imshow(values, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi,
                    extent=extent, aspect="auto", interpolation="nearest")
        axis.set_title(title)
    for axis in axes[:, 0]:
        axis.set_ylabel("Depth (m)")
    for axis in axes[1]:
        axis.set_xlabel("Range (m)")
    fig.suptitle("800 Hz narrow-source phase · preselected system 20267000 · 1500–1600 m",
                 fontsize=13)
    fig.text(0.5, 0.018,
             "Same ±π cyclic scale in all phase panels. Gray: lowest 10% target amplitude "
             "in this window; full-ROI raw metrics never mask these pixels.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.10, top=0.88,
                        wspace=0.16, hspace=0.22)
    phase_bar = fig.add_axes([0.88, 0.58, 0.012, 0.27])
    amplitude_bar = fig.add_axes([0.88, 0.15, 0.012, 0.27])
    fig.colorbar(image_phase, cax=phase_bar, label="Phase / error (rad)")
    fig.colorbar(image_amplitude, cax=amplitude_bar,
                 label="Relative amplitude (dB)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "Preselected first OOD system; local visual only, full ROI metrics elsewhere; exact family knows true synthetic mechanism but not test coefficients or target fields.",
        "config": config, "inputs_sha256": {
            key: digest(Path(value)) for key, value in config.items() if key.endswith("_npz")
        },
        "target_local_amplitude_mask_cutoff": cutoff,
        "masked_fraction": float(mask.mean()),
        "value_scale": "all phase maps and phase errors use [-pi,pi] circular scale",
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
