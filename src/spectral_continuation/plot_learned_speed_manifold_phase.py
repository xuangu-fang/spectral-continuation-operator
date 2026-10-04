"""Preselected new-system phase map; display masking never changes metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .audit_waveguide_frequency_scaling import propagate_code
from .audit_waveguide_new_systems import context_data
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_rank256_inference import infer as infer_q
from .coherent_waveguide_data import depth_basis


def field(modal: np.ndarray, depths: np.ndarray, depth_m: float,
          source_index: int) -> np.ndarray:
    basis = depth_basis(depths, depth_m, modal.shape[1])
    return basis @ modal[:, :, source_index].T


def run(config: dict, output: Path) -> None:
    frozen_view = (
        config["system"], config["frequency"], config["source_index"],
        config["range_min_m"], config["range_max_m"],
        config["depth_min_m"], config["depth_max_m"],
        config["amplitude_mask_quantile"], config["phase_error_display_rad"],
        config["q_initial_prior_precision"],
    )
    if frozen_view != (20268000, 800, 1, 1500, 1600, 5, 95, 0.10, 0.15, 1e-9):
        raise ValueError("first independent system and fixed view required")
    source = json.loads(Path(config["source_config"]).read_text())
    confirm = json.loads(Path(config["confirm_config"]).read_text())
    dev = json.loads(Path(confirm["dev_config"]).read_text())
    with np.load(config["truth_npz"]) as data:
        truth_modal = data["modal_coefficients"]
        ranges, depths = data["ranges_m"], data["depths_m"]
        spec = json.loads(str(data["spec"]))
    with np.load(config["confirm_code"]) as data:
        speed_beta = data["speed_beta"]
        if str(data["spec"]) != json.dumps(spec):
            raise ValueError("code/truth system spec mismatch")
    with np.load(config["frozen_priors"]) as data:
        mean_q, factor_q = data["mean_q"], data["factor_q"]
    context_ranges, observed = context_data(spec, source)
    q_source = {**source, "initial_prior_precision": config["q_initial_prior_precision"]}
    q_beta, _ = infer_q(
        spec, observed, context_ranges, q_source, mean_q, factor_q,
        dev["coarse_steps"], dev["fine_steps"],
    )
    predictions = {}
    for name, beta in [("q_pca", q_beta), ("speed", speed_beta)]:
        predictions[name] = propagate_code(
            spec, config["frequency"], ranges, source["decoder_cells"],
            source["decoder_modes"], source["teacher_quadrature_nodes"],
            source["eigh_device"], source["eigh_batch"], beta,
        )
    take_r = (ranges >= config["range_min_m"]) & (ranges <= config["range_max_m"])
    take_z = (depths >= config["depth_min_m"]) & (depths <= config["depth_max_m"])
    x, z = ranges[take_r], depths[take_z]
    target = field(truth_modal[take_r], z, spec["depth_m"], config["source_index"])
    q_field = field(predictions["q_pca"][take_r], z, spec["depth_m"],
                    config["source_index"])
    speed_field = field(predictions["speed"][take_r], z, spec["depth_m"],
                        config["source_index"])
    amplitude = np.abs(target)
    cutoff = float(np.quantile(amplitude, config["amplitude_mask_quantile"]))
    mask = amplitude < cutoff
    phases = [np.ma.array(np.angle(value), mask=mask)
              for value in (target, q_field, speed_field)]
    differences = [
        np.ma.array(np.angle(value * np.conj(target)), mask=mask)
        for value in (q_field, speed_field)
    ]
    cmap = plt.get_cmap("twilight_shifted").copy()
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), sharex=True, sharey=True)
    extent = (float(x.min()), float(x.max()), float(z.max()), float(z.min()))
    titles = ["Numerical truth phase", "Matched q-PCA255 phase",
              "Learned speed-manifold255 phase"]
    for axis, values, title in zip(axes[0], phases, titles, strict=True):
        phase_image = axis.imshow(
            values, cmap=cmap, vmin=-np.pi, vmax=np.pi, extent=extent,
            aspect="auto", interpolation="nearest",
        )
        axis.set_title(title)
    db = 20 * np.log10(np.maximum(amplitude / np.max(amplitude), 1e-4))
    amplitude_image = axes[1, 0].imshow(
        db, cmap="viridis", vmin=-40, vmax=0, extent=extent,
        aspect="auto", interpolation="nearest",
    )
    axes[1, 0].set_title("Truth amplitude (dB)")
    error_bound = config["phase_error_display_rad"]
    for axis, values, title in zip(
        axes[1, 1:], differences,
        ["q-PCA minus truth phase", "Speed manifold minus truth phase"], strict=True,
    ):
        error_image = axis.imshow(
            values, cmap="coolwarm", vmin=-error_bound, vmax=error_bound,
            extent=extent, aspect="auto", interpolation="nearest",
        )
        axis.set_title(title)
    for axis in axes[:, 0]:
        axis.set_ylabel("Depth (m)")
    for axis in axes[1]:
        axis.set_xlabel("Range (m)")
    fig.suptitle(
        "800 Hz unseen narrow source · first new system 20268000 · 1500–1600 m",
        fontsize=13,
    )
    fig.text(0.5, 0.018,
             "Top phase: ±π; bottom phase-error zoom: ±0.15 rad (saturates beyond). "
             "Gray: lowest 10% target amplitude in this view; full-ROI metrics are unmasked.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.10, top=0.88,
                        wspace=0.16, hspace=0.22)
    phase_bar = fig.add_axes([0.88, 0.58, 0.012, 0.27])
    amplitude_bar = fig.add_axes([0.88, 0.15, 0.012, 0.27])
    error_bar = fig.add_axes([0.94, 0.15, 0.012, 0.27])
    fig.colorbar(phase_image, cax=phase_bar, label="Phase (rad)")
    fig.colorbar(amplitude_image, cax=amplitude_bar,
                 label="Relative amplitude (dB)")
    fig.colorbar(error_image, cax=error_bar, label="Phase residual (rad)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "Preselected first fresh taper7.5 system and narrow source; display-only lowest-10-percent-amplitude mask; raw full-ROI metrics unmasked; q inference rerun with frozen 1e-9 initialization.",
        "config": config,
        "input_sha256": {key: digest(value) for key, value in config.items()
                         if key.endswith("_npz") or key in
                         ["confirm_code", "frozen_priors", "source_config", "confirm_config"]},
        "display_masked_fraction": float(mask.mean()),
        "display_local_amplitude_cutoff": cutoff,
        "value_scale": "top phase in [-pi,pi]; bottom phase residual display saturated at [-0.15,0.15] rad",
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
