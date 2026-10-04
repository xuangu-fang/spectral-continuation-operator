"""Static, fixed-seed comparison of medium diversity and training-label PCA spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit_waveguide_generator_identification import background_spec
from .coherent_waveguide_data import environment, sound_speed, spectral_environment


def residual_spectrum(path, key):
    with np.load(path) as data:
        values = data[key].reshape(len(data[key]), -1).astype(float)
    values -= values.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(values @ values.T)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    residual = np.maximum(1 - np.cumsum(eigenvalues) / eigenvalues.sum(), 1e-12)
    return residual


def plot(output, gaussian_labels, spectral_labels):
    # The first predeclared old/new environments are illustrations, not selected by error.
    old, new = environment(20261332), spectral_environment(20264600)
    ranges = np.linspace(0, old["range_m"], 257)
    depths = np.linspace(0, old["depth_m"], 129)
    deviations = [
        sound_speed(spec, ranges[:, None], depths[None])
        - sound_speed(background_spec(spec), ranges[:, None], depths[None])
        for spec in [old, new]
    ]
    old_residual = residual_spectrum(gaussian_labels, "target")
    new_residual = residual_spectrum(spectral_labels, "target32")
    vmax = max(float(np.max(np.abs(d))) for d in deviations)
    fig = plt.figure(figsize=(9.8, 7.2))
    grid = fig.add_gridspec(
        2, 3, width_ratios=[1, 1, 0.035], height_ratios=[1, 0.86],
        hspace=0.42, wspace=0.28
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for ax, delta, title in zip(
        axes,
        deviations,
        ["15-parameter Gaussian family", "Correlated 16 x 16 spectral family"],
        strict=True,
    ):
        image = ax.imshow(
            delta.T,
            extent=[0, old["range_m"] / 1000, old["depth_m"], 0],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{title}\nRMS perturbation {np.sqrt(np.mean(delta**2)):.1f} m/s")
        ax.set_xlabel("Range (km)")
        ax.set_ylabel("Depth (m)")
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    fig.colorbar(image, cax=fig.add_subplot(grid[0, 2]), label="Speed perturbation (m/s)")
    ax = fig.add_subplot(grid[1, :2])
    ranks = np.arange(1, len(old_residual) + 1)
    ax.semilogy(ranks, old_residual, color="#b9624a", lw=2, label="15-parameter Gaussian")
    ax.semilogy(ranks, new_residual, color="#287d8e", lw=2, label="Correlated spectral")
    ax.axhline(0.01, color="gray", linestyle="--", linewidth=1)
    ax.axvline(128, color="black", linestyle=":", linewidth=1)
    ax.set_xlim(1, 256)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("Training-medium PCA rank")
    ax.set_ylabel("Unexplained coefficient variance")
    ax.grid(alpha=0.17)
    ax.legend(loc="upper right")
    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.09, right=0.91)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{ext}"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    def rank_at(residual, threshold):
        return int(np.flatnonzero(residual <= threshold)[0] + 1)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "old_illustration_system": old["seed"],
                "spectral_illustration_system": new["seed"],
                "old_training_labels": str(gaussian_labels),
                "spectral_training_labels": str(spectral_labels),
                "ranks_for_99_percent": {
                    "gaussian": rank_at(old_residual, 0.01),
                    "spectral": rank_at(new_residual, 0.01),
                },
                "ranks_for_99_9_percent": {
                    "gaussian": rank_at(old_residual, 0.001),
                    "spectral": rank_at(new_residual, 0.001),
                },
                "explained_at_128": {
                    "gaussian": float(1 - old_residual[127]),
                    "spectral": float(1 - new_residual[127]),
                },
                "maps_are_dev_illustrations_not_validation_performance_selection": True,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian-labels", type=Path, required=True)
    parser.add_argument("--spectral-labels", type=Path, required=True)
    args = parser.parse_args()
    plot(args.output, args.gaussian_labels, args.spectral_labels)


if __name__ == "__main__":
    main()
