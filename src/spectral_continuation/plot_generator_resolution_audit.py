"""Resolution/query audit from recorded outcomes, not fitted illustration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(name):
    return json.loads((Path("artifacts") / name / "results.json").read_text())


def curve(rows, method, modes, frequencies):
    return np.array(
        [
            np.mean(
                [
                    r["complex_rel_l2"]
                    for r in rows
                    if r["method"] == method
                    and r["decoder_modes"] == modes
                    and r["frequency"] == frequency
                ]
            )
            for frequency in frequencies
        ]
    )


def main():
    broad = load("coherent_waveguide_adaptive_modes_20261003")
    narrow = load("coherent_waveguide_narrow_source_modes_20261003")
    full = load("coherent_waveguide_narrow_source_modes_full_20261003")
    optimized = load("coherent_waveguide_narrow_source_optimization_control_20261003")
    frequencies = np.array([140, 200, 300, 400])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for rows, color, label in [
        (broad, "#4477aa", "Original broad source"),
        (narrow, "#cc6677", "New 2 m source"),
    ]:
        for modes, style in [(32, "--"), (64, "-")]:
            axes[0].plot(
                frequencies,
                curve(rows, "field_neural", modes, frequencies),
                "o" + style,
                color=color,
                label=f"{label}, {modes} modes",
            )
    axes[0].set_title(
        "Resolution matters when new components are excited\nPaired 2 systems; same frozen generator; seed 42"
    )
    methods = ["field_neural", "field_neural_seed43", "field_neural_seed44"]
    seed_values = np.array([curve(full, method, 64, frequencies) for method in methods])
    axes[1].plot(
        frequencies, seed_values.mean(0), "o-", color="#228833", label="Neural: 3-seed mean"
    )
    axes[1].fill_between(
        frequencies,
        seed_values.min(0),
        seed_values.max(0),
        color="#228833",
        alpha=0.2,
        label="Training-seed min–max (not CI)",
    )
    for rows, method, label, color in [
        (full, "kernel_generator", "Kernel generator", "#4477aa"),
        (optimized, "nonlinear_context_2000", "Context inverse, 2000 steps", "#222222"),
        (full, "true_medium_oracle", "True-medium projection ORACLE", "#aa3377"),
    ]:
        axes[1].plot(
            frequencies, curve(rows, method, 64, frequencies), "o-", color=color, label=label
        )
    axes[1].set_title(
        "Frozen low-band encoding, high-frequency narrow source\nAll 8 dev systems; 64-mode decoder for every method"
    )
    for ax in axes:
        ax.set(xlabel="Query frequency (Hz)", ylabel="Raw complex relative L2", xticks=frequencies)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Fixed 1024-real-coefficient latent does not imply fixed decoding cost or frequency-independent error",
        fontsize=10,
    )
    output = Path("reports/assets/coherent_continuous_20261003")
    fig.savefig(output / "shared_generator_resolution_audit.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
