"""Plot the training-free phase-identifiability and extrapolation-horizon audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

CONDITIONS = {
    "80": ("single 80 Hz", "#777777", "o"),
    "74+77+80": ("3 Hz coarse band", "#C44E52", "s"),
    "79+80": ("1 Hz terminal band", "#3274A1", "o"),
    "78+79+80": ("1 Hz three-point band", "#3A923A", "^"),
}


def run(args):
    records = []
    for path in args.input_dir.glob("target_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(payload)
    records.sort(key=lambda item: item["target_frequency_hz"])
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for condition, (label, color, marker) in CONDITIONS.items():
        frequencies = [record["target_frequency_hz"] for record in records]
        raw = [
            record["conditions"][condition]["methods"]["amplitude_spectral_carrier"][
                "complex_rel_l2"
            ]
            for record in records
        ]
        coherence = [
            record["conditions"][condition]["methods"]["amplitude_spectral_carrier"][
                "phase_coherence"
            ]
            for record in records
        ]
        axes[0].plot(frequencies, raw, color=color, marker=marker, linewidth=2.0, label=label)
        axes[1].plot(frequencies, coherence, color=color, marker=marker, linewidth=2.0, label=label)
    for axis in axes:
        axis.axvspan(90, 100, color="#EAEAEA", alpha=0.75)
        axis.axvline(100, color="#444444", linestyle="--", linewidth=1)
        axis.set_xlabel("Query frequency (Hz)")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Complex relative $L_2$")
    axes[0].set_title("Raw field error")
    axes[1].set_ylabel("Amplitude-weighted phase coherence")
    axes[1].set_title("Phase alignment")
    axes[1].legend(frameon=False, fontsize=9)
    figure.suptitle("Frequency resolution controls spectral-phase identifiability")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
