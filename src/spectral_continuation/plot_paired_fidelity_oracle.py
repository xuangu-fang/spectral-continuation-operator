"""Plot the paired-fidelity calibration oracle gate from its frozen summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = summary["methods"]
    cheap = methods["cheap_query"]["per_query"]
    ar1 = methods["dense_calibration_ar1"]["per_query"]
    velocity = methods["damped_velocity_linear_state"]["per_query"]
    rational = methods["global_rational_ar4"]["per_query"]
    frequencies = np.asarray([row["query_hz"] for row in cheap], dtype=float)
    cheap_error = np.asarray([row["complex_rel_l2"] for row in cheap])
    ar1_error = np.asarray([row["complex_rel_l2"] for row in ar1])
    velocity_error = np.asarray([row["complex_rel_l2"] for row in velocity])
    rational_error = np.asarray([row["complex_rel_l2"] for row in rational])
    improvement = 100.0 * (cheap_error - ar1_error) / cheap_error

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
    axes[0].plot(frequencies, cheap_error, "o-", label="Cheap solver")
    axes[0].plot(frequencies, ar1_error, "o-", label="Dense calibration AR(1)")
    axes[0].plot(frequencies, velocity_error, "o--", label="Damped two-state")
    axes[0].plot(frequencies, rational_error, "o--", label="Global rational AR(4)")
    axes[0].set(xlabel="Unobserved query frequency (Hz)", ylabel="Raw complex RelL2")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(30.0, color="black", linestyle="--", linewidth=1, label="30% gate")
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].plot(frequencies, improvement, "o-", color="#d55e00", label="AR(1) gain")
    axes[1].set(
        xlabel="Unobserved query frequency (Hz)",
        ylabel="Relative error reduction (%)",
        ylim=(min(-10.0, float(improvement.min()) - 4.0), 52.0),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(
        0.98,
        0.05,
        "Macro: 15.7%  |  Gate: FAIL",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#a52a2a",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == "__main__":
    main()
