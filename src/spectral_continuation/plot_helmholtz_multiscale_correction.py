"""Make the classical-baseline physical-range correction visibly auditable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .audit_waveguide_spectral_context_robustness import digest


def run(invalid_path: Path, corrected_path: Path, output: Path) -> None:
    invalid = json.loads(invalid_path.read_text())["summary"]
    corrected = json.loads(corrected_path.read_text())["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), sharey=True)
    plotted = []
    for axis, frequency in zip(axes, [11.3, 17.0], strict=True):
        old = sorted((r for r in invalid if r["frequency"] == frequency),
                     key=lambda r: r["observation_grid"])
        new = sorted((r for r in corrected if r["frequency"] == frequency),
                     key=lambda r: r["observation_grid"])
        if [r["observation_grid"] for r in old] != [65, 129, 257]:
            raise ValueError("invalid baseline grid series incomplete")
        if [r["observation_grid"] for r in new] != [65, 129, 257]:
            raise ValueError("corrected baseline grid series incomplete")
        axis.plot([65, 129, 257], [r["mean_raw"] for r in old], "s--",
                  color="#D55E00", linewidth=2, markersize=6,
                  label="Invalid old q >= 1 clipping")
        axis.plot([65, 129, 257], [r["mean_raw"] for r in new], "o-",
                  color="#009E73", linewidth=2, markersize=6,
                  label="Corrected q in [0.5, 6.25]")
        axis.set_title(f"Query frequency {frequency:g}")
        axis.set_xlabel("Full low-band complex-field grid per frequency")
        axis.set_xticks([65, 129, 257], ["65×65", "129×129", "257×257"])
        axis.set_yscale("log")
        axis.set_ylim(2e-5, 0.3)
        axis.grid(True, which="both", alpha=0.18)
        plotted.append({"frequency": frequency, "invalid": old, "corrected": new})
    axes[0].set_ylabel("Mean complex relative L2 on same grid-257 teacher")
    axes[1].legend(loc="center right", frameon=False, fontsize=8.5)
    fig.suptitle("A wrong physical bound created false learning headroom", fontsize=13)
    fig.text(0.5, 0.055,
             "Same 8 used 2D multiscale pilot systems, same source/PDE/data. Old Gaussian-only q >= 1 bound clips ~40% valid pixels.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.82, wspace=0.13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "Technical error audit, not a paper method comparison. Initial q>=1 classical baseline is invalid for media with speed above background. Corrected result is the strong same-information full-field baseline.",
        "invalid_summary_sha256": digest(invalid_path),
        "corrected_summary_sha256": digest(corrected_path),
        "plotted": plotted,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invalid", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.invalid, args.corrected, args.output)


if __name__ == "__main__":
    main()
