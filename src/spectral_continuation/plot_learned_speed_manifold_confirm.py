"""Paired, same-information phase fidelity on new taper-7.5 systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .audit_waveguide_spectral_context_robustness import digest


def paired(rows: list[dict], source: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    def select(method: str):
        return {r["system"]: r[metric] for r in rows if r["frequency"] == 800
                and r["source"] == source and r["method"] == method}

    speed = select("speed_nonlinear")
    q_one = select("q_pca_1e-09")
    q_two = select("q_pca_1e-07")
    if len(speed) != 16 or set(speed) != set(q_one) or set(speed) != set(q_two):
        raise ValueError("incomplete paired 16-system confirmation")
    systems = sorted(speed)
    # Per-case better of TWO preset q baselines is deliberately even stronger
    # than choosing one initialization precision per source using dev data.
    q = np.array([min(q_one[s], q_two[s]) for s in systems])
    return q, np.array([speed[s] for s in systems])


def run(results: Path, summary: Path, output: Path) -> None:
    rows = json.loads(results.read_text())
    complete = json.loads(summary.read_text())
    if not complete["all_confirm_gates_pass"]:
        raise ValueError("plot only after complete gate evaluation")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1))
    points = []
    metrics = ["complex_rel_l2", "energy_weighted_phase_rmse_rad"]
    for axis, metric in zip(axes, metrics, strict=True):
        for source, color, marker, display in [
            ("original", "#0072B2", "o", "Original broad source"),
            ("narrow2m", "#D55E00", "^", "Unseen 2 m narrow source"),
        ]:
            q, speed = paired(rows, source, metric)
            points.append({"metric": metric, "source": source,
                           "q_pca_per_case_best": q.tolist(),
                           "learned_speed": speed.tolist()})
            axis.scatter(q, speed, s=40, color=color, marker=marker, label=display,
                         edgecolors="white", linewidths=0.45, alpha=0.87)
        limits = (0.0007, 0.25)
        axis.plot(limits, limits, "--", color="#666666", linewidth=1,
                  label="Equal error")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_xlabel("Matched q-PCA255 error")
        axis.set_ylabel("Learned speed-manifold255 error")
        axis.grid(True, which="both", alpha=0.16)
    axes[0].set_title("Full-ROI raw complex RelL2 · 800 Hz")
    axes[1].set_title("Energy-weighted circular phase RMSE · 800 Hz")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Frozen learned speed manifold · 16 unseen taper-7.5 waveguides",
                 fontsize=13, y=0.97)
    fig.text(0.5, 0.065,
             "Same 1,024 medium labels, 255 system coordinates, and low-band context; "
             "each q-PCA point uses its stronger preset initialization.",
             ha="center", fontsize=8.5)
    fig.text(0.5, 0.035,
             "Synthetic one-way waveguide family only; this is not a second physics domain.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.83, wspace=0.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "Prospective independent 16-system statistical interpolation within one one-way waveguide synthetic family; paired same-information q-PCA vs learned speed variable; no target-frequency observations.",
        "results_sha256": digest(results), "summary_sha256": digest(summary),
        "per_system_points": points,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.results, args.summary, args.output)


if __name__ == "__main__":
    main()
