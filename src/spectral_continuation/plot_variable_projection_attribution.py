"""True-medium projection phase-fidelity mechanism, explicitly oracle-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .audit_waveguide_spectral_context_robustness import digest


def pairs(rows: list[dict], family: str, source: str) -> tuple[np.ndarray, np.ndarray]:
    q = {r["system"]: r["complex_rel_l2"] for r in rows
         if r["family"] == family and r["source"] == source
         and r["method"] == "q_true_medium_euclidean_oracle"}
    speed = {r["system"]: r["complex_rel_l2"] for r in rows
             if r["family"] == family and r["source"] == source
             and r["method"] == "speed_true_medium_euclidean_oracle"}
    if set(q) != set(speed) or len(q) not in (8, 16):
        raise ValueError("incomplete paired true-medium projection oracle")
    systems = sorted(q)
    return np.array([q[s] for s in systems]), np.array([speed[s] for s in systems])


def run(results: Path, summary: Path, output: Path) -> None:
    rows = json.loads(results.read_text())
    complete = json.loads(summary.read_text())
    if not complete["all_mechanism_checks_pass"]:
        raise ValueError("expected all preregistered mechanism checks to pass")
    families = [
        ("cosine_taper7p5_new16", "Correlated cosine · new 16 · rank 255"),
        ("gaussian_old8_dev", "Local Gaussians · used DEV 8 · rank 128"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.1), sharex=True, sharey=True)
    plotted = []
    for axis, (family, display) in zip(axes, families, strict=True):
        for source, color, marker, name in [
            ("original", "#0072B2", "o", "Original broad source"),
            ("narrow2m", "#D55E00", "^", "Unseen 2 m narrow source"),
        ]:
            q, speed = pairs(rows, family, source)
            axis.scatter(q, speed, s=42, marker=marker, c=color,
                         edgecolors="white", linewidths=0.45, alpha=0.85,
                         label=name)
            plotted.append({"family": family, "source": source,
                            "q_projection_oracle_raw": q.tolist(),
                            "speed_projection_oracle_raw": speed.tolist()})
        axis.plot([0.0007, 0.25], [0.0007, 0.25], "--", color="#777777",
                  linewidth=1, label="Equal error")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(0.0007, 0.25)
        axis.set_ylim(0.0007, 0.25)
        axis.set_title(display)
        axis.set_xlabel("True-medium q-PCA Euclidean projection raw")
        axis.set_ylabel("True-medium speed-PCA Euclidean projection raw")
        axis.grid(True, which="both", alpha=0.15)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Where the representation loses phase: true-medium projection oracles",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.067,
             "Oracle: both axes READ the true test medium; neither is a deployable low-band prediction.",
             ha="center", fontsize=8.6)
    fig.text(0.5, 0.037,
             "Same one-way PDE, but different families/ranks/splits. Euclidean projections are not best-response capacity bounds.",
             ha="center", fontsize=8.3)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.82, wspace=0.19)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "TRUE-TEST-MEDIUM Euclidean projection oracles, not deployable scores or best-response lower bounds; same one-way PDE, two statistical families with different ranks and splits.",
        "results_sha256": digest(results), "summary_sha256": digest(summary),
        "paired_points": plotted,
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
