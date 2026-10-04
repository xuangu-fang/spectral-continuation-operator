"""Show q-code capacity versus low-band identifiability without mixing information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .audit_waveguide_spectral_context_robustness import digest


def mean_raw(rows: list[dict], *, source: str, method: str) -> float:
    selected = [r["complex_rel_l2"] for r in rows
                if r["source"] == source and r["method"] == method
                and r["frequency"] == 800]
    if len(selected) != 16:
        raise ValueError(f"expected sixteen paired systems: {source} {method}")
    return float(np.mean(selected))


def run(oracle_path: Path, high_path: Path, original_path: Path,
        projection_path: Path, map_dev_path: Path, map_check_path: Path,
        output: Path) -> None:
    oracle = json.loads(oracle_path.read_text())["oracle_summary"]
    high = json.loads(high_path.read_text())["summary"]
    original = json.loads(original_path.read_text())
    projection = json.loads(projection_path.read_text())["oracle_summary"]
    map_rows = json.loads(map_dev_path.read_text()) + json.loads(map_check_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.1), sharey=True)
    plotted = []
    for axis, source, label in zip(
        axes, ["original", "narrow2m"], ["Broad source", "Unseen narrow 2 m source"],
        strict=True,
    ):
        q_oracle = sorted(
            [r for r in oracle if r["source"] == source], key=lambda r: r["rank"]
        )
        ranks = [r["rank"] for r in q_oracle]
        oracle_raw = [r["mean_raw"] for r in q_oracle]
        q255 = min(mean_raw(original, source=source, method=f"q_pca_{p}")
                   for p in ["1e-09", "1e-07"])
        q768 = min(r["mean_raw"] for r in high if r["source"] == source)
        q768_map = mean_raw(map_rows, source=source, method="q_pca768_map_1e-06")
        speed = mean_raw(original, source=source, method="speed_nonlinear")
        speed_oracle = next(
            r["mean_raw"] for r in projection
            if r["family"] == "cosine_taper7p5_new16"
            and r["method"] == "speed_true_medium_euclidean_oracle"
            and r["source"] == source
        )
        axis.plot(ranks, oracle_raw, "o:", color="#0072B2", linewidth=2.0,
                  markersize=6, label="q-PCA true-medium oracle")
        axis.plot([255, 768], [q255, q768], "s-", color="#D55E00",
                  linewidth=2.0, markersize=7, label="q-PCA low-band GN, no ongoing prior")
        axis.scatter([768], [q768_map], marker="^", s=85, color="#CC79A7",
                     edgecolor="black", linewidth=0.4, zorder=5,
                     label="q-PCA768 low-band MAP (posthoc)")
        axis.scatter([255], [speed], marker="*", s=170, color="#009E73",
                     edgecolor="black", linewidth=0.4, zorder=5,
                     label="Speed-255 low-band deployment")
        axis.scatter([255], [speed_oracle], marker="D", s=60, color="#009E73",
                     facecolor="white", edgecolor="#009E73", linewidth=1.5,
                     zorder=5, label="Speed-255 true-medium oracle")
        axis.set_yscale("log")
        axis.set_xlim(220, 810)
        axis.set_ylim(0.0017, 0.22)
        axis.set_xticks(ranks)
        axis.set_title(label)
        axis.set_xlabel("Per-environment real code dimension")
        axis.grid(True, which="both", alpha=0.17)
        plotted.append({
            "source": source, "q_oracle": q_oracle,
            "q255_deploy_best_fixed_precision_mean_raw": q255,
            "q768_deploy_best_fixed_precision_mean_raw": q768,
            "q768_map_1e-6_posthoc_mean_raw": q768_map,
            "speed255_deploy_mean_raw": speed,
            "speed255_true_medium_oracle_mean_raw": speed_oracle,
        })
    axes[0].set_ylabel("800 Hz complex relative L2 (full ROI)")
    axes[1].legend(loc="center right", frameon=False, fontsize=8)
    fig.suptitle("More capacity alone does not secure high-frequency phase prediction",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.07,
             "Dotted q curve and hollow speed markers READ true test media; solid low-band methods use only 60/100/140 Hz context.",
             ha="center", fontsize=8.5)
    fig.text(0.5, 0.04,
             "MAP strength was chosen from used first 8; latter 8 is a conditional check, not an independent test. q-768 uses 3× q-255 code/storage.",
             ha="center", fontsize=8.4)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.82, wspace=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "Posthoc same used16 systems. True-medium oracle and deployable low-band estimates are intentionally visually distinguished. No independent confirmation of rank768 method.",
        "inputs_sha256": {str(p): digest(p) for p in
                          [oracle_path, high_path, original_path, projection_path,
                           map_dev_path, map_check_path]},
        "plotted": plotted,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--map-dev", type=Path, required=True)
    parser.add_argument("--map-check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.oracle, args.high, args.original, args.projection,
        args.map_dev, args.map_check, args.output)


if __name__ == "__main__":
    main()
