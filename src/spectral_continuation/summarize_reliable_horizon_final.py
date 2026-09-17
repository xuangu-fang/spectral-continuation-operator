"""Create a compact final figure for the Reliable-horizon Map POC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> None:
    instance = {"Ocean": _read(args.ocean_risk), "Synthetic": _read(args.synthetic_risk)}
    post_reset = {
        "Ocean": _read(args.ocean_calibrated_risk),
        "Synthetic": _read(args.synthetic_calibrated_risk),
    }
    repeated = {
        "Ocean": _read(args.ocean_repeated),
        "Synthetic": _read(args.synthetic_repeated),
    }
    spatial_paths = {
        "Ocean": args.ocean_spatial,
        "Synthetic": args.synthetic_spatial,
    }
    spatial = {domain: [_read(path) for path in paths] for domain, paths in spatial_paths.items()}

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    methods = ("distance_only", "disagreement_only", "physics_combo")
    method_labels = ("Distance", "Disagreement", "Physics combo")
    x = np.arange(len(instance))
    width = 0.24
    for offset, (method, label) in enumerate(zip(methods, method_labels, strict=True)):
        axes[0, 0].bar(
            x + (offset - 1) * width,
            [instance[domain]["results"][method]["within_frequency_auroc"] for domain in instance],
            width,
            label=label,
        )
    axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(
        xticks=x,
        xticklabels=list(instance),
        ylabel="Fixed-frequency AUROC",
        title="A  Instance risk is not distance alone",
        ylim=(0.3, 1.0),
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    for domain, values in spatial.items():
        grids = [record["patch_grid"][0] * record["patch_grid"][1] for record in values]
        aucs = [
            record["results"]["local_combo"]["within_environment_frequency_spatial_auroc"]
            for record in values
        ]
        axes[0, 1].plot(grids, aucs, marker="o", label=domain)
    axes[0, 1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(
        xlabel="Number of spatial patches",
        ylabel="Within-field spatial AUROC",
        title="B  Spatial localization across scales",
        xscale="log",
        ylim=(0.5, 1.0),
    )
    axes[0, 1].set_xticks([8, 32, 128], labels=["2x4", "4x8", "8x16"])
    axes[0, 1].legend(frameon=False, fontsize=8)

    ocean = repeated["Ocean"]
    base = ocean["policies"]["no_calibration"]
    for label, prefix in (
        ("Best fixed", "fixed_up_to"),
        ("Risk-adaptive", "adaptive_up_to"),
        ("Random", "random_exactly"),
        ("Instance oracle", "oracle_up_to"),
    ):
        records = [base] + [ocean["policies"][f"{prefix}_{action}"] for action in range(1, 4)]
        axes[1, 0].plot(
            [record["mean_unique_complex_observations"] for record in records],
            [record["macro_complex_rel_l2"] for record in records],
            marker="o",
            label=label,
        )
    axes[1, 0].set(
        xlabel="Unique complex observations / env",
        ylabel="Macro complex RelL2",
        title="C  Ocean sequential error--cost frontier",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    for domain, marker in (("Ocean", "o"), ("Synthetic", "s")):
        auc = post_reset[domain]["results"]["disagreement_only"]["failure_auroc"]
        base_error = repeated[domain]["policies"]["no_calibration"]["macro_complex_rel_l2"]
        best_adaptive = min(
            repeated[domain]["policies"][f"adaptive_up_to_{action}"]["macro_complex_rel_l2"]
            for action in range(1, 4)
        )
        improvement = 100 * (base_error - best_adaptive) / base_error
        axes[1, 1].scatter(auc, improvement, s=90, marker=marker, label=domain)
        axes[1, 1].annotate(domain, (auc, improvement), xytext=(5, 4), textcoords="offset points")
    axes[1, 1].set(
        xlabel="Post-reset failure AUROC",
        ylabel="Best adaptive error reduction (%)",
        title="D  Predictable failure is not always repairable",
        xlim=(0.85, 1.0),
        ylim=(-0.5, 10.5),
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.22, linewidth=0.6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "reliable_horizon_final.png", dpi=250)
    figure.savefig(args.output_dir / "reliable_horizon_final.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocean-risk", type=Path, required=True)
    parser.add_argument("--synthetic-risk", type=Path, required=True)
    parser.add_argument("--ocean-calibrated-risk", type=Path, required=True)
    parser.add_argument("--synthetic-calibrated-risk", type=Path, required=True)
    parser.add_argument("--ocean-repeated", type=Path, required=True)
    parser.add_argument("--synthetic-repeated", type=Path, required=True)
    parser.add_argument("--ocean-spatial", type=Path, action="append", required=True)
    parser.add_argument("--synthetic-spatial", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
