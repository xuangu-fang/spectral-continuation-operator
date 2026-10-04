"""Scientific figure: medium-variance compression versus coherent response error."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.run / "summary.json"
    manifest_path = args.run / "supplemental_manifest.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    singular = np.asarray(manifest["training_singular_values"])
    variance = singular**2
    ranks = sorted({row["rank"] for row in summary})
    omitted = np.array([1 - variance[:rank].sum() / variance.sum() for rank in ranks])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    axes[0].plot(ranks, np.maximum(omitted, 1e-12), "o-", color="#4c5e90", lw=2)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-12, 1e-2)
    axes[0].set_xlabel("PCA code dimension")
    axes[0].set_ylabel("Fraction of training medium variance omitted")
    axes[0].annotate("exact within training span", (511, 1e-12),
                     xytext=(280, 1e-10), arrowprops={"arrowstyle": "->"}, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    colors = {400: "#287d8e", 800: "#b9624a"}
    for frequency in [400, 800]:
        for source, style in [("original", "-"), ("narrow2m", "--")]:
            values = [next(row["mean_modal_parseval_raw"] for row in summary
                           if row["rank"] == rank and row["frequency"] == frequency
                           and row["source"] == source) for rank in ranks]
            axes[1].plot(ranks, values, marker="o", lw=1.8, color=colors[frequency],
                         linestyle=style,
                         label=f"{frequency} Hz, {'broad' if source == 'original' else 'new 2 m'}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("PCA code dimension")
    axes[1].set_ylabel("Oracle modal complex relative L2")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(ranks)
    fig.suptitle("32 correlated-spectral media: medium variance is not coherent-response fidelity\n"
                 "True test medium is used only for explicitly labelled PCA-projection oracles")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    for ext in ["png", "pdf"]:
        fig.savefig(args.output.with_suffix(f".{ext}"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({
        "run": str(args.run),
        "summary_sha256": digest(summary_path),
        "manifest_sha256": digest(manifest_path),
        "ranks": ranks,
        "omitted_training_variance": omitted.tolist(),
        "source": "posthoc true-medium projection oracle; not a deployable inverse",
    }, indent=2))


if __name__ == "__main__":
    main()
