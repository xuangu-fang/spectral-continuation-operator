"""Separate local information and true-medium capacity into two honest panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .audit_waveguide_spectral_context_robustness import digest


def run(info: Path, info256: Path, capacity: Path, output: Path) -> None:
    local = json.loads(info.read_text())
    local256 = json.loads(info256.read_text())
    summary = json.loads(capacity.read_text())["summary"]
    if len(local["rows"]) != 18 or len(local256["rows"]) != 4 or len(summary) != 8:
        raise ValueError("incomplete local information or capacity audit")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = {11.3: "#087e8b", 17.0: "#c75b12"}
    for frequency in (11.3, 17.0):
        for rank, sample, style in ((64, local, "-"), (256, local256, "--")):
            selected = sorted(
                (r for r in sample["rows"]
                 if r["query_frequency_hz"] == frequency
                 and r["relative_complex_noise_rms"] == 0.01),
                key=lambda row: row["receiver_lines"],
            )
            axes[0].plot([r["receiver_lines"] for r in selected],
                         [r["linearized_expected_raw_rms"] for r in selected],
                         marker="o" if rank == 64 else "D", linewidth=1.8,
                         linestyle=style, color=colors[frequency],
                         label=f"rank {rank}, query {frequency}")
    axes[0].set_xticks([1, 2, 4])
    axes[0].set_xlabel("Observed vertical receiver lines")
    axes[0].set_ylabel("Model-based expected raw RelL2")
    axes[0].set_title("Local prior, assumed 1% noise\n(no target truth used)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)
    for frequency in (11.3, 17.0):
        selected = sorted(
            (r for r in summary if r["frequency"] == frequency
             and r["method"].startswith("true_speed_pca")),
            key=lambda row: int(row["method"].replace("true_speed_pca", "")),
        )
        axes[1].plot([int(r["method"].replace("true_speed_pca", "")) for r in selected],
                     [r["mean_raw"] for r in selected], marker="s", linewidth=2,
                     color=colors[frequency], label=f"query {frequency}")
    axes[1].set_xticks([64, 128, 256])
    axes[1].set_xlabel("Speed-PCA coordinate rank")
    axes[1].set_ylabel("Mean raw RelL2 on 8 used DEV systems")
    axes[1].set_title("TRUE-medium projection oracle\n(non-deployable extra information)")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("Sparse-wave feasibility: identifiability is not representation capacity",
                 fontsize=12.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "LEFT: rank64/256 Gaussian prior LOCAL LINEARIZATION at training mean/public DEV source and assumed 1% complex RMS noise; expected error, NOT measured reconstruction. RIGHT: true DEV medium projected to trained PCA then query solved; oracle READS TRUE MEDIUM, not deployable or same-information method. Different information assumptions; do not compare panel heights as a leaderboard.",
        "local_info_sha256": digest(info), "local_info_rank256_sha256": digest(info256),
        "capacity_summary_sha256": digest(capacity),
        "implementation_sha256": digest(__file__),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--info256", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.info, args.info256, args.capacity, args.output)


if __name__ == "__main__":
    main()
