"""Freeze the uncertainty-conditioned proximal-prior M0 evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = {
    ("OpenFWI", 10): "pbno_uncertainty_proximal_prior_m0_openfwi_r10_dev",
    ("OpenFWI", 25): "pbno_uncertainty_proximal_prior_m0_openfwi_r25",
    ("OpenFWI", 50): "pbno_uncertainty_proximal_prior_m0_openfwi_r50",
    ("OpenFWI", 100): "pbno_uncertainty_proximal_prior_m0_openfwi_dev",
    ("Deepwave", 10): "pbno_uncertainty_proximal_prior_m0_deepwave_r10_dev",
    ("Deepwave", 25): "pbno_uncertainty_proximal_prior_m0_deepwave_r25",
    ("Deepwave", 50): "pbno_uncertainty_proximal_prior_m0_deepwave_r50",
    ("Deepwave", 100): "pbno_uncertainty_proximal_prior_m0_deepwave_dev",
    ("Ocean", 10): "pbno_uncertainty_proximal_prior_m0_ocean_r10_dev",
    ("Ocean", 25): "pbno_uncertainty_proximal_prior_m0_ocean_r25",
    ("Ocean", 50): "pbno_uncertainty_proximal_prior_m0_ocean_r50",
    ("Ocean", 100): "pbno_uncertainty_proximal_prior_m0_ocean_dev",
}


def run(artifacts: Path, output: Path) -> dict:
    rows = []
    for (domain, fraction), stem in RUNS.items():
        payload = json.loads((artifacts / stem / "summary.json").read_text())
        value = payload["formal_aggregate"][f"{domain}_k2"]
        rows.append(
            {
                "domain": domain,
                "receiver_percent": fraction,
                "baseline_error": value["baseline_mean"],
                "proximal_error": value["candidate_mean"],
                "relative_gain_percent": 100.0 * value["ratio_of_means_gain"],
                "bootstrap_low_percent": 100.0 * value["bootstrap_95_low"],
                "bootstrap_high_percent": 100.0 * value["bootstrap_95_high"],
                "system_win_rate_percent": 100.0 * value["system_win_rate"],
                "selected_blend": value["selected_blend"],
                "artifact": str(artifacts / stem),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "m0_mask_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    colors = {"OpenFWI": "#4C78A8", "Deepwave": "#F58518", "Ocean": "#54A24B"}
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for domain, color in colors.items():
        selected = sorted(
            (row for row in rows if row["domain"] == domain),
            key=lambda row: row["receiver_percent"],
        )
        x = [row["receiver_percent"] for row in selected]
        gain = [row["relative_gain_percent"] for row in selected]
        low = [row["bootstrap_low_percent"] for row in selected]
        high = [row["bootstrap_high_percent"] for row in selected]
        axes[0].plot(x, gain, marker="o", linewidth=2, color=color, label=domain)
        axes[0].fill_between(x, low, high, color=color, alpha=0.15)
        axes[1].plot(
            x,
            [row["baseline_error"] for row in selected],
            marker="o",
            linestyle="--",
            color=color,
            alpha=0.55,
        )
        axes[1].plot(
            x,
            [row["proximal_error"] for row in selected],
            marker="o",
            linewidth=2,
            color=color,
            label=domain,
        )
    axes[0].axhline(5.0, color="#D62728", linestyle="--", linewidth=1.5, label="5% gate")
    axes[0].set_xlabel("Observed receivers per prompt (%)")
    axes[0].set_ylabel("Relative raw-complex error reduction (%)")
    axes[0].set_title("Stable direction, insufficient effect size")
    axes[0].set_xscale("log")
    axes[0].set_xticks([10, 25, 50, 100], labels=["10", "25", "50", "100"])
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.2)
    axes[1].set_xlabel("Observed receivers per prompt (%)")
    axes[1].set_ylabel("Raw complex relative L2")
    axes[1].set_title("Solid: proximal prior; dashed: frozen baseline")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks([10, 25, 50, 100], labels=["10", "25", "50", "100"])
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "uncertainty_proximal_prior_m0.png", dpi=260)
    figure.savefig(output / "uncertainty_proximal_prior_m0.pdf")
    plt.close(figure)

    best = max(rows, key=lambda row: row["relative_gain_percent"])
    summary = {
        "status": "NO-GO: M0 promotion gate failed; M1 remains closed",
        "method_parameters": 29761,
        "best_formal_result": best,
        "promotion_gate_percent": 5.0,
        "domains_passing_gate": [],
        "main_rows": rows,
        "ablation_findings": {
            "shared_cross_domain_full_receiver": "exact fallback selected; no material gain",
            "domain_specific_full_receiver": "0.03%--0.18% gain",
            "denoising_pretraining": "did not improve the best masked result",
            "proximal_strength_0.1_to_10": "did not change the conclusion",
        },
        "decision": [
            "Keep analytic posterior and M4 prior composition as the deployed method.",
            "Do not launch query-decomposition M1 from this checkpoint.",
            "Reopen only with aligned cross-domain charts, new informative prompts, or a target-free correction-sign oracle above 10%.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "config": "configs/pbno_uncertainty_proximal_prior_m0_20260927.json",
                "runs": {f"{domain}_r{fraction}": stem for (domain, fraction), stem in RUNS.items()},
                "files": [
                    "summary.json",
                    "m0_mask_results.csv",
                    "uncertainty_proximal_prior_m0.png",
                    "uncertainty_proximal_prior_m0.pdf",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.artifacts, args.output), indent=2))


if __name__ == "__main__":
    main()
