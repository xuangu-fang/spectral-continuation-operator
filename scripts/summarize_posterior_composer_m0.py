"""Create the frozen M0 posterior-composer table and diagnostic figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run(validation_path: Path, test_path: Path, output: Path) -> None:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, test_value in test["aggregate"].items():
        validation_baseline = validation["validation_kernel_anchor"][key]
        validation_learned = validation["validation"][key]["candidate_mean"]
        rows.append(
            {
                "setting": key,
                "domain": key.rsplit("_k", 1)[0],
                "budget": int(key.rsplit("_k", 1)[1]),
                "validation_gain_percent": 100.0
                * (validation_baseline - validation_learned)
                / validation_baseline,
                "test_baseline": test_value["baseline_mean"],
                "test_learned": test_value["learned_mean"],
                "test_gain_percent": 100.0 * test_value["ratio_of_means_gain"],
                "bootstrap_95_low_percent": 100.0
                * test_value["bootstrap_95_low"],
                "bootstrap_95_high_percent": 100.0
                * test_value["bootstrap_95_high"],
                "system_win_rate": test_value["system_win_rate"],
            }
        )
    rows.sort(key=lambda value: (value["domain"], value["budget"]))
    with (output / "m0_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), constrained_layout=True)
    colors = {"Deepwave": "#0072B2", "Ocean": "#D55E00", "OpenFWI": "#009E73"}
    markers = {"Deepwave": "o", "Ocean": "s", "OpenFWI": "^"}
    for domain, color in colors.items():
        selected = [row for row in rows if row["domain"] == domain]
        budget = [row["budget"] for row in selected]
        baseline = [row["test_baseline"] for row in selected]
        axes[0].plot(
            budget,
            baseline,
            marker=markers[domain],
            color=color,
            label=domain,
            linewidth=1.8,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks([1, 2, 4, 8])
    axes[0].set_xlabel("Prompt budget K")
    axes[0].set_ylabel("Raw complex relative L2")
    axes[0].set_title("(a) Frozen risk-calibrated operator")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)

    positions = np.arange(len(rows))
    validation_gain = np.asarray([row["validation_gain_percent"] for row in rows])
    test_gain = np.asarray([row["test_gain_percent"] for row in rows])
    axes[1].axhline(0.0, color="0.35", linewidth=1.0)
    axes[1].bar(
        positions,
        test_gain,
        color=[colors[row["domain"]] for row in rows],
        alpha=0.82,
        label="Unseen test",
    )
    axes[1].scatter(
        positions,
        validation_gain,
        facecolors="none",
        edgecolors="black",
        marker="o",
        s=25,
        linewidths=0.9,
        label="Calibration selection",
        zorder=3,
    )
    axes[1].set_xticks(
        positions,
        [f"{row['domain']}\nK={row['budget']}" for row in rows],
        rotation=55,
        ha="right",
    )
    axes[1].set_ylabel("Learned correction gain (%)")
    axes[1].set_title("(b) Correction does not survive system holdout")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, loc="lower left")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"posterior_composer_m0.{suffix}", dpi=220)
    plt.close(figure)

    summary = {
        "decision": "M0 learned-correction gate failed; retain zero-update operator",
        "parameters": validation["parameters"],
        "formal_test_prompt_seeds": test["seeds"],
        "worst_test_gain_percent": float(test_gain.min()),
        "best_test_gain_percent": float(test_gain.max()),
        "mean_test_gain_percent": float(test_gain.mean()),
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.validation, args.test, args.output)


if __name__ == "__main__":
    main()
