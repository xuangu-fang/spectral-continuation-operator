"""Create the frozen M1 paper table and figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _row(summary: dict, domain: str, budget: int, method: str) -> dict:
    return next(
        value
        for value in summary["domains"][domain]["budgets"][str(budget)]
        if value["method"] == method
    )


def _relative_gap(candidate: float, reference: float) -> float:
    return candidate / reference - 1.0


def run(
    baseline_path: Path,
    hybrid_path: Path,
    utility_path: Path,
    coordinate_utility_path: Path,
    population_path: Path,
    ood_path: Path,
    output: Path,
) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
    utility = json.loads(utility_path.read_text(encoding="utf-8"))
    coordinate_utility = json.loads(
        coordinate_utility_path.read_text(encoding="utf-8")
    )
    population = json.loads(population_path.read_text(encoding="utf-8"))
    ood = json.loads(ood_path.read_text(encoding="utf-8"))
    domains = ("Deepwave", "Ocean")
    budgets = (1, 2, 4, 8)
    summary = {"domains": {}, "decisions": {}}
    csv_rows = []
    for domain in domains:
        auc = baseline["domains"][domain]["auc"]
        hybrid_auc = hybrid["domains"][domain]["auc"]["continuous_hybrid_top4"]
        heuristic = min(auc[name] for name in ("nearest", "farthest", "d_optimal"))
        k2_population = _row(baseline, domain, 2, "population")["error"]
        oracle = baseline["domains"][domain]["k2_truth_oracle"]["error"]
        domain_summary = {
            "auc": auc,
            "hybrid_auc": hybrid_auc,
            "exact_gain_vs_random_auc": 1.0 - auc["target_risk"] / auc["random"],
            "exact_gain_vs_best_heuristic_auc": 1.0 - auc["target_risk"] / heuristic,
            "pure_utility_auc_gap": _relative_gap(
                auc["continuous_utility"], auc["target_risk"]
            ),
            "hybrid_auc_gap": _relative_gap(hybrid_auc, auc["target_risk"]),
            "population_auc_gap": _relative_gap(auc["population"], auc["target_risk"]),
            "k2_truth_oracle_gap_vs_population": 1.0 - oracle / k2_population,
            "utility": utility["domains"][domain],
            "coordinate_holdout_utility": coordinate_utility["domains"][domain],
            "ood": ood["domains"][domain],
        }
        summary["domains"][domain] = domain_summary
        for budget in budgets:
            for method in (
                "random",
                "nearest",
                "farthest",
                "d_optimal",
                "target_risk",
                "population",
                "continuous_utility",
            ):
                csv_rows.append(_row(baseline, domain, budget, method))
            csv_rows.append(
                _row(hybrid, domain, budget, "continuous_hybrid_top4")
            )
    summary["population_offline_seconds"] = population["timing_seconds"]
    summary["decisions"] = {
        "untouched_iid_gate": True,
        "heldout_action_regret_gate": all(
            utility["domains"][domain]["heldout_action_mean_regret"] < 0.05
            for domain in domains
        ),
        "pure_utility_auc_gate": all(
            summary["domains"][domain]["pure_utility_auc_gap"] <= 0.02
            for domain in domains
        ),
        "hybrid_auc_gate": all(
            summary["domains"][domain]["hybrid_auc_gap"] <= 0.02
            for domain in domains
        ),
        "five_x_speed_gate": False,
        "strict_coordinate_generalization_gate": False,
        "ocean_ood_gate": False,
        "response_adaptive_cross_domain_gate": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "paper_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    colors = {
        "random": "#9A9A9A",
        "nearest": "#4C78A8",
        "d_optimal": "#F58518",
        "target_risk": "#54A24B",
        "population": "#B279A2",
        "continuous_hybrid_top4": "#E45756",
    }
    labels = {
        "random": "random",
        "nearest": "nearest target",
        "d_optimal": "D-optimal",
        "target_risk": "exact target risk",
        "population": "population search",
        "continuous_hybrid_top4": "learned proposal + cert.",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for axis, domain in zip(axes[0], domains, strict=True):
        for method, color in colors.items():
            source = hybrid if method == "continuous_hybrid_top4" else baseline
            values = [_row(source, domain, budget, method)["error"] for budget in budgets]
            axis.plot(
                budgets,
                values,
                marker="o",
                color=color,
                label=labels[method],
                linewidth=2.0,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(budgets, labels=[str(value) for value in budgets])
        axis.set_title(f"{domain}: untouched IID")
        axis.set_xlabel("probe budget K")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("raw complex relative L2")
    axes[0, 1].legend(frameon=False, fontsize=8)

    methods = ("nearest", "d_optimal", "target_risk", "population", "continuous_hybrid_top4")
    positions = np.arange(len(methods))
    width = 0.36
    for offset, domain in enumerate(domains):
        values = []
        for method in methods:
            source = hybrid if method == "continuous_hybrid_top4" else baseline
            values.append(100.0 * _row(source, domain, 2, method)["gain_vs_random"])
        axes[1, 0].bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            label=domain,
        )
    axes[1, 0].set_xticks(
        positions,
        labels=["nearest", "D-opt", "exact", "population", "learned+cert."],
        rotation=18,
    )
    axes[1, 0].set_ylabel("K=2 error reduction vs random (%)")
    axes[1, 0].set_title("Strong acquisition baselines")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.2)

    metric_names = (
        "pair-held-out regret",
        "coordinate-held-out regret",
        "pure utility AUC gap",
        "hybrid AUC gap",
    )
    positions = np.arange(len(metric_names))
    for offset, domain in enumerate(domains):
        values = [
            100.0 * utility["domains"][domain]["heldout_action_mean_regret"],
            100.0
            * coordinate_utility["domains"][domain]["heldout_action_mean_regret"],
            100.0 * summary["domains"][domain]["pure_utility_auc_gap"],
            100.0 * summary["domains"][domain]["hybrid_auc_gap"],
        ]
        axes[1, 1].bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            label=domain,
        )
    axes[1, 1].axhline(2.0, color="#E45756", linestyle="--", linewidth=1.2, label="2% AUC gate")
    axes[1, 1].axhline(5.0, color="#777777", linestyle=":", linewidth=1.2, label="5% regret gate")
    axes[1, 1].set_xticks(positions, labels=metric_names, rotation=18)
    axes[1, 1].set_ylabel("relative gap / regret (%)")
    axes[1, 1].set_title("Continuous utility generalization")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.suptitle("Target-aware wave interrogation on untouched systems", fontsize=14)
    figure.tight_layout()
    figure.savefig(output / "active_wave_interrogation_m1.png", dpi=240)
    figure.savefig(output / "active_wave_interrogation_m1.pdf")
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--utility", type=Path, required=True)
    parser.add_argument("--coordinate-utility", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.baseline,
                args.hybrid,
                args.utility,
                args.coordinate_utility,
                args.population,
                args.ood,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
