"""Aggregate three-seed residual-query results at the physical-system level."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _rows(root: Path) -> list[dict]:
    with (root / "test_system_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bootstrap(baseline: np.ndarray, model: np.ndarray, seed: int, draws: int = 20_000):
    improvement = (baseline - model) / np.maximum(baseline, 1e-12)
    generator = np.random.default_rng(seed)
    samples = np.empty(draws)
    for start in range(0, draws, 1000):
        count = min(1000, draws - start)
        index = generator.integers(0, len(model), size=(count, len(model)))
        samples[start : start + count] = improvement[index].mean(axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "mean_relative_improvement": float(improvement.mean()),
        "bootstrap_95_low": float(lower),
        "bootstrap_95_high": float(upper),
        "positive_system_fraction": float(np.mean(model < baseline)),
    }


def summarize_domain(roots: list[Path], seed: int) -> dict:
    all_rows = [_rows(root) for root in roots]
    budgets = sorted({int(row["budget"]) for row in all_rows[0]})
    result = {}
    for budget in budgets:
        per_seed_model = []
        per_seed_baseline = []
        system_ids = None
        for rows in all_rows:
            selected = [row for row in rows if int(row["budget"]) == budget]
            selected.sort(key=lambda row: row["system_id"])
            ids = [row["system_id"] for row in selected]
            if system_ids is None:
                system_ids = ids
            elif ids != system_ids:
                raise ValueError("seed outputs do not share physical-system IDs")
            per_seed_model.append([float(row["model_rel_l2"]) for row in selected])
            per_seed_baseline.append([float(row["subspace_rel_l2"]) for row in selected])
        model = np.asarray(per_seed_model)
        baseline = np.asarray(per_seed_baseline)
        model_system = model.mean(axis=0)
        baseline_system = baseline.mean(axis=0)
        result[str(budget)] = {
            "model_seed_means": model.mean(axis=1).tolist(),
            "model_mean": float(model.mean(axis=1).mean()),
            "model_seed_std": float(model.mean(axis=1).std(ddof=0)),
            "subspace_mean": float(baseline.mean()),
            "paired_after_seed_average": _bootstrap(
                baseline_system, model_system, seed + budget
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    domains = {
        "openfwi": [root / f"openfwi_m1_seed{seed}" for seed in (20260924, 20260925, 20260926)],
        "deepwave": [root / f"deepwave_m1_seed{seed}" for seed in (20260924, 20260925, 20260926)],
    }
    summary = {
        domain: summarize_domain(roots, 20260924 + index * 100)
        for index, (domain, roots) in enumerate(domains.items())
    }
    (root / "m1_three_seed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for axis, (domain, values) in zip(axes, summary.items(), strict=True):
        budgets = sorted(int(value) for value in values)
        baseline = [values[str(value)]["subspace_mean"] for value in budgets]
        model = [values[str(value)]["model_mean"] for value in budgets]
        deviation = [values[str(value)]["model_seed_std"] for value in budgets]
        axis.plot(budgets, baseline, "o-", color="#6b7280", label="subspace")
        axis.errorbar(
            budgets, model, yerr=deviation, marker="o", color="#1b9e77", label="+ local residual"
        )
        axis.set_title(domain)
        axis.set_xlabel("context probes K")
        axis.set_ylabel("raw complex relative L2")
        axis.set_xticks(budgets)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "m1_three_seed_refinement.png", dpi=220)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
