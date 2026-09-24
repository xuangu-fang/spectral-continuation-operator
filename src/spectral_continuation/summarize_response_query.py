"""Aggregate formal response-query runs across seeds and physical systems."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    evaluations = [json.loads((run / "evaluation.json").read_text()) for run in args.runs]
    model_rows = [_jsonl(run / "model_rows.jsonl") for run in args.runs]
    baseline_rows = _jsonl(args.runs[0] / "baseline_rows.jsonl")
    budgets = sorted(int(value) for value in evaluations[0]["calibrated_reciprocity"])
    generator = np.random.default_rng(args.seed)
    rows = []
    for budget in budgets:
        seed_means = np.asarray(
            [evaluation["model"]["q2"][str(budget)]["mean"] for evaluation in evaluations]
        )
        baseline_mean = evaluations[0]["calibrated_reciprocity"][str(budget)]["mean"]
        systems = sorted({row["system"] for row in baseline_rows if row["budget"] == budget})
        model_system = []
        baseline_system = []
        for system in systems:
            model_system.append(
                np.mean(
                    [
                        np.mean(
                            [
                                row["relative_l2"]
                                for row in seed_rows
                                if row["budget"] == budget and row["system"] == system
                            ]
                        )
                        for seed_rows in model_rows
                    ]
                )
            )
            baseline_system.append(
                np.mean(
                    [
                        row["relative_l2"]
                        for row in baseline_rows
                        if row["budget"] == budget and row["system"] == system
                    ]
                )
            )
        model_system = np.asarray(model_system)
        baseline_system = np.asarray(baseline_system)
        bootstrap = []
        for _ in range(10000):
            indices = generator.integers(len(systems), size=len(systems))
            bootstrap.append(
                1.0 - model_system[indices].mean() / baseline_system[indices].mean()
            )
        rows.append(
            {
                "budget": budget,
                "model_mean": float(seed_means.mean()),
                "model_seed_std": float(seed_means.std(ddof=1)),
                "calibrated_reciprocity": float(baseline_mean),
                "relative_improvement": float(1.0 - seed_means.mean() / baseline_mean),
                "bootstrap_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "formal_budget_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"seeds": len(args.runs), "systems": 24, "budgets": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    x = np.asarray(budgets)
    model_mean = np.asarray([row["model_mean"] for row in rows])
    model_std = np.asarray([row["model_seed_std"] for row in rows])
    baseline = np.asarray([row["calibrated_reciprocity"] for row in rows])
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(x, baseline, marker="o", color="#6b7280", label="Calibrated reciprocity")
    axis.plot(x, model_mean, marker="o", color="#0f766e", label="Ours (3 seeds)")
    axis.fill_between(x, model_mean - model_std, model_mean + model_std, color="#0f766e", alpha=0.2)
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, labels=[str(value) for value in x])
    axis.set_xlabel("Context probes K")
    axis.set_ylabel("Raw complex relative L2 ↓")
    axis.set_title("Held-out source response query on unseen systems")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "formal_budget_scaling.png", dpi=220)
    fig.savefig(args.output / "formal_budget_scaling.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
