"""Frozen split evaluation for the selected response-query checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .audit_response_query import _reciprocity_prediction
from .response_query import (
    MaskedResponseTransformer,
    ResponseNormalizer,
    complex_relative_l2,
    frozen_episodes,
    load_response_archive,
)
from .train_response_query import _encoded_table, evaluate


def paired_bootstrap(model_rows, baseline_rows, budget, seed=20260924, draws=10000):
    systems = sorted({row["system"] for row in model_rows if row["budget"] == budget})
    model_by_system = {
        system: np.mean(
            [row["relative_l2"] for row in model_rows if row["budget"] == budget and row["system"] == system]
        )
        for system in systems
    }
    baseline_by_system = {
        system: np.mean(
            [row["relative_l2"] for row in baseline_rows if row["budget"] == budget and row["system"] == system]
        )
        for system in systems
    }
    model = np.asarray([model_by_system[system] for system in systems])
    baseline = np.asarray([baseline_by_system[system] for system in systems])
    improvement = 1.0 - model.mean() / baseline.mean()
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        indices = generator.integers(len(systems), size=len(systems))
        samples.append(1.0 - model[indices].mean() / baseline[indices].mean())
    return {
        "systems": len(systems),
        "model_mean": float(model.mean()),
        "baseline_mean": float(baseline.mean()),
        "relative_improvement": float(improvement),
        "bootstrap_95": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--normalization-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--episodes-per-system", type=int, default=8)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    receivers = int(train_args["receivers"])
    archive = load_response_archive(args.dataset, receivers, train_args["receiver_mode"])
    normalization_archive = (
        load_response_archive(
            args.normalization_dataset, receivers, train_args["receiver_mode"]
        )
        if args.normalization_dataset is not None
        else archive
    )
    normalizer = ResponseNormalizer(normalization_archive)
    encoded = _encoded_table(archive, normalizer)
    episodes = frozen_episodes(
        archive,
        split=args.split,
        regimes=tuple(str(train_args["regimes"]).split(",")),
        episodes_per_system=args.episodes_per_system,
        seed=20260924,
    )
    model = MaskedResponseTransformer(
        receivers,
        int(train_args["hidden"]),
        int(train_args["transformer_layers"]),
        int(train_args["transformer_heads"]),
    ).cuda()
    model.load_state_dict(checkpoint["model"])
    budgets = tuple(int(value) for value in str(train_args["budgets"]).split(","))
    summary, model_rows = evaluate(
        model,
        archive,
        normalizer,
        encoded,
        episodes,
        budgets,
        torch.device("cuda"),
        reciprocity_residual=bool(train_args["reciprocity_residual"]),
        calibrated_reciprocity=bool(train_args["calibrated_reciprocity"]),
    )
    baseline_rows = []
    baseline_summary = {}
    for budget in budgets:
        values = []
        for episode in episodes:
            prediction = _reciprocity_prediction(
                archive, normalizer, episode, budget, calibrated=True
            )
            truth = archive.responses[
                episode.system, episode.target_source, episode.target_frequency
            ]
            value = float(complex_relative_l2(prediction, truth))
            values.append(value)
            baseline_rows.append(
                {"system": episode.system, "budget": budget, "relative_l2": value}
            )
        baseline_summary[str(budget)] = {
            "mean": float(np.mean(values)), "median": float(np.median(values))
        }
    paired = {
        str(budget): paired_bootstrap(model_rows, baseline_rows, budget)
        for budget in budgets
    }
    result = {
        "split": args.split,
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "model": summary,
        "calibrated_reciprocity": baseline_summary,
        "paired_vs_calibrated_reciprocity": paired,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
    with (args.output / "model_rows.jsonl").open("w") as handle:
        for row in model_rows:
            handle.write(json.dumps(row) + "\n")
    with (args.output / "baseline_rows.jsonl").open("w") as handle:
        for row in baseline_rows:
            handle.write(json.dumps(row) + "\n")

    regime = next(iter(summary))
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.plot(
        budgets,
        [baseline_summary[str(budget)]["mean"] for budget in budgets],
        marker="o",
        label="calibrated reciprocity",
    )
    axis.plot(
        budgets,
        [summary[regime][str(budget)]["mean"] for budget in budgets],
        marker="o",
        label="physics-routed query model",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, labels=[str(value) for value in budgets])
    axis.set_xlabel("Context probes K")
    axis.set_ylabel("Raw complex relative L2")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "budget_scaling.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
