"""Evaluate one frozen validation selection over multiple prompt-order seeds."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .probabilistic_response_operator import (
    BatchedQueryEvaluator,
    LatentConfig,
    fit_probabilistic_subspaces,
    paired_summary,
)

CONFIG_PATTERN = re.compile(
    r"(?P<prior>scalar_ridge|spectrum_prior)_r(?P<rank>\d+)_l(?P<strength>.+)"
)


def parse_latent_config(name: str) -> LatentConfig:
    match = CONFIG_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"cannot parse latent config {name!r}")
    prior = "isotropic" if match.group("prior") == "scalar_ridge" else "spectrum"
    return LatentConfig(
        rank=int(match.group("rank")),
        strength=float(match.group("strength")),
        prior=prior,
    )


def evaluate_domain(
    domain: str,
    dataset: Path,
    selection_path: Path,
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    repeats: int,
    device: str,
) -> dict:
    archive = load_cross_domain_archive(dataset)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    maximum_rank = max(
        parse_latent_config(value).rank
        for choices in selection.values()
        for value in choices.values()
    )
    subspaces = fit_probabilistic_subspaces(archive, maximum_rank)
    result: dict[str, dict] = {}
    for budget in budgets:
        scalar_config = parse_latent_config(selection[str(budget)]["scalar"])
        spectrum_config = parse_latent_config(selection[str(budget)]["spectrum"])
        split_result = {}
        for split in ("test", "ood"):
            if not np.any(archive.split == split):
                continue
            scalar_by_seed = []
            spectrum_by_seed = []
            for seed in seeds:
                evaluator = BatchedQueryEvaluator(
                    archive, subspaces, split, budget, repeats, seed, device
                )
                scalar_by_seed.append(evaluator.evaluate(scalar_config))
                spectrum_by_seed.append(evaluator.evaluate(spectrum_config))
            scalar = np.stack(scalar_by_seed)
            spectrum = np.stack(spectrum_by_seed)
            summary = paired_summary(
                scalar.mean(axis=0), spectrum.mean(axis=0), seeds[0] + budget
            )
            summary["baseline_seed_means"] = scalar.mean(axis=1).tolist()
            summary["candidate_seed_means"] = spectrum.mean(axis=1).tolist()
            summary["baseline_seed_std"] = float(scalar.mean(axis=1).std(ddof=1))
            summary["candidate_seed_std"] = float(spectrum.mean(axis=1).std(ddof=1))
            split_result[split] = summary
        result[str(budget)] = {
            "scalar_config": scalar_config.name,
            "spectrum_config": spectrum_config.name,
            **split_result,
        }
    full_budget = max(budgets)
    full_scalar = result[str(full_budget)]["test"]["baseline_mean"]
    equivalent = next(
        (
            budget
            for budget in budgets
            if result[str(budget)]["test"]["candidate_mean"] <= full_scalar
        ),
        None,
    )
    return {
        "domain": domain,
        "selection_path": str(selection_path.resolve()),
        "selection_policy": "one frozen validation protocol; no retuning across test seeds",
        "seeds": list(seeds),
        "repeats_per_seed": repeats,
        "budgets": result,
        "measurement_efficiency": {
            "full_budget": full_budget,
            "full_scalar_error": full_scalar,
            "equivalent_spectrum_budget": equivalent,
            "probe_saving": None if equivalent is None else 1.0 - equivalent / full_budget,
        },
    }


def write_outputs(result: dict[str, dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for domain, domain_result in result.items():
        for budget, values in domain_result["budgets"].items():
            for split in ("test", "ood"):
                if split not in values:
                    continue
                metric = values[split]
                rows.append(
                    {
                        "domain": domain,
                        "budget": int(budget),
                        "split": split,
                        "baseline_rel_l2": metric["baseline_mean"],
                        "spectrum_prior_rel_l2": metric["candidate_mean"],
                        "relative_gain": metric["ratio_of_means_gain"],
                        "bootstrap_95_low": metric["bootstrap_95_low"],
                        "bootstrap_95_high": metric["bootstrap_95_high"],
                        "system_win_rate": metric["system_win_rate"],
                    }
                )
    with (output / "main_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, len(result), figsize=(5.0 * len(result), 4.0))
    if len(result) == 1:
        axes = [axes]
    for axis, (domain, domain_result) in zip(axes, result.items(), strict=True):
        budgets = [int(value) for value in domain_result["budgets"]]
        scalar = [
            domain_result["budgets"][str(value)]["test"]["baseline_mean"]
            for value in budgets
        ]
        spectrum = [
            domain_result["budgets"][str(value)]["test"]["candidate_mean"]
            for value in budgets
        ]
        axis.plot(budgets, scalar, "o--", label="scalar ridge")
        axis.plot(budgets, spectrum, "s-", label="latent spectrum prior")
        axis.set_title(domain)
        axis.set_xlabel("Prompt source budget K")
        axis.set_ylabel("Raw complex relative L2")
        axis.set_xticks(budgets)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "multidomain_budget_curve.png", dpi=220)
    figure.savefig(output / "multidomain_budget_curve.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="20260924,20260925,20260926")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    result = {}
    for domain, values in payload["domains"].items():
        result[domain] = evaluate_domain(
            domain,
            Path(values["dataset"]),
            args.selection_root / domain / "summary.json",
            tuple(int(value) for value in values["budgets"]),
            seeds,
            int(payload["test_repeats"]),
            args.device,
        )
    write_outputs(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
