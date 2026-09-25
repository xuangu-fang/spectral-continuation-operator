"""Prompt-only leave-one-out calibration of latent response priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .probabilistic_response_operator import (
    BatchedQueryEvaluator,
    fit_probabilistic_subspaces,
    paired_summary,
)
from .summarize_latent_prior_multiseed import parse_latent_config


def run_domain(
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
    results = {}
    for budget in budgets:
        if budget < 2:
            continue
        baseline_config = parse_latent_config(selection[str(budget)]["scalar"])
        candidate_config = parse_latent_config(selection[str(budget)]["spectrum"])
        budget_result = {}
        for split in ("test", "ood"):
            if not np.any(archive.split == split):
                continue
            baseline_by_seed = []
            fixed_by_seed = []
            adaptive_by_seed = []
            selection_rates = []
            for seed in seeds:
                evaluator = BatchedQueryEvaluator(
                    archive, subspaces, split, budget, repeats, seed, device
                )
                baseline_by_seed.append(evaluator.evaluate(baseline_config))
                fixed_by_seed.append(evaluator.evaluate(candidate_config))
                adaptive, selection_rate = evaluator.evaluate_adaptive(
                    baseline_config, candidate_config
                )
                adaptive_by_seed.append(adaptive)
                selection_rates.append(selection_rate)
            baseline = np.stack(baseline_by_seed).mean(axis=0)
            fixed = np.stack(fixed_by_seed).mean(axis=0)
            adaptive = np.stack(adaptive_by_seed).mean(axis=0)
            budget_result[split] = {
                "fixed_spectrum_vs_scalar": paired_summary(
                    baseline, fixed, seeds[0] + budget
                ),
                "adaptive_vs_scalar": paired_summary(
                    baseline, adaptive, seeds[0] + 100 + budget
                ),
                "adaptive_vs_fixed_spectrum": paired_summary(
                    fixed, adaptive, seeds[0] + 200 + budget
                ),
                "spectrum_selection_rate": float(np.mean(selection_rates)),
                "spectrum_selection_rate_by_seed": selection_rates,
            }
        results[str(budget)] = {
            "scalar_config": baseline_config.name,
            "spectrum_config": candidate_config.name,
            **budget_result,
        }
    return {
        "domain": domain,
        "policy": (
            "per episode/system/frequency choose scalar or spectrum prior by "
            "leave-one-prompt-source-out error; target query is never read"
        ),
        "seeds": list(seeds),
        "repeats": repeats,
        "budgets": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domains", default="Deepwave,Ocean")
    parser.add_argument("--seeds", default="20260924,20260925,20260926")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    selected_domains = set(args.domains.split(","))
    result = {}
    for domain, values in payload["domains"].items():
        if domain not in selected_domains:
            continue
        result[domain] = run_domain(
            domain,
            Path(values["dataset"]),
            args.selection_root / domain / "summary.json",
            tuple(int(value) for value in values["budgets"]),
            seeds,
            int(payload["test_repeats"]),
            args.device,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
