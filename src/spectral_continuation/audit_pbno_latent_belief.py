"""Audit whether PBNO prompt beliefs justify a conditional latent flow.

The audit retrieves training systems in a prompt-posterior latent metric.  It
compares the deployable neighbor mean with a target-reading best-of-N oracle.
A flow is only justified when the oracle leaves a substantial gap over the
strongest deterministic predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import (
    GlobalConfig,
    GlobalQ3Evaluator,
    fit_global_subspace,
    solve_complex_posterior,
)


@torch.inference_mode()
def evaluate_split(
    evaluator: GlobalQ3Evaluator,
    config: GlobalConfig,
    latent_dimensions: int,
    neighbor_count: int,
) -> dict:
    basis = evaluator.basis[:, : config.rank]
    variance = evaluator.variance[: config.rank]
    reference = torch.quantile(variance, 0.5)
    penalty = config.strength * reference / variance.clamp_min(1e-30)
    scale = torch.sqrt(variance[:latent_dimensions]).clamp_min(1e-15)
    names = ("global_mean", "nearest_memory", "memory_mean", "best_of_memory")
    errors = {
        name: torch.zeros(len(evaluator.raw), device=evaluator.raw.device)
        for name in names
    }
    nearest_context_distance = []
    candidate_target_spread = []
    event_count = len(evaluator.observed)
    for event in range(event_count):
        observed = evaluator.observed[event]
        target = evaluator.target[event]
        query_latent, _ = solve_complex_posterior(
            basis[observed], evaluator.context[event], penalty
        )
        bank_latent, _ = solve_complex_posterior(
            basis[observed], evaluator.train_encoded[:, observed], penalty
        )
        query_metric = query_latent[:, :latent_dimensions] / scale
        bank_metric = bank_latent[:, :latent_dimensions] / scale
        distance = (
            query_metric.abs().square().sum(dim=1, keepdim=True)
            + bank_metric.abs().square().sum(dim=1)[None]
            - 2.0 * torch.real(query_metric @ bank_metric.conj().T)
        )
        values, indices = torch.topk(distance, neighbor_count, largest=False)
        candidates = evaluator.train_raw[indices][:, :, target]
        truth = evaluator.truth[event]
        denominator = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)
        global_prediction = evaluator.predict(
            config,
            evaluator.observed[event : event + 1],
            evaluator.context[event : event + 1],
            evaluator.target[event : event + 1],
        )[0]
        candidate_error = (
            torch.linalg.vector_norm(candidates - truth[:, None], dim=-1)
            / denominator[:, None]
        )
        errors["global_mean"] += (
            torch.linalg.vector_norm(global_prediction - truth, dim=-1) / denominator
        )
        errors["nearest_memory"] += candidate_error[:, 0]
        errors["memory_mean"] += (
            torch.linalg.vector_norm(candidates.mean(dim=1) - truth, dim=-1)
            / denominator
        )
        errors["best_of_memory"] += candidate_error.min(dim=1).values
        nearest_context_distance.append(values[:, 0].clamp_min(0.0).sqrt().cpu())
        candidate_target_spread.append(
            (
                torch.linalg.vector_norm(
                    candidates - candidates.mean(dim=1, keepdim=True), dim=-1
                ).mean(dim=1)
                / denominator
            ).cpu()
        )
    system_errors = {name: value / event_count for name, value in errors.items()}
    means = {name: float(value.mean().cpu()) for name, value in system_errors.items()}
    deterministic = min(means["global_mean"], means["memory_mean"])
    oracle_gap = 1.0 - means["best_of_memory"] / deterministic
    context_distance = torch.cat(nearest_context_distance).numpy()
    target_spread = torch.cat(candidate_target_spread).numpy()
    return {
        **means,
        "strongest_deterministic": deterministic,
        "best_of_memory_relative_gap": oracle_gap,
        "flow_gate_threshold": 0.15,
        "flow_gate_passed": bool(oracle_gap >= 0.15),
        "nearest_context_vs_target_spread_correlation": float(
            np.corrcoef(context_distance, target_spread)[0, 1]
        ),
        "mean_candidate_target_spread": float(target_spread.mean()),
    }


def run(config_path: Path, output: Path, device: str) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    result = {}
    for domain, setting in payload["domains"].items():
        archive = load_cross_domain_archive(Path(setting["dataset"]))
        operator = GlobalConfig(
            rank=int(setting["rank"]),
            strength=float(setting["strength"]),
            prior="spectrum",
            normalization=str(setting["normalization"]),
        )
        subspace = fit_global_subspace(
            archive, operator.rank, operator.normalization, device
        )
        splits = {}
        for split, repeats in (
            ("validation", int(payload["validation_repeats"])),
            ("test", int(payload["test_repeats"])),
        ):
            evaluator = GlobalQ3Evaluator(
                archive,
                subspace,
                split,
                int(setting["budget"]),
                repeats,
                int(payload["seed"]),
                device,
            )
            splits[split] = evaluate_split(
                evaluator,
                operator,
                int(setting["latent_dimensions"]),
                int(setting["neighbor_count"]),
            )
        result[domain] = {
            "task": "strict Q3 latent conditional-belief audit",
            "budget": int(setting["budget"]),
            "latent_dimensions": int(setting["latent_dimensions"]),
            "neighbor_count": int(setting["neighbor_count"]),
            "splits": splits,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
