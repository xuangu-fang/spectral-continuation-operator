"""Posterior-memory extension of the global Bayesian response operator.

The same analytic prompt posterior embeds both a query system and every train
system under the observed coordinates.  Neighbors in this posterior metric
form an empirical-Bayes local prior over complete response operators.  A
validation-selected convex shrinkage combines the parametric posterior mean
with the local memory mean; the linear Gaussian operator is alpha=0.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
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
from .probabilistic_response_operator import paired_summary


@dataclass(frozen=True)
class MemoryConfig:
    latent_dimensions: int
    neighbors: int
    temperature: float
    blend: float
    bank_mode: str = "posterior"

    @property
    def name(self) -> str:
        base = (
            f"memory_d{self.latent_dimensions}_k{self.neighbors}_"
            f"t{self.temperature:g}_a{self.blend:g}"
        )
        return base if self.bank_mode == "posterior" else f"{base}_b{self.bank_mode}"


def _penalty(evaluator: GlobalQ3Evaluator, config: GlobalConfig) -> torch.Tensor:
    variance = evaluator.variance[: config.rank]
    if config.prior == "isotropic":
        return torch.full_like(variance, config.strength)
    reference = torch.quantile(variance, 0.5)
    return config.strength * reference / variance.clamp_min(1e-30)


@torch.inference_mode()
def evaluate_memory_configs(
    evaluator: GlobalQ3Evaluator,
    operator: GlobalConfig,
    configs: tuple[MemoryConfig, ...],
    latent_threshold: float | None = None,
    memory_indices: np.ndarray | torch.Tensor | None = None,
    memory_encoded: torch.Tensor | None = None,
    memory_raw: torch.Tensor | None = None,
    memory_weights: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Return system-macro errors for many memory settings in one pass."""

    basis = evaluator.basis[:, : operator.rank]
    variance = evaluator.variance[: operator.rank]
    penalty = _penalty(evaluator, operator)
    global_prediction = evaluator.predict(operator)
    latent_energy = evaluator.latent_energy(operator)
    totals = {
        config.name: torch.zeros(len(evaluator.raw), device=evaluator.raw.device)
        for config in configs
    }
    total_global = torch.zeros(len(evaluator.raw), device=evaluator.raw.device)
    maximum_neighbors = max(config.neighbors for config in configs)
    event_count = len(evaluator.observed)
    if memory_encoded is not None or memory_raw is not None:
        if memory_encoded is None or memory_raw is None:
            raise ValueError("memory_encoded and memory_raw must be provided together")
        memory_train_encoded = memory_encoded.to(evaluator.raw.device)
        memory_train_raw = memory_raw.to(evaluator.raw.device)
    elif memory_indices is None:
        memory_indices_tensor = torch.arange(
            len(evaluator.train_raw), device=evaluator.raw.device
        )
        memory_train_encoded = evaluator.train_encoded[memory_indices_tensor]
        memory_train_raw = evaluator.train_raw[memory_indices_tensor]
    else:
        memory_indices_tensor = torch.as_tensor(
            memory_indices, dtype=torch.long, device=evaluator.raw.device
        )
        memory_train_encoded = evaluator.train_encoded[memory_indices_tensor]
        memory_train_raw = evaluator.train_raw[memory_indices_tensor]
    if memory_weights is not None:
        memory_weights = memory_weights.to(evaluator.raw.device).clamp_min(1e-12)
    full_bank_latent = memory_train_encoded @ basis.conj()
    for event in range(event_count):
        observed = evaluator.observed[event]
        target = evaluator.target[event]
        query_latent, _ = solve_complex_posterior(
            basis[observed], evaluator.context[event], penalty
        )
        bank_latent, _ = solve_complex_posterior(
            basis[observed], memory_train_encoded[:, observed], penalty
        )
        truth = evaluator.truth[event]
        denominator = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)
        base = global_prediction[event]
        mean = evaluator.center[target][None]
        use_operator = None
        if latent_threshold is not None:
            use_operator = latent_energy[event] <= latent_threshold
            base = torch.where(use_operator[:, None], base, mean)
        total_global += torch.linalg.vector_norm(base - truth, dim=-1) / denominator
        neighbor_data: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for dimension, bank_mode in {
            (config.latent_dimensions, config.bank_mode) for config in configs
        }:
            scale = torch.sqrt(variance[:dimension]).clamp_min(1e-15)
            query_metric = query_latent[:, :dimension] / scale
            selected_bank = (
                bank_latent if bank_mode == "posterior" else full_bank_latent
            )
            bank_metric = selected_bank[:, :dimension] / scale
            distance = (
                query_metric.abs().square().sum(dim=1, keepdim=True)
                + bank_metric.abs().square().sum(dim=1)[None]
                - 2.0 * torch.real(query_metric @ bank_metric.conj().T)
            )
            values, order = torch.topk(distance, maximum_neighbors, largest=False)
            selected_weights = None if memory_weights is None else memory_weights[order]
            neighbor_data[(dimension, bank_mode)] = (
                values,
                memory_train_raw[order][:, :, target],
                selected_weights,
            )
        ungated_base = global_prediction[event]
        for config in configs:
            distance, candidates, selected_weights = neighbor_data[
                (config.latent_dimensions, config.bank_mode)
            ]
            distance = distance[:, : config.neighbors]
            candidates = candidates[:, : config.neighbors]
            if selected_weights is not None:
                selected_weights = selected_weights[:, : config.neighbors]
            if config.temperature <= 0.0 or config.neighbors == 1:
                if selected_weights is None:
                    weights = torch.full_like(distance, 1.0 / config.neighbors)
                else:
                    weights = selected_weights / selected_weights.sum(dim=1, keepdim=True)
            else:
                relative = (distance - distance[:, :1]).clamp_min(0.0)
                local_scale = relative[:, -1:].clamp_min(1e-8)
                logits = -relative / (config.temperature * local_scale)
                if selected_weights is not None:
                    logits = logits + torch.log(selected_weights)
                weights = torch.softmax(logits, dim=1)
            memory = (weights[:, :, None] * candidates).sum(dim=1)
            prediction = ungated_base + config.blend * (memory - ungated_base)
            if use_operator is not None:
                prediction = torch.where(use_operator[:, None], prediction, mean)
            totals[config.name] += (
                torch.linalg.vector_norm(prediction - truth, dim=-1) / denominator
            )
    result = {
        name: (value / event_count).cpu().numpy().astype(np.float64)
        for name, value in totals.items()
    }
    result["global_operator"] = (
        total_global / event_count
    ).cpu().numpy().astype(np.float64)
    return result


@torch.inference_mode()
def predict_memory_config_batched(
    evaluator: GlobalQ3Evaluator,
    operator: GlobalConfig,
    config: MemoryConfig,
    latent_threshold: float | None = None,
    memory_indices: np.ndarray | torch.Tensor | None = None,
    memory_encoded: torch.Tensor | None = None,
    memory_raw: torch.Tensor | None = None,
    memory_weights: torch.Tensor | None = None,
    memory_coefficients: torch.Tensor | None = None,
    observed: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return vectorized predictions for one memory operator and prompt mask."""

    basis = evaluator.basis[:, : operator.rank]
    variance = evaluator.variance[: operator.rank]
    penalty = _penalty(evaluator, operator)
    observed = evaluator.observed if observed is None else observed
    context = evaluator.context if context is None else context
    target = evaluator.target if target is None else target
    observed_basis = basis[observed]
    query_latent, _ = solve_complex_posterior(
        observed_basis, context, penalty
    )
    if memory_coefficients is not None:
        if memory_encoded is not None or memory_raw is not None or memory_indices is not None:
            raise ValueError("memory_coefficients cannot be combined with another bank")
        coefficients = memory_coefficients.to(evaluator.raw.device)
        coefficient_rank = coefficients.shape[-1]
        memory_train_encoded = None
        memory_train_raw = None
    elif memory_encoded is not None or memory_raw is not None:
        if memory_encoded is None or memory_raw is None:
            raise ValueError("memory_encoded and memory_raw must be provided together")
        memory_train_encoded = memory_encoded.to(evaluator.raw.device)
        memory_train_raw = memory_raw.to(evaluator.raw.device)
    elif memory_indices is None:
        memory_indices_tensor = torch.arange(
            len(evaluator.train_raw), device=evaluator.raw.device
        )
        memory_train_encoded = evaluator.train_encoded[memory_indices_tensor]
        memory_train_raw = evaluator.train_raw[memory_indices_tensor]
    else:
        memory_indices_tensor = torch.as_tensor(
            memory_indices, dtype=torch.long, device=evaluator.raw.device
        )
        memory_train_encoded = evaluator.train_encoded[memory_indices_tensor]
        memory_train_raw = evaluator.train_raw[memory_indices_tensor]
    if memory_weights is not None:
        memory_weights = memory_weights.to(evaluator.raw.device).clamp_min(1e-12)
    if config.bank_mode == "posterior":
        if memory_coefficients is None:
            bank_context = memory_train_encoded[:, observed].permute(1, 0, 2)
        else:
            bank_context = torch.einsum(
                "nl,eol->eno", coefficients, observed_basis[..., :coefficient_rank]
            )
        bank_latent, _ = solve_complex_posterior(
            observed_basis, bank_context, penalty
        )
    else:
        if memory_coefficients is None:
            full_latent = memory_train_encoded @ basis.conj()
        else:
            full_latent = torch.zeros(
                (len(coefficients), operator.rank),
                dtype=coefficients.dtype,
                device=coefficients.device,
            )
            full_latent[:, :coefficient_rank] = coefficients
        bank_latent = full_latent[None].expand(len(observed), -1, -1)

    dimension = config.latent_dimensions
    scale = torch.sqrt(variance[:dimension]).clamp_min(1e-15)
    query_metric = query_latent[..., :dimension] / scale
    bank_metric = bank_latent[..., :dimension] / scale
    distance = (
        query_metric.abs().square().sum(dim=-1, keepdim=True)
        + bank_metric.abs().square().sum(dim=-1)[:, None]
        - 2.0 * torch.real(
            torch.einsum("ebd,end->ebn", query_metric, bank_metric.conj())
        )
    )
    distance, order = torch.topk(
        distance, config.neighbors, largest=False, dim=-1
    )
    selected_weights = None
    if memory_weights is not None:
        selected_weights = memory_weights[order]
    if config.temperature <= 0.0 or config.neighbors == 1:
        if selected_weights is None:
            weights = torch.full_like(distance, 1.0 / config.neighbors)
        else:
            weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
    else:
        relative = (distance - distance[..., :1]).clamp_min(0.0)
        local_scale = relative[..., -1:].clamp_min(1e-8)
        logits = -relative / (config.temperature * local_scale)
        if selected_weights is not None:
            logits = logits + torch.log(selected_weights)
        weights = torch.softmax(logits, dim=-1)
    if memory_coefficients is None:
        train_targets = memory_train_raw[:, target].permute(1, 0, 2)
    else:
        normalized_targets = torch.einsum(
            "nl,erl->enr", coefficients, basis[target][..., :coefficient_rank]
        )
        train_targets = (
            evaluator.center[target][:, None]
            + evaluator.scale[target][:, None] * normalized_targets
        )
    event = torch.arange(len(order), device=order.device)[:, None, None]
    candidates = train_targets[event, order]
    memory = (weights[..., None] * candidates).sum(dim=2)
    global_prediction = evaluator.predict(operator, observed, context, target)
    prediction = global_prediction + config.blend * (memory - global_prediction)
    if latent_threshold is not None:
        latent_energy = (
            query_latent.abs().square()
            / variance.clamp_min(1e-30)[None, None]
        ).mean(dim=-1)
        mean = evaluator.center[target][:, None]
        prediction = torch.where(
            (latent_energy <= latent_threshold)[..., None], prediction, mean
        )
    return prediction


@torch.inference_mode()
def evaluate_memory_config_batched(
    evaluator: GlobalQ3Evaluator,
    operator: GlobalConfig,
    config: MemoryConfig,
    latent_threshold: float | None = None,
    memory_indices: np.ndarray | torch.Tensor | None = None,
    memory_encoded: torch.Tensor | None = None,
    memory_raw: torch.Tensor | None = None,
    memory_weights: torch.Tensor | None = None,
    memory_coefficients: torch.Tensor | None = None,
) -> np.ndarray:
    """Vectorized deployment path for one validation-selected memory operator."""

    prediction = predict_memory_config_batched(
        evaluator,
        operator,
        config,
        latent_threshold,
        memory_indices,
        memory_encoded,
        memory_raw,
        memory_weights,
        memory_coefficients,
    )
    error = torch.linalg.vector_norm(
        prediction - evaluator.truth, dim=-1
    ) / torch.linalg.vector_norm(evaluator.truth, dim=-1).clamp_min(1e-12)
    return error.mean(dim=0).cpu().numpy().astype(np.float64)


def run(config_path: Path, output: Path, device: str) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in payload["test_seeds"])
    validation_rows = []
    result = {}
    output.mkdir(parents=True, exist_ok=True)
    for domain, setting in payload["domains"].items():
        archive = load_cross_domain_archive(Path(setting["dataset"]))
        subspaces = {}
        candidates = tuple(
            MemoryConfig(
                int(dimension),
                int(neighbors),
                float(temperature),
                float(blend),
                str(bank_mode),
            )
            for dimension in setting["latent_dimensions"]
            for neighbors in setting["neighbors"]
            for temperature in setting["temperatures"]
            for blend in setting["blends"]
            for bank_mode in setting.get("bank_modes", ["posterior"])
        )
        domain_result = {}
        for budget in payload["budgets"]:
            operator_setting = {
                **setting,
                **setting.get("operator_by_budget", {}).get(str(budget), {}),
            }
            operator = GlobalConfig(
                rank=int(operator_setting["rank"]),
                strength=float(operator_setting["strength"]),
                prior="spectrum",
                normalization=str(operator_setting["normalization"]),
            )
            subspace_key = (operator.rank, operator.normalization)
            if subspace_key not in subspaces:
                subspaces[subspace_key] = fit_global_subspace(
                    archive, operator.rank, operator.normalization, device
                )
            subspace = subspaces[subspace_key]
            validation = GlobalQ3Evaluator(
                archive,
                subspace,
                "validation",
                int(budget),
                int(payload["validation_repeats"]),
                int(payload["selection_seed"]),
                device,
            )
            validation_values = evaluate_memory_configs(
                validation, operator, candidates
            )
            selected = min(
                candidates,
                key=lambda candidate: float(validation_values[candidate.name].mean()),
            )
            baseline_method = min(
                ("global_operator", "system_retrieval"),
                key=lambda method: (
                    float(validation_values["global_operator"].mean())
                    if method == "global_operator"
                    else float(validation.evaluate_system_retrieval().mean())
                ),
            )
            threshold = float(validation.latent_energy(operator).max().cpu())
            for candidate in candidates:
                validation_rows.append(
                    {
                        "domain": domain,
                        "budget": int(budget),
                        "method": candidate.name,
                        "raw_complex_rel_l2": float(
                            validation_values[candidate.name].mean()
                        ),
                        "selected": candidate == selected,
                    }
                )
            split_result = {}
            for split in ("test", "ood"):
                if not np.any(archive.split == split):
                    continue
                candidate_by_seed = []
                baseline_by_seed = []
                global_by_seed = []
                for seed in seeds:
                    evaluator = GlobalQ3Evaluator(
                        archive,
                        subspace,
                        split,
                        int(budget),
                        int(payload["test_repeats"]),
                        seed,
                        device,
                    )
                    values = evaluate_memory_configs(
                        evaluator,
                        operator,
                        (selected,),
                        threshold if setting.get("use_latent_gate", False) else None,
                    )
                    candidate_by_seed.append(values[selected.name])
                    global_by_seed.append(values["global_operator"])
                    if baseline_method == "system_retrieval":
                        baseline_by_seed.append(evaluator.evaluate_system_retrieval())
                    else:
                        baseline_by_seed.append(values["global_operator"])
                candidate_values = np.stack(candidate_by_seed)
                baseline_values = np.stack(baseline_by_seed)
                global_values = np.stack(global_by_seed)
                statistics = paired_summary(
                    baseline_values.mean(axis=0),
                    candidate_values.mean(axis=0),
                    seeds[0] + int(budget),
                )
                statistics.update(
                    {
                        "candidate_seed_means": candidate_values.mean(axis=1).tolist(),
                        "baseline_seed_means": baseline_values.mean(axis=1).tolist(),
                        "global_seed_means": global_values.mean(axis=1).tolist(),
                    }
                )
                split_result[split] = statistics
                np.savez_compressed(
                    output / f"{domain.lower()}_k{budget}_{split}_systems.npz",
                    candidate=candidate_values,
                    baseline=baseline_values,
                    global_operator=global_values,
                )
            domain_result[str(budget)] = {
                "operator": operator.name,
                "memory": selected.name,
                "validation_baseline": baseline_method,
                "latent_gate_threshold": threshold,
                **split_result,
            }
        result[domain] = domain_result
    with (output / "validation_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=validation_rows[0].keys())
        writer.writeheader()
        writer.writerows(validation_rows)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
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
