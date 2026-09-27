"""Learn a query-conditioned metric for posterior operator retrieval.

The analytic complex posterior remains the system identifier.  Learning is
restricted to a positive Mahalanobis metric which decides which posterior
directions are relevant for a requested source--frequency query.  Training
uses leave-one-system-out retrieval inside the training split, so the model
cannot win by retrieving the query system itself.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cross_domain_response_data import CrossDomainResponseArchive, load_cross_domain_archive
from .global_bayesian_response_operator import (
    GlobalConfig,
    _q3_events,
    fit_global_subspace,
    solve_complex_posterior,
)
from .probabilistic_response_operator import paired_summary
from .train_prompted_bayesian_neural_operator import fourier_features, response_coordinates


@dataclass
class MetricState:
    archive: CrossDomainResponseArchive
    center: torch.Tensor
    scale: torch.Tensor
    basis: torch.Tensor
    variance: torch.Tensor
    penalty: torch.Tensor
    train_encoded: torch.Tensor
    train_raw: torch.Tensor
    memory_coefficients: torch.Tensor
    split_encoded: dict[str, torch.Tensor]
    split_raw: dict[str, torch.Tensor]
    coordinates: torch.Tensor
    source_count: int
    frequency_count: int
    receiver_count: int


@dataclass
class MetricEpisode:
    coefficient: torch.Tensor
    bank_coefficient: torch.Tensor
    target_basis: torch.Tensor
    target_center: torch.Tensor
    target_scale: torch.Tensor
    target_coordinate: torch.Tensor
    truth: torch.Tensor
    query_indices: torch.Tensor | None


class QueryConditionedPosteriorMetric(nn.Module):
    """Small positive metric over observable posterior coordinates."""

    def __init__(
        self,
        rank: int,
        initial_dimensions: int,
        metric_dimensions: int,
        variance: torch.Tensor,
        width: int,
        mode_width: int,
        tail_log_weight: float,
        residual_bound: float,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.initial_dimensions = initial_dimensions
        self.metric_dimensions = metric_dimensions
        self.residual_bound = residual_bound
        self.register_buffer("variance", variance.detach().float().clone())
        initial = torch.zeros(metric_dimensions)
        self.log_mode_weight = nn.Parameter(initial)
        self.mode_embedding = nn.Parameter(torch.zeros(metric_dimensions, mode_width))
        nn.init.normal_(self.mode_embedding, std=0.02)
        projection = torch.zeros(rank, metric_dimensions, dtype=torch.complex64)
        diagonal = min(initial_dimensions, metric_dimensions)
        projection[torch.arange(diagonal), torch.arange(diagonal)] = 1.0
        if metric_dimensions > diagonal:
            generator = torch.Generator().manual_seed(1729)
            projection[:, diagonal:] = float(np.exp(tail_log_weight / 2.0)) * torch.complex(
                torch.randn(rank, metric_dimensions - diagonal, generator=generator),
                torch.randn(rank, metric_dimensions - diagonal, generator=generator),
            ) / np.sqrt(2.0 * rank)
        self.projection_parts = nn.Parameter(torch.view_as_real(projection))
        query_features = 3 * 11
        self.query_encoder = nn.Sequential(
            nn.Linear(query_features, width),
            nn.SiLU(),
            nn.Linear(width, mode_width),
        )
        self.conditioner = nn.Sequential(
            nn.Linear(2 * mode_width + 3, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        self.log_temperature_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self, target_coordinate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_encoder(fourier_features(target_coordinate[:3][None]))[0]
        position = torch.linspace(-1.0, 1.0, self.metric_dimensions, device=query.device)
        variance = self.variance[: self.metric_dimensions]
        log_variance = torch.log(variance.clamp_min(1e-30))
        log_variance = (log_variance - log_variance.mean()) / log_variance.std().clamp_min(
            1e-6
        )
        base = torch.cat(
            (
                self.mode_embedding,
                query[None].expand(self.metric_dimensions, -1),
                position[:, None],
                log_variance[:, None],
                (
                    torch.arange(self.metric_dimensions, device=query.device)
                    < self.initial_dimensions
                )
                .float()[:, None],
            ),
            dim=-1,
        )
        residual = self.residual_bound * torch.tanh(self.conditioner(base)[:, 0])
        weight = torch.exp((self.log_mode_weight + residual).clamp(-14.0, 8.0))
        temperature_scale = torch.exp(self.log_temperature_scale.clamp(-2.0, 2.0))
        projection = torch.view_as_complex(self.projection_parts)
        return weight, temperature_scale, projection


def _penalty(variance: torch.Tensor, strength: float) -> torch.Tensor:
    reference = torch.quantile(variance, 0.5)
    return strength * reference / variance.clamp_min(1e-30)


def build_state(setting: dict, rank: int, value_rank: int, device: torch.device) -> MetricState:
    archive = load_cross_domain_archive(Path(setting["dataset"]))
    operator = GlobalConfig(
        rank=rank,
        strength=float(setting["strength"]),
        prior="spectrum",
        normalization=str(setting["normalization"]),
    )
    subspace = fit_global_subspace(archive, rank, operator.normalization, str(device))
    center = torch.as_tensor(subspace.center, device=device)
    scale = torch.as_tensor(subspace.scale, device=device)
    basis = torch.as_tensor(subspace.basis[:, :rank], device=device)
    variance = torch.as_tensor(subspace.variance[:rank], dtype=torch.float32, device=device)
    train = archive.responses[archive.split == "train"].reshape(
        int(np.sum(archive.split == "train")), -1
    )
    train_raw = torch.as_tensor(train, dtype=torch.complex64, device=device)
    train_encoded = (train_raw - center[None]) / scale[None]
    memory_coefficients = (train_encoded @ basis.conj())[:, :value_rank]
    split_raw = {}
    split_encoded = {}
    for split in ("validation", "test", "ood"):
        if not np.any(archive.split == split):
            continue
        raw = torch.as_tensor(
            archive.responses[archive.split == split].reshape(
                int(np.sum(archive.split == split)), -1
            ),
            dtype=torch.complex64,
            device=device,
        )
        split_raw[split] = raw
        split_encoded[split] = (raw - center[None]) / scale[None]
    source_count, frequency_count, receiver_count = archive.responses.shape[1:]
    return MetricState(
        archive=archive,
        center=center,
        scale=scale,
        basis=basis,
        variance=variance,
        penalty=_penalty(variance, operator.strength),
        train_encoded=train_encoded,
        train_raw=train_raw,
        memory_coefficients=memory_coefficients,
        split_encoded=split_encoded,
        split_raw=split_raw,
        coordinates=torch.as_tensor(response_coordinates(archive), device=device),
        source_count=source_count,
        frequency_count=frequency_count,
        receiver_count=receiver_count,
    )


@torch.inference_mode()
def cache_split(
    state: MetricState,
    split: str,
    budget: int,
    repeats: int,
    seed: int,
) -> list[MetricEpisode]:
    if split == "train":
        encoded = state.train_encoded
        raw = state.train_raw
    else:
        encoded = state.split_encoded[split]
        raw = state.split_raw[split]
    contexts, targets = _q3_events(
        state.source_count, state.frequency_count, budget, repeats, seed
    )
    receiver = torch.arange(state.receiver_count, device=state.basis.device)
    episodes = []
    for context, target_pair in zip(contexts, targets, strict=True):
        query = torch.as_tensor(
            context[:, 0] * state.frequency_count + context[:, 1],
            dtype=torch.long,
            device=state.basis.device,
        )
        observed = (query[:, None] * state.receiver_count + receiver[None]).reshape(-1)
        target_query = int(target_pair[0] * state.frequency_count + target_pair[1])
        target = target_query * state.receiver_count + receiver
        observed_basis = state.basis[observed]
        combined = torch.cat((encoded[:, observed], state.train_encoded[:, observed]), dim=0)
        coefficient, _ = solve_complex_posterior(
            observed_basis,
            combined,
            state.penalty,
        )
        split_count = len(encoded)
        episodes.append(
            MetricEpisode(
                coefficient=coefficient[:split_count],
                bank_coefficient=coefficient[split_count:],
                target_basis=state.basis[target],
                target_center=state.center[target],
                target_scale=state.scale[target],
                target_coordinate=state.coordinates[target[0]],
                truth=raw[:, target],
                query_indices=(
                    torch.arange(split_count, device=state.basis.device)
                    if split == "train"
                    else None
                ),
            )
        )
    return episodes


def posterior_kernel(
    query: torch.Tensor,
    bank: torch.Tensor,
    memory_value: torch.Tensor,
    variance: torch.Tensor,
    weight: torch.Tensor,
    neighbors: int,
    temperature: float,
    temperature_scale: torch.Tensor,
    blend: float,
    value_rank: int,
    query_indices: torch.Tensor | None = None,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    bank = bank.clone()
    memory_value = memory_value.clone()
    scale = torch.sqrt(variance.clone()).clamp_min(1e-15)
    query_metric = query / scale[None]
    bank_metric = bank / scale[None]
    if projection is not None:
        query_metric = query_metric @ projection
        bank_metric = bank_metric @ projection
    distance = (
        (query_metric.abs().square() * weight[None]).sum(dim=-1, keepdim=True)
        + (bank_metric.abs().square() * weight[None]).sum(dim=-1)[None]
        - 2.0
        * torch.real((query_metric * weight[None]) @ bank_metric.conj().T)
    ).clamp_min(0.0)
    if query_indices is not None:
        distance = distance.clone()
        distance[torch.arange(len(distance), device=distance.device), query_indices] = torch.inf
    selected, order = torch.topk(
        distance, min(neighbors, distance.shape[1] - int(query_indices is not None)), largest=False
    )
    relative = selected - selected[:, :1]
    local_scale = relative[:, -1:].clamp_min(1e-8)
    effective_temperature = max(temperature, 1e-6) * temperature_scale
    kernel_weight = torch.softmax(-relative / (effective_temperature * local_scale), dim=-1)
    memory_mean = (kernel_weight[..., None] * memory_value[order, :value_rank]).sum(dim=1)
    result = query.clone()
    result[:, :value_rank] = query[:, :value_rank] + blend * (
        memory_mean - query[:, :value_rank]
    )
    return result


def decode(coefficient: torch.Tensor, episode: MetricEpisode) -> torch.Tensor:
    normalized = coefficient @ episode.target_basis.clone().T
    return episode.target_center.clone()[None] + episode.target_scale.clone()[None] * normalized


def relative_error(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
        truth, dim=-1
    ).clamp_min(1e-12)


@torch.inference_mode()
def evaluate(
    model: QueryConditionedPosteriorMetric,
    episodes: list[MetricEpisode],
    state: MetricState,
    setting: dict,
    value_rank: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    system_count = len(episodes[0].truth)
    fixed_total = torch.zeros(system_count, device=state.basis.device)
    learned_total = torch.zeros_like(fixed_total)
    fixed_weight = torch.zeros_like(state.variance)
    fixed_weight[: int(setting["latent_dimensions"])] = 1.0
    for episode in episodes:
        learned_weight, temperature_scale, projection = model(episode.target_coordinate)
        for start in range(0, system_count, batch_size):
            stop = min(start + batch_size, system_count)
            query = episode.coefficient[start:stop]
            fixed = posterior_kernel(
                query,
                episode.bank_coefficient,
                state.memory_coefficients,
                state.variance,
                fixed_weight,
                int(setting["neighbors"]),
                float(setting["temperature"]),
                torch.ones((), device=query.device),
                float(setting["blend"]),
                value_rank,
            )
            learned = posterior_kernel(
                query,
                episode.bank_coefficient,
                state.memory_coefficients,
                state.variance,
                learned_weight,
                int(setting["neighbors"]),
                float(setting["temperature"]),
                temperature_scale,
                float(setting["blend"]),
                value_rank,
                projection=projection,
            )
            truth = episode.truth[start:stop]
            fixed_total[start:stop] += relative_error(decode(fixed, episode), truth)
            learned_total[start:stop] += relative_error(decode(learned, episode), truth)
    return (
        (fixed_total / len(episodes)).cpu().numpy().astype(np.float64),
        (learned_total / len(episodes)).cpu().numpy().astype(np.float64),
    )


def run_domain(
    domain: str,
    config: dict,
    output: Path,
    device_name: str,
    steps_override: int | None = None,
) -> dict:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    device = torch.device(device_name)
    rank = int(config["rank"])
    value_rank = int(config["value_rank"])
    setting = config["domains"][domain]
    state = build_state(setting, rank, value_rank, device)
    train_episodes = cache_split(
        state,
        "train",
        int(config["budget"]),
        int(config["train_repeats"]),
        int(config["train_prompt_seed"]),
    )
    validation_episodes = cache_split(
        state,
        "validation",
        int(config["budget"]),
        int(config["validation_repeats"]),
        int(config["validation_prompt_seed"]),
    )
    model = QueryConditionedPosteriorMetric(
        rank,
        int(setting["latent_dimensions"]),
        int(setting.get("metric_dimensions", setting["latent_dimensions"])),
        state.variance,
        int(config["width"]),
        int(config["mode_width"]),
        float(config["tail_log_weight"]),
        float(config["residual_bound"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    steps = int(config["steps"] if steps_override is None else steps_override)
    fixed_initial, learned_initial = evaluate(
        model,
        validation_episodes,
        state,
        setting,
        value_rank,
        int(config["eval_batch_size"]),
    )
    best_score = float(learned_initial.mean() / fixed_initial.mean())
    best_step = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [
        {
            "step": 0,
            "fixed_validation": float(fixed_initial.mean()),
            "learned_validation": float(learned_initial.mean()),
            "ratio": best_score,
        }
    ]
    started = time.perf_counter()
    batch_size = int(config["batch_size"])
    for step in range(1, steps + 1):
        model.train()
        episode = train_episodes[int(generator.integers(len(train_episodes)))]
        indices = torch.as_tensor(
            generator.choice(
                len(episode.coefficient),
                size=min(batch_size, len(episode.coefficient)),
                replace=False,
            ),
            dtype=torch.long,
            device=device,
        )
        weight, temperature_scale, projection = model(episode.target_coordinate)
        coefficient = posterior_kernel(
            episode.coefficient[indices],
            episode.bank_coefficient,
            state.memory_coefficients,
            state.variance,
            weight,
            int(setting["neighbors"]),
            float(setting["temperature"]),
            temperature_scale,
            float(setting["blend"]),
            value_rank,
            indices,
            projection,
        )
        loss = relative_error(decode(coefficient, episode), episode.truth[indices]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0 or step == steps:
            fixed, learned = evaluate(
                model,
                validation_episodes,
                state,
                setting,
                value_rank,
                int(config["eval_batch_size"]),
            )
            score = float(learned.mean() / fixed.mean())
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "fixed_validation": float(fixed.mean()),
                    "learned_validation": float(learned.mean()),
                    "ratio": score,
                }
            )
            if score < best_score:
                best_score = score
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    validation_fixed, validation_learned = evaluate(
        model,
        validation_episodes,
        state,
        setting,
        value_rank,
        int(config["eval_batch_size"]),
    )
    test_by_seed = []
    for test_seed in config["test_prompt_seeds"]:
        test_episodes = cache_split(
            state,
            "test",
            int(config["budget"]),
            int(config["test_repeats"]),
            int(test_seed),
        )
        fixed, learned = evaluate(
            model,
            test_episodes,
            state,
            setting,
            value_rank,
            int(config["eval_batch_size"]),
        )
        test_by_seed.append((int(test_seed), fixed, learned))
    fixed_test = np.mean([item[1] for item in test_by_seed], axis=0)
    learned_test = np.mean([item[2] for item in test_by_seed], axis=0)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "domain": domain,
            "config": config,
            "best_step": best_step,
        },
        output / "best.pt",
    )
    np.savez_compressed(
        output / "system_metrics.npz",
        validation_fixed=validation_fixed,
        validation_learned=validation_learned,
        test_fixed=fixed_test,
        test_learned=learned_test,
    )
    summary = {
        "domain": domain,
        "method": "query-conditioned observable posterior metric",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_step": best_step,
        "validation": {
            "fixed_mean": float(validation_fixed.mean()),
            "learned_mean": float(validation_learned.mean()),
            "relative_improvement": float(1.0 - validation_learned.mean() / validation_fixed.mean()),
        },
        "test": {
            "fixed_mean": float(fixed_test.mean()),
            "learned_mean": float(learned_test.mean()),
            "relative_improvement": float(1.0 - learned_test.mean() / fixed_test.mean()),
            **paired_summary(fixed_test, learned_test, seed),
            "by_prompt_seed": [
                {
                    "seed": item[0],
                    "fixed_mean": float(item[1].mean()),
                    "learned_mean": float(item[2].mean()),
                    "relative_improvement": float(1.0 - item[2].mean() / item[1].mean()),
                }
                for item in test_by_seed
            ],
        },
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.domain not in config["domains"]:
        raise ValueError(f"unknown domain {args.domain!r}")
    print(
        json.dumps(
            run_domain(args.domain, config, args.output, args.device, args.steps),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
