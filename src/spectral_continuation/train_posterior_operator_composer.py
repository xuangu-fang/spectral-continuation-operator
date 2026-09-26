"""Train a shared posterior-initialized retrieval operator composer.

The composer receives analytic complex posterior coefficients, their
uncertainty, a compressed rank-limited operator dictionary, and the target
source--frequency query.  It updates latent operator tokens with cross-attention
before decoding through the frozen global response basis.  With the learned
update set to zero, the model exactly recovers the frozen target-free continuous
prior adapter between posterior-kernel and neural proposals.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit_prompt_only_router import load_neural
from .calibrate_prompt_router import features as loo_features
from .cross_domain_response_data import CrossDomainResponseArchive, load_cross_domain_archive
from .global_bayesian_response_operator import (
    GlobalConfig,
    _q3_events,
    fit_global_subspace,
    solve_complex_posterior,
)
from .posterior_memory_response_operator import _penalty
from .probabilistic_response_operator import paired_summary
from .train_prompted_bayesian_neural_operator import (
    _indices,
    fourier_features,
    response_coordinates,
)
from .train_prompted_bayesian_neural_operator import (
    predict as predict_neural,
)
from .train_unified_prompt_prior_adapter import UnifiedPriorAdapter


@dataclass
class DomainState:
    domain: str
    budget: int
    operator: GlobalConfig
    archive: CrossDomainResponseArchive
    center: torch.Tensor
    scale: torch.Tensor
    basis: torch.Tensor
    variance: torch.Tensor
    penalty: torch.Tensor
    memory_coefficients: torch.Tensor
    train_encoded: torch.Tensor
    split_encoded: dict[str, torch.Tensor]
    split_raw: dict[str, torch.Tensor]
    coordinates: torch.Tensor
    source_count: int
    frequency_count: int
    receiver_count: int
    memory_dimensions: int
    memory_neighbors: int
    memory_temperature: float
    memory_blend: float
    neural_model: nn.Module
    neural_center: torch.Tensor
    neural_scale: torch.Tensor
    neural_encoded: dict[str, torch.Tensor]
    adapter: UnifiedPriorAdapter
    adapter_mean: torch.Tensor
    adapter_standard_deviation: torch.Tensor


@dataclass
class CachedEpisode:
    domain: str
    budget: int
    split: str
    query_coefficient: torch.Tensor
    bank_coefficient: torch.Tensor
    posterior_variance: torch.Tensor
    memory_coefficients: torch.Tensor
    variance: torch.Tensor
    target_basis: torch.Tensor
    target_center: torch.Tensor
    target_scale: torch.Tensor
    target_features: torch.Tensor
    truth: torch.Tensor
    memory_dimensions: int
    memory_neighbors: int
    memory_temperature: float
    memory_blend: float
    neural_proposal: torch.Tensor
    prior_gate: torch.Tensor
    risk_features: torch.Tensor


class ComposerBlock(nn.Module):
    def __init__(self, width: int, heads: int, expansion: int) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.SiLU(),
            nn.Linear(expansion * width, width),
        )

    def forward(self, latent: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.cross_norm(latent)
        update, _ = self.cross_attention(normalized, memory, memory, need_weights=False)
        latent = latent + update
        normalized = self.self_norm(latent)
        update, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        latent = latent + update
        return latent + self.feedforward(self.feedforward_norm(latent))


class PosteriorOperatorComposer(nn.Module):
    """Shared latent-token composer without domain identifiers."""

    def __init__(
        self,
        rank: int,
        value_rank: int,
        retrieval_dimensions: int,
        neighbors: int,
        width: int,
        heads: int,
        layers: int,
        expansion: int,
        residual_bound: float,
    ) -> None:
        super().__init__()
        if value_rank > rank or retrieval_dimensions > rank:
            raise ValueError("dictionary and retrieval ranks cannot exceed model rank")
        self.rank = rank
        self.value_rank = value_rank
        self.retrieval_dimensions = retrieval_dimensions
        self.neighbors = neighbors
        self.residual_bound = residual_bound
        self.latent_embedding = nn.Sequential(
            nn.Linear(10, width), nn.SiLU(), nn.Linear(width, width)
        )
        memory_features = 4 * value_rank + 2
        self.memory_embedding = nn.Sequential(
            nn.Linear(memory_features, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.query_embedding = nn.Sequential(
            nn.Linear(51, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            ComposerBlock(width, heads, expansion) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, 1)
        self.residual = nn.Linear(width, 2)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        coefficient: torch.Tensor,
        posterior_variance: torch.Tensor,
        bank_coefficient: torch.Tensor,
        memory_coefficients: torch.Tensor,
        variance: torch.Tensor,
        target_features: torch.Tensor,
        memory_dimensions: int,
        memory_neighbors: int,
        memory_temperature: float,
        memory_blend: float,
        neural_proposal: torch.Tensor,
        prior_gate: torch.Tensor,
        risk_features: torch.Tensor,
        query_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        standard_deviation = torch.sqrt(variance.clamp_min(1e-30))
        whitened = coefficient / standard_deviation[None]
        uncertainty = torch.log1p(
            posterior_variance.clamp_min(0.0) / variance.clamp_min(1e-30)
        )
        log_variance = torch.log(variance.clamp_min(1e-30))
        log_variance = (log_variance - log_variance.mean()) / log_variance.std().clamp_min(
            1e-6
        )
        position = torch.linspace(-1.0, 1.0, self.rank, device=coefficient.device)
        latent_features = torch.stack(
            (
                whitened.real,
                whitened.imag,
                uncertainty[None].expand(len(coefficient), -1),
                log_variance[None].expand(len(coefficient), -1),
                torch.sin(torch.pi * position)[None].expand(len(coefficient), -1),
                torch.cos(torch.pi * position)[None].expand(len(coefficient), -1),
            ),
            dim=-1,
        )

        dimension = min(memory_dimensions, self.retrieval_dimensions)
        # Reproduce the validated posterior-kernel metric exactly at zero
        # learned update.  Posterior uncertainty remains an explicit latent
        # token and can modulate the learned correction without changing the
        # analytic anchor's neighbor geometry.
        metric_scale = torch.sqrt(variance[:dimension]).clamp_min(1e-15)
        query_metric = coefficient[:, :dimension] / metric_scale[None]
        bank_metric = bank_coefficient[:, :dimension] / metric_scale[None]
        distance = (
            query_metric.abs().square().sum(dim=-1, keepdim=True)
            + bank_metric.abs().square().sum(dim=-1)[None]
            - 2.0 * torch.real(query_metric @ bank_metric.conj().T)
        ).clamp_min(0.0)
        if query_indices is not None:
            distance = distance.clone()
            distance[torch.arange(len(distance), device=distance.device), query_indices] = torch.inf
        selected_distance, order = torch.topk(
            distance,
            min(memory_neighbors, self.neighbors, distance.shape[1]),
            largest=False,
            dim=-1,
        )

        value_rank = self.value_rank
        relative = (
            bank_coefficient[order, :value_rank] - coefficient[:, None, :value_rank]
        ) / standard_deviation[None, None, :value_rank]
        value = memory_coefficients[order, :value_rank] / standard_deviation[
            None, None, :value_rank
        ]
        relative_distance = selected_distance - selected_distance[:, :1]
        local_scale = relative_distance[:, -1:].clamp_min(1e-8)
        normalized_distance = relative_distance / local_scale
        if memory_temperature <= 0.0 or selected_distance.shape[-1] == 1:
            kernel_weight = torch.full_like(
                normalized_distance, 1.0 / selected_distance.shape[-1]
            )
        else:
            kernel_weight = torch.softmax(
                -normalized_distance / memory_temperature, dim=-1
            )
        memory_mean = (
            kernel_weight[..., None] * memory_coefficients[order, :value_rank]
        ).sum(dim=1)
        base_coefficient = coefficient.clone()
        base_coefficient[:, :value_rank] = (
            coefficient[:, :value_rank]
            + memory_blend * (memory_mean - coefficient[:, :value_rank])
        )
        kernel_delta = (base_coefficient - coefficient) / standard_deviation[None]
        neural_delta = (neural_proposal - coefficient) / standard_deviation[None]
        latent_features = torch.cat(
            (
                latent_features[..., :2],
                torch.stack((kernel_delta.real, kernel_delta.imag), dim=-1),
                torch.stack((neural_delta.real, neural_delta.imag), dim=-1),
                latent_features[..., 2:],
            ),
            dim=-1,
        )
        memory_features = torch.cat(
            (
                relative.real,
                relative.imag,
                value.real,
                value.imag,
                torch.log1p(normalized_distance)[..., None],
                kernel_weight[..., None],
            ),
            dim=-1,
        )
        memory = self.memory_embedding(memory_features)
        # Cached episode tensors are built under inference mode.  Clone the
        # small query vector so autograd can save it for the first linear layer.
        query_features = torch.cat(
            (
                target_features.clone()[None].expand(len(coefficient), -1),
                risk_features.clone(),
            ),
            dim=-1,
        )
        query = self.query_embedding(query_features)[:, None]
        latent = self.latent_embedding(latent_features) + query
        for block in self.blocks:
            latent = block(latent, memory)
        latent = self.output_norm(latent)
        prior_logit = torch.logit(prior_gate.clamp(1e-4, 1.0 - 1e-4))[:, None]
        gate = torch.sigmoid(prior_logit + 0.25 * self.gate(latent)[..., 0])
        mixed_coefficient = base_coefficient + gate * (
            neural_proposal - base_coefficient
        )
        residual = self.residual(latent)
        correction = self.residual_bound * torch.complex(
            torch.tanh(residual[..., 0]), torch.tanh(residual[..., 1])
        )
        return mixed_coefficient + correction * standard_deviation[None]


def _kernel_coefficient(
    coefficient: torch.Tensor,
    bank_coefficient: torch.Tensor,
    memory_coefficients: torch.Tensor,
    variance: torch.Tensor,
    dimensions: int,
    neighbors: int,
    temperature: float,
    blend: float,
    query_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the validated posterior-memory update in coefficient space."""

    dimensions = min(dimensions, coefficient.shape[-1])
    metric_scale = torch.sqrt(variance[:dimensions]).clamp_min(1e-15)
    query_metric = coefficient[:, :dimensions] / metric_scale[None]
    bank_metric = bank_coefficient[:, :dimensions] / metric_scale[None]
    distance = (
        query_metric.abs().square().sum(dim=-1, keepdim=True)
        + bank_metric.abs().square().sum(dim=-1)[None]
        - 2.0 * torch.real(query_metric @ bank_metric.conj().T)
    ).clamp_min(0.0)
    if query_indices is not None:
        distance = distance.clone()
        distance[
            torch.arange(len(distance), device=distance.device), query_indices
        ] = torch.inf
    selected_distance, order = torch.topk(
        distance, min(neighbors, distance.shape[1]), largest=False, dim=-1
    )
    relative_distance = selected_distance - selected_distance[:, :1]
    local_scale = relative_distance[:, -1:].clamp_min(1e-8)
    normalized_distance = relative_distance / local_scale
    if temperature <= 0.0 or selected_distance.shape[-1] == 1:
        weights = torch.full_like(
            normalized_distance, 1.0 / selected_distance.shape[-1]
        )
    else:
        weights = torch.softmax(-normalized_distance / temperature, dim=-1)
    value_rank = memory_coefficients.shape[-1]
    memory_mean = (
        weights[..., None] * memory_coefficients[order, :value_rank]
    ).sum(dim=1)
    result = coefficient.clone()
    result[:, :value_rank] = coefficient[:, :value_rank] + blend * (
        memory_mean - coefficient[:, :value_rank]
    )
    return result


@torch.inference_mode()
def _episode_branches(
    state: DomainState,
    split: str,
    context: np.ndarray,
    target_pair: np.ndarray,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return coefficients, uncertainty, indices, and the three predictions."""

    device = state.basis.device
    observed, target = _indices(
        context,
        target_pair,
        state.frequency_count,
        state.receiver_count,
        device,
    )
    observed_basis = state.basis[observed]
    encoded = state.split_encoded[split]
    bank_context = state.memory_coefficients @ observed_basis[:, : len(
        state.memory_coefficients[0]
    )].T
    combined = torch.cat((encoded[:, observed], bank_context), dim=0)
    coefficient, posterior_variance = solve_complex_posterior(
        observed_basis,
        combined,
        state.penalty,
        return_variance=True,
    )
    if posterior_variance is None:
        raise RuntimeError("posterior variance was requested")
    split_count = len(encoded)
    query_coefficient = coefficient[:split_count]
    bank_coefficient = coefficient[split_count:]
    query_indices = (
        torch.arange(split_count, device=device) if split == "train" else None
    )
    kernel_coefficient = _kernel_coefficient(
        query_coefficient,
        bank_coefficient,
        state.memory_coefficients,
        state.variance,
        state.memory_dimensions,
        state.memory_neighbors,
        state.memory_temperature,
        state.memory_blend,
        query_indices,
    )

    def decode(value: torch.Tensor) -> torch.Tensor:
        normalized = value @ state.basis[target].T
        return state.center[target][None] + state.scale[target][None] * normalized

    global_prediction = decode(query_coefficient)
    kernel_prediction = decode(kernel_coefficient)
    neural_prediction = predict_neural(
        state.neural_model,
        state.neural_encoded[split],
        state.neural_center,
        state.neural_scale,
        observed,
        target,
    )
    return (
        query_coefficient,
        bank_coefficient,
        posterior_variance,
        observed,
        target,
        torch.stack((global_prediction, kernel_prediction, neural_prediction)),
    )


def _relative_error(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
        truth, dim=-1
    ).clamp_min(1e-12)


@torch.inference_mode()
def _adapter_gate(
    state: DomainState,
    split: str,
    context: np.ndarray,
    observed: torch.Tensor,
    target: torch.Tensor,
    branches: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the frozen target-free strong-baseline gate for one episode."""

    raw_prompt = state.split_raw[split][:, observed]
    amplitude = raw_prompt.abs()
    amplitude_mean = amplitude.mean(dim=-1).clamp_min(1e-12)
    amplitude_cv = amplitude.std(dim=-1) / amplitude_mean
    amplitude_peak = amplitude.amax(dim=-1) / amplitude_mean
    unit_phase = raw_prompt / amplitude.clamp_min(1e-12)
    phase_roughness = (unit_phase[..., 1:] - unit_phase[..., :-1]).abs().mean(dim=-1)

    def disagreement(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        numerator = torch.linalg.vector_norm(left - right, dim=-1)
        denominator = 0.5 * (
            torch.linalg.vector_norm(left, dim=-1)
            + torch.linalg.vector_norm(right, dim=-1)
        ).clamp_min(1e-12)
        return numerator / denominator

    global_prediction, kernel_prediction, neural_prediction = branches
    prediction_scale = torch.linalg.vector_norm(global_prediction, dim=-1).clamp_min(
        1e-12
    )
    prompt_scale = torch.linalg.vector_norm(raw_prompt, dim=-1).clamp_min(1e-12)
    prompt_features = torch.stack(
        (
            torch.log1p(amplitude_cv),
            torch.log1p(amplitude_peak),
            phase_roughness,
            torch.log1p(disagreement(global_prediction, kernel_prediction)),
            torch.log1p(disagreement(global_prediction, neural_prediction)),
            torch.log1p(disagreement(kernel_prediction, neural_prediction)),
            torch.log1p(
                torch.linalg.vector_norm(kernel_prediction, dim=-1) / prediction_scale
            ),
            torch.log1p(prediction_scale / prompt_scale),
        ),
        dim=-1,
    )
    budget = len(context)
    if budget < 2:
        loo = torch.zeros(
            (len(raw_prompt), 8), dtype=torch.float32, device=raw_prompt.device
        )
        availability = torch.zeros(
            (len(raw_prompt), 1), dtype=torch.float32, device=raw_prompt.device
        )
    else:
        self_error = torch.zeros(
            (len(raw_prompt), 3), dtype=torch.float32, device=raw_prompt.device
        )
        held_positions = np.linspace(0, budget - 1, min(2, budget), dtype=np.int64)
        for held in held_positions:
            held_context = np.delete(context, held, axis=0)
            held_target = context[held]
            *_, held_target_indices, held_branches = _episode_branches(
                state, split, held_context, held_target
            )
            held_truth = state.split_raw[split][:, held_target_indices]
            self_error += _relative_error(held_branches, held_truth[None]).T
        self_error /= len(held_positions)
        loo = torch.as_tensor(
            loo_features(self_error.cpu().numpy()),
            dtype=torch.float32,
            device=raw_prompt.device,
        )
        availability = torch.ones(
            (len(raw_prompt), 1), dtype=torch.float32, device=raw_prompt.device
        )
    features = torch.cat((prompt_features, loo, availability), dim=-1)
    normalized = (features - state.adapter_mean[None]) / state.adapter_standard_deviation[
        None
    ]
    return state.adapter(normalized), normalized


def _build_state(
    domain: str,
    budget: int,
    setting: dict,
    specification: dict,
    rank: int,
    value_rank: int,
    adapter_checkpoint: Path,
    device: torch.device,
) -> DomainState:
    archive = load_cross_domain_archive(Path(setting["dataset"]))
    operator_payload = specification["operator"]
    memory_payload = specification["memory"]
    operator = GlobalConfig(
        rank=rank,
        strength=float(operator_payload["strength"]),
        prior="spectrum",
        normalization=str(operator_payload["normalization"]),
    )
    subspace = fit_global_subspace(archive, rank, operator.normalization, str(device))
    center = torch.as_tensor(subspace.center, device=device)
    scale = torch.as_tensor(subspace.scale, device=device)
    basis = torch.as_tensor(subspace.basis[:, :rank], device=device)
    variance = torch.as_tensor(
        subspace.variance[:rank], dtype=torch.float32, device=device
    )
    train = archive.responses[archive.split == "train"].reshape(
        int(np.sum(archive.split == "train")), -1
    )
    train_raw = torch.as_tensor(train, dtype=torch.complex64, device=device)
    train_encoded = (train_raw - center[None]) / scale[None]
    memory_coefficients = (train_encoded @ basis.conj())[:, :value_rank]
    split_raw = {}
    split_encoded = {}
    for split in ("train", "validation", "test", "ood"):
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
    neural_model, neural_center, neural_scale, _ = load_neural(
        Path(setting["neural_checkpoint"]), archive, str(device)
    )
    neural_encoded = {
        split: (raw - neural_center[None]) / neural_scale[None]
        for split, raw in split_raw.items()
    }
    adapter_payload = torch.load(
        adapter_checkpoint, map_location=device, weights_only=False
    )
    adapter = UnifiedPriorAdapter().to(device)
    adapter.load_state_dict(adapter_payload["model"])
    adapter.eval()
    adapter_mean = torch.as_tensor(
        adapter_payload["feature_mean"], dtype=torch.float32, device=device
    )
    adapter_standard_deviation = torch.as_tensor(
        adapter_payload["feature_std"], dtype=torch.float32, device=device
    )
    temporary = type(
        "PenaltyState",
        (),
        {"variance": variance},
    )()
    penalty = _penalty(temporary, operator)
    return DomainState(
        domain=domain,
        budget=budget,
        operator=operator,
        archive=archive,
        center=center,
        scale=scale,
        basis=basis,
        variance=variance,
        penalty=penalty,
        memory_coefficients=memory_coefficients,
        train_encoded=train_encoded,
        split_encoded=split_encoded,
        split_raw=split_raw,
        coordinates=torch.as_tensor(response_coordinates(archive), device=device),
        source_count=source_count,
        frequency_count=frequency_count,
        receiver_count=receiver_count,
        memory_dimensions=int(memory_payload["latent_dimensions"]),
        memory_neighbors=int(memory_payload["neighbors"]),
        memory_temperature=float(memory_payload["temperature"]),
        memory_blend=float(memory_payload["blend"]),
        neural_model=neural_model,
        neural_center=neural_center,
        neural_scale=neural_scale,
        neural_encoded=neural_encoded,
        adapter=adapter,
        adapter_mean=adapter_mean,
        adapter_standard_deviation=adapter_standard_deviation,
    )


@torch.inference_mode()
def _cache_episode(
    state: DomainState,
    split: str,
    context: np.ndarray,
    target_pair: np.ndarray,
) -> CachedEpisode:
    device = state.basis.device
    (
        query_coefficient,
        bank_coefficient,
        posterior_variance,
        observed,
        target,
        branches,
    ) = _episode_branches(state, split, context, target_pair)
    query_coordinate = state.coordinates[target[0], :3]
    target_features = torch.cat(
        (
            fourier_features(query_coordinate),
            torch.tensor(
                [np.log2(state.budget) / 3.0],
                dtype=torch.float32,
                device=device,
            ),
        )
    )
    neural_prediction = branches[2]
    neural_normalized = (
        neural_prediction - state.center[target][None]
    ) / state.scale[target][None]
    neural_proposal = neural_normalized @ torch.linalg.pinv(state.basis[target]).T
    prior_gate, risk_features = _adapter_gate(
        state, split, context, observed, target, branches
    )
    return CachedEpisode(
        domain=state.domain,
        budget=state.budget,
        split=split,
        query_coefficient=query_coefficient,
        bank_coefficient=bank_coefficient,
        posterior_variance=posterior_variance,
        memory_coefficients=state.memory_coefficients,
        variance=state.variance,
        target_basis=state.basis[target],
        target_center=state.center[target],
        target_scale=state.scale[target],
        target_features=target_features,
        truth=state.split_raw[split][:, target],
        memory_dimensions=state.memory_dimensions,
        memory_neighbors=state.memory_neighbors,
        memory_temperature=state.memory_temperature,
        memory_blend=state.memory_blend,
        neural_proposal=neural_proposal,
        prior_gate=prior_gate,
        risk_features=risk_features,
    )


def _cache_split(
    states: dict[tuple[str, int], DomainState],
    split: str,
    repeats: int,
    seed: int,
) -> dict[tuple[str, int], list[CachedEpisode]]:
    cached = {}
    for key, state in states.items():
        if split not in state.split_raw:
            continue
        contexts, targets = _q3_events(
            state.source_count,
            state.frequency_count,
            state.budget,
            repeats,
            seed,
        )
        cached[key] = [
            _cache_episode(state, split, context, target)
            for context, target in zip(contexts, targets, strict=True)
        ]
    return cached


def _partition_cached_systems(
    cached: dict[tuple[str, int], list[CachedEpisode]],
    fit_fraction: float,
    seed: int,
) -> tuple[
    dict[tuple[str, int], list[CachedEpisode]],
    dict[tuple[str, int], list[CachedEpisode]],
]:
    """Split calibration systems while preserving identical prompt events."""

    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie strictly between zero and one")
    fit = {}
    selection = {}
    for offset, (key, episodes) in enumerate(sorted(cached.items())):
        system_count = len(episodes[0].truth)
        order = np.random.default_rng(seed + offset).permutation(system_count)
        fit_count = max(1, min(system_count - 1, int(fit_fraction * system_count)))
        fit_index = torch.as_tensor(
            order[:fit_count], dtype=torch.long, device=episodes[0].truth.device
        )
        selection_index = torch.as_tensor(
            order[fit_count:], dtype=torch.long, device=episodes[0].truth.device
        )

        def subset(episode: CachedEpisode, index: torch.Tensor) -> CachedEpisode:
            return replace(
                episode,
                query_coefficient=episode.query_coefficient[index],
                neural_proposal=episode.neural_proposal[index],
                prior_gate=episode.prior_gate[index],
                risk_features=episode.risk_features[index],
                truth=episode.truth[index],
            )

        fit[key] = [subset(episode, fit_index) for episode in episodes]
        selection[key] = [subset(episode, selection_index) for episode in episodes]
    return fit, selection


def _decode(
    coefficient: torch.Tensor,
    episode: CachedEpisode,
) -> torch.Tensor:
    target_basis = episode.target_basis.clone()
    target_scale = episode.target_scale.clone()
    normalized = coefficient @ target_basis.T
    return episode.target_center[None] + target_scale[None] * normalized


@torch.inference_mode()
def evaluate_cached(
    model: PosteriorOperatorComposer,
    cached: dict[tuple[str, int], list[CachedEpisode]],
    batch_size: int,
) -> dict[str, dict[str, np.ndarray | float]]:
    model.eval()
    result = {}
    for (domain, budget), episodes in cached.items():
        system_count = len(episodes[0].truth)
        candidate_total = torch.zeros(system_count, device=episodes[0].truth.device)
        global_total = torch.zeros_like(candidate_total)
        for episode in episodes:
            denominator = torch.linalg.vector_norm(episode.truth, dim=-1).clamp_min(1e-12)
            global_prediction = _decode(episode.query_coefficient, episode)
            global_total += (
                torch.linalg.vector_norm(global_prediction - episode.truth, dim=-1)
                / denominator
            )
            for start in range(0, system_count, batch_size):
                stop = min(start + batch_size, system_count)
                coefficient = model(
                    episode.query_coefficient[start:stop],
                    episode.posterior_variance,
                    episode.bank_coefficient,
                    episode.memory_coefficients,
                    episode.variance,
                    episode.target_features,
                    episode.memory_dimensions,
                    episode.memory_neighbors,
                    episode.memory_temperature,
                    episode.memory_blend,
                    episode.neural_proposal[start:stop],
                    episode.prior_gate[start:stop],
                    episode.risk_features[start:stop],
                )
                prediction = _decode(coefficient, episode)
                candidate_total[start:stop] += (
                    torch.linalg.vector_norm(
                        prediction - episode.truth[start:stop], dim=-1
                    )
                    / denominator[start:stop]
                )
        global_error = (global_total / len(episodes)).cpu().numpy().astype(np.float64)
        candidate_error = (
            candidate_total / len(episodes)
        ).cpu().numpy().astype(np.float64)
        result[f"{domain}_k{budget}"] = {
            "global_mean": float(global_error.mean()),
            "candidate_mean": float(candidate_error.mean()),
            **paired_summary(global_error, candidate_error, 20260926 + budget),
            "_global": global_error,
            "_candidate": candidate_error,
        }
    return result


def run(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    torch.manual_seed(int(config["seed"]))
    generator = np.random.default_rng(int(config["seed"]))
    device = torch.device(device_name)
    rank = int(config["rank"])
    value_rank = int(config["value_rank"])
    states = {}
    for domain, setting in config["domains"].items():
        for budget_text, specification in setting["budgets"].items():
            budget = int(budget_text)
            states[(domain, budget)] = _build_state(
                domain,
                budget,
                setting,
                specification,
                rank,
                value_rank,
                Path(config["adapter_checkpoint"]),
                device,
            )
    started = time.perf_counter()
    calibration_cache = _cache_split(
        states,
        "validation",
        int(config["calibration_repeats"]),
        int(config["selection_seed"]),
    )
    train_cache, validation_cache = _partition_cached_systems(
        calibration_cache,
        float(config["calibration_fit_fraction"]),
        int(config["seed"]),
    )
    model = PosteriorOperatorComposer(
        rank=rank,
        value_rank=value_rank,
        retrieval_dimensions=int(config["retrieval_dimensions"]),
        neighbors=int(config["neighbors"]),
        width=int(config["width"]),
        heads=int(config["heads"]),
        layers=int(config["layers"]),
        expansion=int(config["expansion"]),
        residual_bound=float(config["residual_bound"]),
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= 1_000_000:
        raise ValueError(f"composer has {parameters:,} parameters, exceeding the M0 gate")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    keys = sorted(train_cache)
    keys_by_domain = {
        domain: [key for key in keys if key[0] == domain]
        for domain in sorted({key[0] for key in keys})
    }
    batch_size = int(config["batch_size"])
    initial_validation = evaluate_cached(
        model, validation_cache, int(config["eval_batch_size"])
    )
    validation_anchor = {
        key: float(value["candidate_mean"])
        for key, value in initial_validation.items()
    }
    history = [
        {
            "step": 0,
            "selection_score": 1.25,
            "mean_ratio_to_kernel_anchor": 1.0,
            "worst_ratio_to_kernel_anchor": 1.0,
        }
    ]
    best_score = 1.25
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    for step in range(1, int(config["steps"]) + 1):
        model.train()
        domain_losses = []
        raw_losses = []
        for domain_keys in keys_by_domain.values():
            key = domain_keys[int(generator.integers(len(domain_keys)))]
            episodes = train_cache[key]
            episode = episodes[int(generator.integers(len(episodes)))]
            indices_np = generator.integers(
                0, len(episode.query_coefficient), size=batch_size
            )
            indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
            coefficient = model(
                episode.query_coefficient[indices],
                episode.posterior_variance,
                episode.bank_coefficient,
                episode.memory_coefficients,
                episode.variance,
                episode.target_features,
                episode.memory_dimensions,
                episode.memory_neighbors,
                episode.memory_temperature,
                episode.memory_blend,
                episode.neural_proposal[indices],
                episode.prior_gate[indices],
                episode.risk_features[indices],
            )
            prediction = _decode(coefficient, episode)
            truth = episode.truth[indices]
            raw_loss = (
                torch.linalg.vector_norm(prediction - truth, dim=-1)
                / torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)
            ).mean()
            raw_losses.append(raw_loss)
            domain_losses.append(raw_loss / validation_anchor[f"{key[0]}_k{key[1]}"])
        loss = torch.stack(domain_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0 or step == int(config["steps"]):
            validation = evaluate_cached(model, validation_cache, int(config["eval_batch_size"]))
            ratios = [
                value["candidate_mean"] / validation_anchor[key]
                for key, value in validation.items()
            ]
            score = float(max(ratios) + 0.25 * np.mean(ratios))
            record = {
                "step": step,
                "train_loss": float(torch.stack(raw_losses).mean().detach()),
                "normalized_train_loss": float(loss.detach()),
                "selection_score": score,
                "mean_ratio_to_kernel_anchor": float(np.mean(ratios)),
                "worst_ratio_to_kernel_anchor": float(max(ratios)),
            }
            history.append(record)
            if score < best_score:
                best_score = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    validation = evaluate_cached(model, validation_cache, int(config["eval_batch_size"]))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state, "config": config, "parameters": parameters},
        output / "best.pt",
    )
    machine_validation = {}
    for key, value in validation.items():
        np.savez_compressed(
            output / f"{key.lower()}_validation_systems.npz",
            global_error=value.pop("_global"),
            candidate_error=value.pop("_candidate"),
        )
        machine_validation[key] = value
    summary = {
        "parameters": parameters,
        "best_selection_score": best_score,
        "validation_kernel_anchor": validation_anchor,
        "cache_and_training_seconds": time.perf_counter() - started,
        "validation": machine_validation,
        "history": history,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
