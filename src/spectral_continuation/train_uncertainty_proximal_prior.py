"""Learn an uncertainty-conditioned operator prior with analytic assimilation.

The learned map predicts a *prior mean* for the complete response-operator
latent.  It never writes the target response directly.  For every prompt
episode the proposal is assimilated through the observed complex response
operator in closed form, after which the frozen response basis answers a
query.  A validation-selected convex blend contains the current M4 adapter as
the exact ``blend=0`` fallback.
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
from torch.nn import functional as F

from .audit_heterogeneous_prompt_headroom import _indices as heterogeneous_indices
from .audit_heterogeneous_prompt_headroom import _receiver_masks
from .global_bayesian_response_operator import _q3_events, solve_complex_posterior
from .probabilistic_response_operator import paired_summary
from .train_posterior_operator_composer import (
    DomainState,
    _adapter_gate,
    _build_state,
    _episode_branches,
    _kernel_coefficient,
)


class ComplexLinear(nn.Module):
    """Bias-free complex linear map, equivariant to a shared phase rotation."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.real = nn.Parameter(torch.empty(outputs, inputs))
        self.imag = nn.Parameter(torch.empty(outputs, inputs))
        nn.init.kaiming_uniform_(self.real, a=np.sqrt(5.0))
        nn.init.kaiming_uniform_(self.imag, a=np.sqrt(5.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            F.linear(value.real, self.real) - F.linear(value.imag, self.imag),
            F.linear(value.real, self.imag) + F.linear(value.imag, self.real),
        )


class ComplexRadialActivation(nn.Module):
    """Stable phase-preserving nonlinearity without division by magnitude."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(width))
        self.log_slope = nn.Parameter(torch.full((width,), -1.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        magnitude_feature = torch.log1p(value.abs().square())
        gate = torch.sigmoid(
            self.log_gain[None] + torch.exp(self.log_slope)[None] * magnitude_feature
        )
        return value * gate


class UncertaintyConditionedOperatorPrior(nn.Module):
    """Small global complex prior over response-operator spectral modes."""

    def __init__(
        self,
        rank: int,
        width: int,
        depth: int,
        proposal_bound: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.rank = rank
        self.proposal_bound = proposal_bound
        layers: list[nn.Module] = [ComplexLinear(rank, width), ComplexRadialActivation(width)]
        for _ in range(depth - 1):
            layers.extend((ComplexLinear(width, width), ComplexRadialActivation(width)))
        self.complex_layers = nn.ModuleList(layers)
        self.output = ComplexLinear(width, rank)
        # Step zero is exactly the zero-mean analytic Bayesian prior.
        nn.init.zeros_(self.output.real)
        nn.init.zeros_(self.output.imag)
        self.mode_gate = nn.Sequential(
            nn.Linear(6, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        posterior_mean: torch.Tensor,
        posterior_variance: torch.Tensor,
        latent_variance: torch.Tensor,
    ) -> torch.Tensor:
        standard_deviation = torch.sqrt(latent_variance.clamp_min(1e-30))
        normalized = posterior_mean / standard_deviation[None]
        hidden = normalized
        for layer in self.complex_layers:
            hidden = layer(hidden)
        proposal = self.output(hidden)
        # Smoothly bound the proposal without changing its phase.
        proposal = proposal / torch.sqrt(
            1.0 + proposal.abs().square() / max(self.proposal_bound**2, 1e-12)
        )

        uncertainty = torch.log1p(
            posterior_variance.clamp_min(0.0) / latent_variance.clamp_min(1e-30)
        )
        log_variance = torch.log(latent_variance.clamp_min(1e-30))
        log_variance = (log_variance - log_variance.mean()) / log_variance.std().clamp_min(
            1e-6
        )
        position = torch.linspace(-1.0, 1.0, self.rank, device=posterior_mean.device)
        invariant = torch.stack(
            (
                torch.log1p(normalized.abs()),
                uncertainty[None].expand(len(posterior_mean), -1),
                log_variance[None].expand(len(posterior_mean), -1),
                position[None].expand(len(posterior_mean), -1),
                torch.sin(torch.pi * position)[None].expand(len(posterior_mean), -1),
                torch.cos(torch.pi * position)[None].expand(len(posterior_mean), -1),
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.mode_gate(invariant)[..., 0])
        return gate * proposal * standard_deviation[None]


@dataclass
class ProximalEpisode:
    domain: str
    budget: int
    posterior_mean: torch.Tensor
    posterior_variance: torch.Tensor
    latent_variance: torch.Tensor
    data_coefficient: torch.Tensor
    prior_projector: torch.Tensor
    baseline_coefficient: torch.Tensor | None
    target_basis: torch.Tensor
    target_center: torch.Tensor
    target_scale: torch.Tensor
    truth: torch.Tensor
    observed_basis: torch.Tensor
    observed_values: torch.Tensor


def assimilate_prior(
    prior_mean: torch.Tensor,
    data_coefficient: torch.Tensor,
    prior_projector: torch.Tensor,
) -> torch.Tensor:
    """Closed-form likelihood/prior assimilation in response latent space."""

    # Episode caches are intentionally built under inference mode.  Clone the
    # small frozen tensors before autograd needs to save them for backward.
    return data_coefficient.clone() + prior_mean @ prior_projector.clone().T


def apply_proximal_correction(
    baseline: torch.Tensor,
    correction: torch.Tensor,
    correction_projector: torch.Tensor,
    observed_basis: torch.Tensor,
    observed_values: torch.Tensor,
    safeguard: bool,
) -> torch.Tensor:
    """Apply a precision-filtered correction, optionally enforcing prompt fit."""

    filtered = correction @ correction_projector.clone().T
    if not safeguard:
        return baseline + filtered
    prompt_delta = filtered @ observed_basis.clone().T
    prompt_residual = baseline @ observed_basis.clone().T - observed_values.clone()
    numerator = -torch.real(
        (prompt_delta.conj() * prompt_residual).sum(dim=-1)
    )
    denominator = prompt_delta.abs().square().sum(dim=-1).clamp_min(1e-20)
    step = (numerator / denominator).clamp(0.0, 1.0)
    return baseline + step[:, None] * filtered


def _strong_baseline_coefficient(
    state: DomainState,
    split: str,
    context: np.ndarray,
    target_pair: np.ndarray,
    posterior_mean: torch.Tensor,
    bank_coefficient: torch.Tensor,
    observed: torch.Tensor,
    target: torch.Tensor,
    branches: torch.Tensor,
) -> torch.Tensor:
    query_indices = (
        torch.arange(len(posterior_mean), device=posterior_mean.device)
        if split == "train"
        else None
    )
    kernel = _kernel_coefficient(
        posterior_mean,
        bank_coefficient,
        state.memory_coefficients,
        state.variance,
        state.memory_dimensions,
        state.memory_neighbors,
        state.memory_temperature,
        state.memory_blend,
        query_indices,
    )
    neural_prediction = branches[2]
    neural_normalized = (
        neural_prediction - state.center[target][None]
    ) / state.scale[target][None]
    neural_proposal = neural_normalized @ torch.linalg.pinv(state.basis[target]).T
    prior_gate, _ = _adapter_gate(state, split, context, observed, target, branches)
    return kernel + prior_gate[:, None] * (neural_proposal - kernel)


@torch.inference_mode()
def cache_episode(
    state: DomainState,
    split: str,
    context: np.ndarray,
    target_pair: np.ndarray,
    proximal_multiplier: float,
    include_baseline: bool,
    receiver_fraction: float = 1.0,
    layout_seed: int = 0,
) -> ProximalEpisode:
    heterogeneous = receiver_fraction < 1.0
    branches = None
    if heterogeneous:
        masks = _receiver_masks(
            context,
            state.receiver_count,
            receiver_fraction,
            layout_seed,
        )
        observed, target = heterogeneous_indices(
            context,
            target_pair,
            masks,
            state.frequency_count,
            state.receiver_count,
            state.basis.device,
        )
        observed_basis = state.basis[observed]
        combined = torch.cat(
            (state.split_encoded[split][:, observed], state.train_encoded[:, observed]),
            dim=0,
        )
        coefficient, posterior_variance = solve_complex_posterior(
            observed_basis,
            combined,
            state.penalty,
            return_variance=True,
        )
        if posterior_variance is None:
            raise RuntimeError("posterior variance was requested")
        split_count = len(state.split_encoded[split])
        posterior_mean = coefficient[:split_count]
        bank_coefficient = coefficient[split_count:]
    else:
        (
            posterior_mean,
            bank_coefficient,
            posterior_variance,
            observed,
            target,
            branches,
        ) = _episode_branches(state, split, context, target_pair)
    observed_basis = state.basis[observed]
    observed_values = state.split_encoded[split][:, observed]
    penalty = proximal_multiplier * state.penalty
    basis128 = observed_basis.to(torch.complex128)
    precision = basis128.mH @ basis128 + torch.diag_embed(
        penalty.to(torch.float64)
    ).to(torch.complex128)
    data_projector = torch.linalg.solve(precision, basis128.mH)
    prior_projector = torch.linalg.solve(
        precision, torch.diag_embed(penalty.to(torch.float64)).to(torch.complex128)
    )
    data_coefficient = observed_values.to(torch.complex128) @ data_projector.T
    baseline = None
    if include_baseline:
        if heterogeneous:
            query_indices = (
                torch.arange(len(posterior_mean), device=posterior_mean.device)
                if split == "train"
                else None
            )
            baseline = _kernel_coefficient(
                posterior_mean,
                bank_coefficient,
                state.memory_coefficients,
                state.variance,
                state.memory_dimensions,
                state.memory_neighbors,
                state.memory_temperature,
                state.memory_blend,
                query_indices,
            )
        else:
            assert branches is not None
            baseline = _strong_baseline_coefficient(
                state,
                split,
                context,
                target_pair,
                posterior_mean,
                bank_coefficient,
                observed,
                target,
                branches,
            )
    return ProximalEpisode(
        domain=state.domain,
        budget=state.budget,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        latent_variance=state.variance,
        data_coefficient=data_coefficient.to(torch.complex64),
        prior_projector=prior_projector.to(torch.complex64),
        baseline_coefficient=baseline,
        target_basis=state.basis[target],
        target_center=state.center[target],
        target_scale=state.scale[target],
        truth=state.split_raw[split][:, target],
        observed_basis=observed_basis,
        observed_values=observed_values,
    )


def cache_split(
    states: dict[tuple[str, int], DomainState],
    split: str,
    repeats: int,
    seed: int,
    proximal_multiplier: float,
    include_baseline: bool,
    receiver_fraction: float = 1.0,
    layout_seed: int = 0,
) -> dict[tuple[str, int], list[ProximalEpisode]]:
    cached = {}
    for key, state in states.items():
        contexts, targets = _q3_events(
            state.source_count, state.frequency_count, state.budget, repeats, seed
        )
        cached[key] = [
            cache_episode(
                state,
                split,
                context,
                target,
                proximal_multiplier,
                include_baseline,
                receiver_fraction,
                layout_seed + event * 1_000_033,
            )
            for event, (context, target) in enumerate(
                zip(contexts, targets, strict=True)
            )
        ]
    return cached


def decode(coefficient: torch.Tensor, episode: ProximalEpisode) -> torch.Tensor:
    normalized = coefficient @ episode.target_basis.clone().T
    return episode.target_center.clone()[None] + episode.target_scale.clone()[None] * normalized


def _system_error(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
        truth, dim=-1
    ).clamp_min(1e-12)


@torch.inference_mode()
def evaluate_cached(
    model: UncertaintyConditionedOperatorPrior,
    cached: dict[tuple[str, int], list[ProximalEpisode]],
    blend_grid: list[float],
    batch_size: int,
    update_mode: str,
    prompt_safeguard: bool,
) -> dict[str, dict[str, np.ndarray | float]]:
    model.eval()
    result = {}
    for (domain, budget), episodes in cached.items():
        system_count = len(episodes[0].truth)
        totals = {
            blend: torch.zeros(system_count, device=episodes[0].truth.device)
            for blend in blend_grid
        }
        prompt_baseline = torch.zeros(system_count, device=episodes[0].truth.device)
        prompt_candidate = torch.zeros_like(prompt_baseline)
        for episode in episodes:
            if episode.baseline_coefficient is None:
                raise ValueError("evaluation cache requires the strong baseline")
            baseline = episode.baseline_coefficient
            baseline_prompt = baseline @ episode.observed_basis.T
            prompt_baseline += _system_error(baseline_prompt, episode.observed_values)
            for start in range(0, system_count, batch_size):
                stop = min(start + batch_size, system_count)
                model_input = (
                    baseline[start:stop]
                    if update_mode == "correction"
                    else episode.posterior_mean[start:stop]
                )
                learned = model(
                    model_input,
                    episode.posterior_variance,
                    episode.latent_variance,
                )
                if update_mode == "correction":
                    candidate = apply_proximal_correction(
                        baseline[start:stop],
                        learned,
                        episode.prior_projector,
                        episode.observed_basis,
                        episode.observed_values[start:stop],
                        prompt_safeguard,
                    )
                elif update_mode == "prior_mean":
                    candidate = assimilate_prior(
                        learned,
                        episode.data_coefficient[start:stop],
                        episode.prior_projector,
                    )
                else:
                    raise ValueError(f"unknown update mode {update_mode!r}")
                candidate_prompt = candidate @ episode.observed_basis.T
                prompt_candidate[start:stop] += _system_error(
                    candidate_prompt, episode.observed_values[start:stop]
                )
                for blend in blend_grid:
                    coefficient = baseline[start:stop] + blend * (
                        candidate - baseline[start:stop]
                    )
                    totals[blend][start:stop] += _system_error(
                        decode(coefficient, episode), episode.truth[start:stop]
                    )
        payload: dict[str, np.ndarray | float] = {}
        count = len(episodes)
        for blend, total in totals.items():
            payload[f"blend_{blend:g}"] = (total / count).cpu().numpy().astype(np.float64)
        payload["prompt_baseline_mean"] = float((prompt_baseline / count).mean())
        payload["prompt_candidate_mean"] = float((prompt_candidate / count).mean())
        result[f"{domain}_k{budget}"] = payload
    return result


def select_blend(
    evaluation: dict[str, dict[str, np.ndarray | float]],
    blend_grid: list[float],
) -> tuple[float, float, dict[str, float]]:
    anchors = {
        key: float(np.asarray(value["blend_0"]).mean())
        for key, value in evaluation.items()
    }
    best = (float("inf"), 0.0, {})
    for blend in blend_grid:
        ratios = {
            key: float(np.asarray(value[f"blend_{blend:g}"]).mean()) / anchors[key]
            for key, value in evaluation.items()
        }
        # Match the preregistered M0 rule: reward cross-domain mean gain while
        # allowing at most one percent degradation in the protected domain.
        worst = max(ratios.values())
        score = float(np.mean(list(ratios.values()))) + 10.0 * max(0.0, worst - 1.01)
        if score < best[0]:
            best = (score, blend, ratios)
    return best


def _machine_evaluation(
    evaluation: dict[str, dict[str, np.ndarray | float]],
    blend: float,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    machine = {}
    arrays = {}
    for offset, (key, payload) in enumerate(sorted(evaluation.items())):
        baseline = np.asarray(payload["blend_0"])
        candidate = np.asarray(payload[f"blend_{blend:g}"])
        arrays[key] = (baseline, candidate)
        machine[key] = {
            "baseline_mean": float(baseline.mean()),
            "candidate_mean": float(candidate.mean()),
            "blend": blend,
            "prompt_baseline_mean": float(payload["prompt_baseline_mean"]),
            "prompt_proximal_mean": float(payload["prompt_candidate_mean"]),
            **paired_summary(baseline, candidate, seed + offset),
        }
    return machine, arrays


def _build_states(config: dict, device: torch.device) -> dict[tuple[str, int], DomainState]:
    states = {}
    for domain, setting in config["domains"].items():
        for budget_text, specification in setting["budgets"].items():
            budget = int(budget_text)
            states[(domain, budget)] = _build_state(
                domain,
                budget,
                setting,
                specification,
                int(config["rank"]),
                int(config["value_rank"]),
                Path(config["adapter_checkpoint"]),
                device,
            )
    return states


def run(
    config_path: Path,
    output: Path,
    device_name: str,
    domains: tuple[str, ...] | None = None,
    steps: int | None = None,
    receiver_fraction: float | None = None,
    pretrain_steps: int | None = None,
    proximal_multiplier: float | None = None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if domains is not None:
        unknown = sorted(set(domains) - set(config["domains"]))
        if unknown:
            raise ValueError(f"unknown domains: {unknown}")
        config["domains"] = {
            domain: setting
            for domain, setting in config["domains"].items()
            if domain in domains
        }
    if steps is not None:
        config["steps"] = steps
    if receiver_fraction is not None:
        config["receiver_fraction"] = receiver_fraction
    if pretrain_steps is not None:
        config["denoising_pretrain_steps"] = pretrain_steps
    if proximal_multiplier is not None:
        config["proximal_multiplier"] = proximal_multiplier
    seed = int(config["seed"])
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    device = torch.device(device_name)
    states = _build_states(config, device)
    multiplier = float(config["proximal_multiplier"])
    update_mode = str(config.get("update_mode", "prior_mean"))
    prompt_safeguard = bool(config.get("prompt_safeguard", False))
    receiver_fraction = float(config.get("receiver_fraction", 1.0))
    layout_seed = int(config.get("layout_seed", 20260927))
    started = time.perf_counter()
    train_cache = cache_split(
        states,
        "train",
        int(config["train_repeats"]),
        int(config["train_prompt_seed"]),
        multiplier,
        update_mode == "correction",
        receiver_fraction,
        layout_seed,
    )
    validation_cache = cache_split(
        states,
        "validation",
        int(config["validation_repeats"]),
        int(config["validation_prompt_seed"]),
        multiplier,
        True,
        receiver_fraction,
        layout_seed + 100_000_007,
    )
    model = UncertaintyConditionedOperatorPrior(
        int(config["rank"]),
        int(config["width"]),
        int(config["depth"]),
        float(config["proposal_bound"]),
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    blend_grid = list(dict.fromkeys(float(value) for value in config["blend_grid"]))
    keys = sorted(train_cache)
    batch_size = int(config["batch_size"])
    pretrain_history = []
    pretrain_count = int(config.get("denoising_pretrain_steps", 0))
    if pretrain_count:
        latent_by_key = {
            key: state.train_encoded @ state.basis.conj()
            for key, state in states.items()
        }
        for step in range(1, pretrain_count + 1):
            losses = []
            for key in keys:
                latent = latent_by_key[key]
                index = torch.as_tensor(
                    generator.integers(
                        0, len(latent), size=min(batch_size, len(latent))
                    ),
                    dtype=torch.long,
                    device=device,
                )
                clean = latent[index]
                variance = states[key].variance
                standard_deviation = torch.sqrt(variance.clamp_min(1e-30))
                log_sigma = generator.uniform(np.log(0.05), np.log(0.75))
                sigma = float(np.exp(log_sigma))
                noise = torch.complex(
                    torch.randn_like(clean.real), torch.randn_like(clean.real)
                ) / np.sqrt(2.0)
                noisy = clean + sigma * standard_deviation[None] * noise
                posterior_variance = sigma**2 * variance
                correction = model(noisy, posterior_variance, variance)
                target = clean - noisy
                losses.append(
                    (
                        (correction - target).abs().square()
                        / variance.clamp_min(1e-30)[None]
                    ).mean()
                )
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % max(1, int(config["evaluation_interval"])) == 0:
                pretrain_history.append(
                    {"step": step, "denoising_loss": float(loss.detach())}
                )
    initial = evaluate_cached(
        model,
        validation_cache,
        blend_grid,
        int(config["eval_batch_size"]),
        update_mode,
        prompt_safeguard,
    )
    best_score, best_blend, best_ratios = select_blend(initial, blend_grid)
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    def all_ratios(evaluation: dict[str, dict[str, np.ndarray | float]]) -> dict[str, dict[str, float]]:
        return {
            f"{blend:g}": {
                key: float(np.asarray(value[f"blend_{blend:g}"]).mean())
                / float(np.asarray(value["blend_0"]).mean())
                for key, value in evaluation.items()
            }
            for blend in blend_grid
        }

    history = [
        {
            "step": 0,
            "selection_score": best_score,
            "selected_blend": best_blend,
            "ratios": best_ratios,
            "all_blend_ratios": all_ratios(initial),
        }
    ]
    for step in range(1, int(config["steps"]) + 1):
        model.train()
        losses = []
        for key in keys:
            episodes = train_cache[key]
            episode = episodes[int(generator.integers(len(episodes)))]
            index = torch.as_tensor(
                generator.integers(
                    0, len(episode.posterior_mean), size=min(batch_size, len(episode.posterior_mean))
                ),
                dtype=torch.long,
                device=device,
            )
            if update_mode == "correction" and episode.baseline_coefficient is None:
                raise ValueError("correction training requires cached strong baseline")
            baseline = episode.baseline_coefficient
            model_input = (
                baseline[index]
                if update_mode == "correction" and baseline is not None
                else episode.posterior_mean[index]
            )
            learned = model(
                model_input,
                episode.posterior_variance,
                episode.latent_variance,
            )
            if update_mode == "correction":
                assert baseline is not None
                coefficient = apply_proximal_correction(
                    baseline[index],
                    learned,
                    episode.prior_projector,
                    episode.observed_basis,
                    episode.observed_values[index],
                    False,
                )
            else:
                coefficient = assimilate_prior(
                    learned, episode.data_coefficient[index], episode.prior_projector
                )
            losses.append(
                _system_error(decode(coefficient, episode), episode.truth[index]).mean()
            )
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0 or step == int(config["steps"]):
            evaluation = evaluate_cached(
                model,
                validation_cache,
                blend_grid,
                int(config["eval_batch_size"]),
                update_mode,
                prompt_safeguard,
            )
            score, blend, ratios = select_blend(evaluation, blend_grid)
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "selection_score": score,
                    "selected_blend": blend,
                    "ratios": ratios,
                    "all_blend_ratios": all_ratios(evaluation),
                }
            )
            if score < best_score:
                best_score, best_blend, best_ratios = score, blend, ratios
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    validation = evaluate_cached(
        model,
        validation_cache,
        blend_grid,
        int(config["eval_batch_size"]),
        update_mode,
        prompt_safeguard,
    )
    validation_machine, validation_arrays = _machine_evaluation(
        validation, best_blend, seed
    )
    output.mkdir(parents=True, exist_ok=True)
    for key, (baseline, candidate) in validation_arrays.items():
        np.savez_compressed(
            output / f"{key.lower()}_validation_systems.npz",
            baseline_error=baseline,
            candidate_error=candidate,
        )
    torch.save(
        {
            "model": best_state,
            "config": config,
            "parameters": parameters,
            "selected_blend": best_blend,
        },
        output / "best.pt",
    )

    formal = {}
    formal_arrays: dict[str, dict[str, list[np.ndarray]]] = {}
    for prompt_seed in [int(value) for value in config["formal_test_seeds"]]:
        test_cache = cache_split(
            states,
            "test",
            int(config["formal_test_repeats"]),
            prompt_seed,
            multiplier,
            True,
            receiver_fraction,
            layout_seed + 200_000_033 + prompt_seed,
        )
        evaluation = evaluate_cached(
            model,
            test_cache,
            list(dict.fromkeys((0.0, best_blend))),
            int(config["eval_batch_size"]),
            update_mode,
            prompt_safeguard,
        )
        machine, arrays = _machine_evaluation(evaluation, best_blend, prompt_seed)
        formal[str(prompt_seed)] = machine
        for key, (baseline, candidate) in arrays.items():
            formal_arrays.setdefault(key, {"baseline": [], "candidate": []})
            formal_arrays[key]["baseline"].append(baseline)
            formal_arrays[key]["candidate"].append(candidate)
    aggregate = {}
    for offset, (key, arrays) in enumerate(sorted(formal_arrays.items())):
        baseline = np.stack(arrays["baseline"]).mean(axis=0)
        candidate = np.stack(arrays["candidate"]).mean(axis=0)
        np.savez_compressed(
            output / f"{key.lower()}_test_systems.npz",
            baseline_error=baseline,
            candidate_error=candidate,
        )
        aggregate[key] = {
            "baseline_mean": float(baseline.mean()),
            "candidate_mean": float(candidate.mean()),
            "selected_blend": best_blend,
            **paired_summary(baseline, candidate, seed + 100 + offset),
        }
    summary = {
        "parameters": parameters,
        "best_selection_score": best_score,
        "selected_blend": best_blend,
        "validation_ratios": best_ratios,
        "validation": validation_machine,
        "formal_by_seed": formal,
        "formal_aggregate": aggregate,
        "history": history,
        "denoising_pretrain_history": pretrain_history,
        "elapsed_seconds": time.perf_counter() - started,
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
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--receiver-fraction", type=float)
    parser.add_argument("--pretrain-steps", type=int)
    parser.add_argument("--proximal-multiplier", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.output,
                args.device,
                tuple(args.domains) if args.domains else None,
                args.steps,
                args.receiver_fraction,
                args.pretrain_steps,
                args.proximal_multiplier,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
