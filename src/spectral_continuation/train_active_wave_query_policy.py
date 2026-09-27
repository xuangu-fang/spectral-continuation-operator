"""Learn the next source--frequency query from an analytic wave-system belief.

The task starts with one strict-Q3 prompt and a known target query.  A policy
selects the second source--frequency probe before seeing its response.  The
selected response is then assimilated by the frozen Bayesian posterior-kernel
operator.  The policy is initialized to analytic posterior-variance reduction
and learns only a bounded, system-conditioned utility residual.
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

from .global_bayesian_response_operator import solve_complex_posterior
from .probabilistic_response_operator import paired_summary
from .train_prompted_bayesian_neural_operator import fourier_features
from .train_query_conditioned_posterior_metric import (
    MetricEpisode,
    MetricState,
    build_state,
    decode,
    posterior_kernel,
    relative_error,
)


@dataclass
class ActiveEvent:
    belief: torch.Tensor
    action_features: torch.Tensor
    analytic_risk: torch.Tensor
    downstream_error: torch.Tensor


class ActiveWaveQueryPolicy(nn.Module):
    """Belief-conditioned residual around analytic information gain."""

    def __init__(
        self,
        belief_dimensions: int,
        action_features: int,
        width: int,
        residual_bound: float,
    ) -> None:
        super().__init__()
        self.belief_dimensions = belief_dimensions
        self.residual_bound = residual_bound
        self.belief_encoder = nn.Sequential(
            nn.Linear(2 * belief_dimensions, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_features, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.score = nn.Sequential(
            nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(
        self,
        belief: torch.Tensor,
        action_features: torch.Tensor,
        analytic_risk: torch.Tensor,
    ) -> torch.Tensor:
        belief = belief.clone()
        action_features = action_features.clone()
        analytic_risk = analytic_risk.clone()
        belief_token = self.belief_encoder(belief)
        action_token = self.action_encoder(action_features)
        left = belief_token[:, None].expand(-1, len(action_token), -1)
        right = action_token[None].expand(len(belief_token), -1, -1)
        residual = self.residual_bound * torch.tanh(
            self.score(torch.cat((left, right, left * right), dim=-1))[..., 0]
        )
        normalized_risk = (analytic_risk - analytic_risk.min()) / analytic_risk.std().clamp_min(
            1e-6
        )
        analytic = -normalized_risk
        return analytic[None] + residual


def _candidate_order(
    source_count: int,
    frequency_count: int,
    target_source: int,
    target_frequency: int,
    seed: int,
) -> np.ndarray:
    candidates = np.asarray(
        [
            (source, frequency)
            for source in range(source_count)
            for frequency in range(frequency_count)
            if source != target_source and frequency != target_frequency
        ],
        dtype=np.int64,
    )
    target_seed = seed + 1_000_003 * target_source + 10_007 * target_frequency
    return candidates[np.random.default_rng(target_seed).permutation(len(candidates))]


def _action_features(
    state: MetricState,
    initial: np.ndarray,
    candidates: np.ndarray,
    target: np.ndarray,
) -> torch.Tensor:
    receiver = 0
    initial_index = (int(initial[0]) * state.frequency_count + int(initial[1])) * state.receiver_count + receiver
    target_index = (int(target[0]) * state.frequency_count + int(target[1])) * state.receiver_count + receiver
    candidate_index = (
        (candidates[:, 0] * state.frequency_count + candidates[:, 1]) * state.receiver_count
        + receiver
    )
    initial_coordinate = state.coordinates[initial_index, :3]
    target_coordinate = state.coordinates[target_index, :3]
    candidate_coordinate = state.coordinates[
        torch.as_tensor(candidate_index, dtype=torch.long, device=state.basis.device), :3
    ]
    initial_feature = fourier_features(initial_coordinate[None]).expand(len(candidates), -1)
    target_feature = fourier_features(target_coordinate[None]).expand(len(candidates), -1)
    candidate_feature = fourier_features(candidate_coordinate)
    relative = torch.cat(
        (
            candidate_coordinate - initial_coordinate[None],
            candidate_coordinate - target_coordinate[None],
        ),
        dim=-1,
    )
    return torch.cat((initial_feature, target_feature, candidate_feature, relative), dim=-1)


@torch.inference_mode()
def cache_active_split(
    state: MetricState,
    split: str,
    prompt_seed: int,
    setting: dict,
    value_rank: int,
    belief_dimensions: int,
) -> list[ActiveEvent]:
    if split == "train":
        encoded = state.train_encoded
        raw = state.train_raw
    else:
        encoded = state.split_encoded[split]
        raw = state.split_raw[split]
    receiver = torch.arange(state.receiver_count, device=state.basis.device)
    fixed_weight = torch.zeros_like(state.variance)
    fixed_weight[: int(setting["latent_dimensions"])] = 1.0
    events = []
    for target_source in range(state.source_count):
        for target_frequency in range(state.frequency_count):
            order = _candidate_order(
                state.source_count,
                state.frequency_count,
                target_source,
                target_frequency,
                prompt_seed,
            )
            initial = order[0]
            candidates = order[1:]
            initial_query = int(initial[0] * state.frequency_count + initial[1])
            initial_observed = initial_query * state.receiver_count + receiver
            initial_basis = state.basis[initial_observed]
            initial_combined = torch.cat(
                (encoded[:, initial_observed], state.train_encoded[:, initial_observed]), dim=0
            )
            initial_coefficient, _ = solve_complex_posterior(
                initial_basis, initial_combined, state.penalty
            )
            split_count = len(encoded)
            whitened = initial_coefficient[:split_count, :belief_dimensions] / torch.sqrt(
                state.variance[:belief_dimensions]
            ).clamp_min(1e-15)[None]
            belief = torch.cat((whitened.real, whitened.imag), dim=-1)
            target_pair = np.asarray([target_source, target_frequency], dtype=np.int64)
            target_query = target_source * state.frequency_count + target_frequency
            target = target_query * state.receiver_count + receiver
            target_basis = state.basis[target]
            target_scaled = target_basis.to(torch.complex128) * state.scale[target, None].double()
            target_gram = target_scaled.mH @ target_scaled
            action_risk = []
            action_error = []
            for candidate in candidates:
                context = np.stack((initial, candidate))
                query = torch.as_tensor(
                    context[:, 0] * state.frequency_count + context[:, 1],
                    dtype=torch.long,
                    device=state.basis.device,
                )
                observed = (query[:, None] * state.receiver_count + receiver[None]).reshape(-1)
                observed_basis = state.basis[observed]
                combined = torch.cat(
                    (encoded[:, observed], state.train_encoded[:, observed]), dim=0
                )
                coefficient, _ = solve_complex_posterior(
                    observed_basis, combined, state.penalty
                )
                query_coefficient = coefficient[:split_count]
                bank_coefficient = coefficient[split_count:]
                query_indices = (
                    torch.arange(split_count, device=state.basis.device)
                    if split == "train"
                    else None
                )
                prediction_coefficient = posterior_kernel(
                    query_coefficient,
                    bank_coefficient,
                    state.memory_coefficients,
                    state.variance,
                    fixed_weight,
                    int(setting["neighbors"]),
                    float(setting["temperature"]),
                    torch.ones((), device=state.basis.device),
                    float(setting["blend"]),
                    value_rank,
                    query_indices,
                )
                episode = MetricEpisode(
                    query_coefficient,
                    bank_coefficient,
                    target_basis,
                    state.center[target],
                    state.scale[target],
                    state.coordinates[target[0]],
                    raw[:, target],
                    query_indices,
                )
                action_error.append(
                    relative_error(decode(prediction_coefficient, episode), episode.truth)
                )
                basis128 = observed_basis.to(torch.complex128)
                precision = basis128.mH @ basis128 + torch.diag(state.penalty.double()).to(
                    torch.complex128
                )
                action_risk.append(
                    torch.real(torch.trace(torch.linalg.solve(precision, target_gram))).float()
                )
            events.append(
                ActiveEvent(
                    belief=belief,
                    action_features=_action_features(state, initial, candidates, target_pair),
                    analytic_risk=torch.stack(action_risk),
                    downstream_error=torch.stack(action_error, dim=1),
                )
            )
    return events


@torch.inference_mode()
def evaluate(
    model: ActiveWaveQueryPolicy,
    events: list[ActiveEvent],
) -> dict[str, np.ndarray]:
    model.eval()
    systems = len(events[0].belief)
    totals = {
        "random": torch.zeros(systems, device=events[0].belief.device),
        "analytic": torch.zeros(systems, device=events[0].belief.device),
        "learned": torch.zeros(systems, device=events[0].belief.device),
        "oracle": torch.zeros(systems, device=events[0].belief.device),
    }
    system = torch.arange(systems, device=events[0].belief.device)
    for event in events:
        score = model(event.belief, event.action_features, event.analytic_risk)
        learned_index = score.argmax(dim=-1)
        analytic_index = event.analytic_risk.argmin().expand(systems)
        totals["random"] += event.downstream_error[:, 0]
        totals["analytic"] += event.downstream_error[system, analytic_index]
        totals["learned"] += event.downstream_error[system, learned_index]
        totals["oracle"] += event.downstream_error.min(dim=-1).values
    return {
        key: (value / len(events)).cpu().numpy().astype(np.float64)
        for key, value in totals.items()
    }


def run_domain(
    domain: str,
    config: dict,
    output: Path,
    device_name: str,
    steps_override: int | None,
) -> dict:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    device = torch.device(device_name)
    setting = config["domains"][domain]
    state = build_state(setting, int(config["rank"]), int(config["value_rank"]), device)
    belief_dimensions = int(setting["belief_dimensions"])
    started = time.perf_counter()
    train_prompt_seeds = config.get("train_prompt_seeds", [config["train_prompt_seed"]])
    train_events = []
    for prompt_seed in train_prompt_seeds:
        train_events.extend(
            cache_active_split(
                state,
                "train",
                int(prompt_seed),
                setting,
                int(config["value_rank"]),
                belief_dimensions,
            )
        )
    validation_events = cache_active_split(
        state,
        "validation",
        int(config["validation_prompt_seed"]),
        setting,
        int(config["value_rank"]),
        belief_dimensions,
    )
    model = ActiveWaveQueryPolicy(
        belief_dimensions,
        train_events[0].action_features.shape[-1],
        int(config["width"]),
        float(config["residual_bound"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    initial = evaluate(model, validation_events)
    best_score = float(initial["learned"].mean())
    best_step = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [
        {
            "step": 0,
            **{f"validation_{key}": float(value.mean()) for key, value in initial.items()},
        }
    ]
    steps = int(config["steps"] if steps_override is None else steps_override)
    batch_size = int(config["batch_size"])
    for step in range(1, steps + 1):
        model.train()
        event = train_events[int(generator.integers(len(train_events)))]
        index = torch.as_tensor(
            generator.choice(
                len(event.belief), size=min(batch_size, len(event.belief)), replace=False
            ),
            dtype=torch.long,
            device=device,
        )
        score = model(event.belief[index], event.action_features, event.analytic_risk)
        error = event.downstream_error[index]
        normalized_error = (error - error.min(dim=-1, keepdim=True).values) / error.std(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        probability = torch.softmax(score, dim=-1)
        loss = (probability * normalized_error).sum(dim=-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0 or step == steps:
            validation = evaluate(model, validation_events)
            score_value = float(validation["learned"].mean())
            history.append(
                {
                    "step": step,
                    "train_expected_regret": float(loss.detach()),
                    **{
                        f"validation_{key}": float(value.mean())
                        for key, value in validation.items()
                    },
                }
            )
            if score_value < best_score:
                best_score = score_value
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    validation = evaluate(model, validation_events)
    test_results = []
    for prompt_seed in config["test_prompt_seeds"]:
        events = cache_active_split(
            state,
            "test",
            int(prompt_seed),
            setting,
            int(config["value_rank"]),
            belief_dimensions,
        )
        test_results.append((int(prompt_seed), evaluate(model, events)))
    test = {
        key: np.mean([item[1][key] for item in test_results], axis=0)
        for key in ("random", "analytic", "learned", "oracle")
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state, "config": config, "domain": domain, "best_step": best_step},
        output / "best.pt",
    )
    np.savez_compressed(output / "system_metrics.npz", **test)
    learned_stats = paired_summary(test["random"], test["learned"], seed)
    analytic_stats = paired_summary(test["random"], test["analytic"], seed + 1)
    summary = {
        "domain": domain,
        "task": "strict-Q3 active second source-frequency query",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_step": best_step,
        "validation": {key: float(value.mean()) for key, value in validation.items()},
        "test": {
            **{f"{key}_mean": float(value.mean()) for key, value in test.items()},
            "analytic_gain_vs_random": float(1.0 - test["analytic"].mean() / test["random"].mean()),
            "learned_gain_vs_random": float(1.0 - test["learned"].mean() / test["random"].mean()),
            "learned_gain_vs_analytic": float(1.0 - test["learned"].mean() / test["analytic"].mean()),
            "oracle_gain_vs_random": float(1.0 - test["oracle"].mean() / test["random"].mean()),
            "learned_paired": learned_stats,
            "analytic_paired": analytic_stats,
            "by_prompt_seed": [
                {
                    "seed": prompt_seed,
                    **{f"{key}_mean": float(value.mean()) for key, value in result.items()},
                }
                for prompt_seed, result in test_results
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
    print(
        json.dumps(
            run_domain(args.domain, config, args.output, args.device, args.steps), indent=2
        )
    )


if __name__ == "__main__":
    main()
