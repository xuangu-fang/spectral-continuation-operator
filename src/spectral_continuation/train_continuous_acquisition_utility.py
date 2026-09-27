"""Distill analytic wave-query design into a continuous set utility operator."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluate_acquisition_baselines import action_coordinates, random_order
from .evaluate_sequential_active_wave_query import candidate_target_risks
from .train_posterior_operator_composer import DomainState, _build_state


@dataclass(frozen=True)
class UtilityEvent:
    context: np.ndarray
    target: np.ndarray
    candidates: np.ndarray
    risk: np.ndarray
    heldout: np.ndarray


class ContinuousAcquisitionUtility(nn.Module):
    """Deep-set utility operator over continuous source--frequency coordinates."""

    def __init__(self, coordinate_dimensions: int, width: int, frequencies: int) -> None:
        super().__init__()
        self.coordinate_dimensions = coordinate_dimensions
        self.register_buffer(
            "bands", 2.0 ** torch.arange(frequencies, dtype=torch.float32)
        )
        feature_dimensions = coordinate_dimensions * (1 + 2 * frequencies)
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(feature_dimensions, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.score = nn.Sequential(
            nn.Linear(7 * width + 1, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def features(self, coordinate: torch.Tensor) -> torch.Tensor:
        angle = 2.0 * torch.pi * coordinate[..., None] * self.bands
        return torch.cat(
            (coordinate, torch.sin(angle).flatten(-2), torch.cos(angle).flatten(-2)),
            dim=-1,
        )

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        target: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        context_token = self.coordinate_encoder(self.features(context))
        mask = context_mask[..., None]
        context_mean = (context_token * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        context_max = context_token.masked_fill(~mask, -torch.inf).amax(dim=1)
        target_token = self.coordinate_encoder(self.features(target))
        action_token = self.coordinate_encoder(self.features(candidates))
        target_expanded = target_token[:, None].expand_as(action_token)
        mean_expanded = context_mean[:, None].expand_as(action_token)
        max_expanded = context_max[:, None].expand_as(action_token)
        budget = context_mask.sum(dim=1, keepdim=True).float() / context.shape[1]
        budget = budget[:, None].expand(-1, candidates.shape[1], -1)
        combined = torch.cat(
            (
                action_token,
                target_expanded,
                action_token - target_expanded,
                action_token * target_expanded,
                mean_expanded,
                max_expanded,
                action_token * mean_expanded,
                budget,
            ),
            dim=-1,
        )
        score = self.score(combined)[..., 0]
        return score.masked_fill(~candidate_mask, torch.inf)


def _heldout_actions(
    state: DomainState,
    fraction: int,
    offset: int,
    mode: str = "pair",
) -> np.ndarray:
    indices = np.arange(state.source_count * state.frequency_count)
    if mode == "pair":
        return (indices * 17 + 5 + offset) % fraction == 0
    if mode == "source_frequency":
        source = indices // state.frequency_count
        frequency = indices % state.frequency_count
        heldout_source = (source * 17 + 5 + offset) % fraction == 0
        heldout_frequency = (frequency * 17 + 5 + offset) % fraction == 0
        return heldout_source | heldout_frequency
    raise ValueError(f"unsupported heldout action mode {mode!r}")


@torch.inference_mode()
def build_events(
    state: DomainState,
    seeds: list[int],
    maximum_budget: int,
    heldout_fraction: int,
    heldout_offset: int,
    include_analytic_path: bool,
    heldout_mode: str = "pair",
) -> list[UtilityEvent]:
    coordinate = action_coordinates(state).astype(np.float32)
    heldout_actions = _heldout_actions(
        state, heldout_fraction, heldout_offset, heldout_mode
    )
    events = []
    for seed in seeds:
        for target_source in range(state.source_count):
            for target_frequency in range(state.frequency_count):
                target_pair = np.asarray([target_source, target_frequency], dtype=np.int64)
                target_index = target_source * state.frequency_count + target_frequency
                if heldout_actions[target_index]:
                    continue
                ordered = random_order(state, target_pair, seed)
                paths = ["random", "analytic"] if include_analytic_path else ["random"]
                for path in paths:
                    ordered_indices = (
                        ordered[:, 0] * state.frequency_count + ordered[:, 1]
                    )
                    initial_position = int(
                        np.flatnonzero(~heldout_actions[ordered_indices])[0]
                    )
                    selected = [ordered[initial_position]]
                    remaining = [
                        value
                        for position, value in enumerate(ordered)
                        if position != initial_position
                    ]
                    while len(selected) < maximum_budget:
                        candidates = np.stack(remaining)
                        indices = candidates[:, 0] * state.frequency_count + candidates[:, 1]
                        heldout = heldout_actions[indices]
                        if not (~heldout).any():
                            break
                        risks = candidate_target_risks(
                            state, target_pair, np.stack(selected), candidates
                        )
                        # Every event must retain both supervised and held-out actions.
                        if heldout.any() and (~heldout).any():
                            context_indices = np.asarray(
                                [
                                    item[0] * state.frequency_count + item[1]
                                    for item in selected
                                ]
                            )
                            events.append(
                                UtilityEvent(
                                    context=coordinate[context_indices],
                                    target=coordinate[target_index],
                                    candidates=coordinate[indices],
                                    risk=risks.astype(np.float32),
                                    heldout=heldout,
                                )
                            )
                        if path == "analytic":
                            supervised = np.flatnonzero(~heldout)
                            choice = int(supervised[np.argmin(risks[supervised])])
                        else:
                            choice = int(np.flatnonzero(~heldout)[0])
                        selected.append(remaining.pop(choice))
    return events


def collate_events(
    events: list[UtilityEvent],
    indices: np.ndarray,
    maximum_context: int,
    maximum_actions: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = len(indices)
    dimensions = events[0].target.shape[0]
    context = torch.zeros((batch, maximum_context, dimensions), device=device)
    context_mask = torch.zeros((batch, maximum_context), dtype=torch.bool, device=device)
    target = torch.zeros((batch, dimensions), device=device)
    candidates = torch.zeros((batch, maximum_actions, dimensions), device=device)
    candidate_mask = torch.zeros((batch, maximum_actions), dtype=torch.bool, device=device)
    heldout = torch.zeros_like(candidate_mask)
    label = torch.zeros((batch, maximum_actions), device=device)
    raw_risk = torch.zeros_like(label)
    for row, index in enumerate(indices):
        event = events[int(index)]
        context_count = len(event.context)
        action_count = len(event.candidates)
        context[row, :context_count] = torch.as_tensor(event.context, device=device)
        context_mask[row, :context_count] = True
        target[row] = torch.as_tensor(event.target, device=device)
        candidates[row, :action_count] = torch.as_tensor(event.candidates, device=device)
        candidate_mask[row, :action_count] = True
        heldout[row, :action_count] = torch.as_tensor(event.heldout, device=device)
        risk = torch.as_tensor(event.risk, device=device)
        log_risk = torch.log(risk.clamp_min(1e-20))
        normalized = (log_risk - log_risk.mean()) / log_risk.std().clamp_min(1e-6)
        label[row, :action_count] = normalized
        raw_risk[row, :action_count] = risk
    return {
        "context": context,
        "context_mask": context_mask,
        "target": target,
        "candidates": candidates,
        "candidate_mask": candidate_mask,
        "heldout": heldout,
        "label": label,
        "raw_risk": raw_risk,
    }


def utility_loss(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = batch["candidate_mask"] & ~batch["heldout"]
    square = (prediction - batch["label"]).square()
    regression = square[mask].mean()
    logits = (-prediction).masked_fill(~mask, -torch.inf)
    target_logits = (-batch["label"]).masked_fill(~mask, -torch.inf)
    target_probability = torch.softmax(target_logits, dim=-1)
    log_probability = torch.log_softmax(logits, dim=-1)
    ranking = -torch.where(
        mask, target_probability * log_probability, torch.zeros_like(log_probability)
    ).sum(dim=-1).mean()
    return regression + 0.1 * ranking


@torch.inference_mode()
def evaluate_regret(
    model: ContinuousAcquisitionUtility,
    events: list[UtilityEvent],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    heldout_regret = []
    all_regret = []
    exact = []
    maximum_context = max(len(event.context) for event in events)
    maximum_actions = max(len(event.candidates) for event in events)
    for start in range(0, len(events), batch_size):
        indices = np.arange(start, min(start + batch_size, len(events)))
        batch = collate_events(
            events, indices, maximum_context, maximum_actions, device
        )
        prediction = model(
            batch["context"],
            batch["context_mask"],
            batch["target"],
            batch["candidates"],
            batch["candidate_mask"],
        )
        for row, index in enumerate(indices):
            event = events[int(index)]
            count = len(event.risk)
            score = prediction[row, :count].cpu().numpy()
            best = int(np.argmin(event.risk))
            selected = int(np.argmin(score))
            all_regret.append(float(event.risk[selected] / event.risk[best] - 1.0))
            exact.append(float(selected == best))
            heldout_indices = np.flatnonzero(event.heldout)
            if len(heldout_indices):
                heldout_best = heldout_indices[np.argmin(event.risk[heldout_indices])]
                heldout_selected = heldout_indices[np.argmin(score[heldout_indices])]
                heldout_regret.append(
                    float(
                        event.risk[heldout_selected] / event.risk[heldout_best] - 1.0
                    )
                )
    return {
        "heldout_action_mean_regret": float(np.mean(heldout_regret)),
        "heldout_action_median_regret": float(np.median(heldout_regret)),
        "heldout_action_p90_regret": float(np.quantile(heldout_regret, 0.9)),
        "all_action_mean_regret": float(np.mean(all_regret)),
        "all_action_median_regret": float(np.median(all_regret)),
        "exact_action_rate": float(np.mean(exact)),
        "selection_regret": float(np.mean(heldout_regret) + np.mean(all_regret)),
    }


@torch.inference_mode()
def learned_schedule(
    model: ContinuousAcquisitionUtility,
    state: DomainState,
    target: np.ndarray,
    ordered: np.ndarray,
    maximum_budget: int,
) -> np.ndarray:
    coordinate = action_coordinates(state).astype(np.float32)
    selected = [ordered[0]]
    remaining = [value for value in ordered[1:]]
    while len(selected) < maximum_budget:
        context_indices = np.asarray(
            [value[0] * state.frequency_count + value[1] for value in selected]
        )
        candidate = np.stack(remaining)
        candidate_indices = candidate[:, 0] * state.frequency_count + candidate[:, 1]
        target_index = int(target[0]) * state.frequency_count + int(target[1])
        context_tensor = torch.as_tensor(
            coordinate[context_indices][None], device=state.basis.device
        )
        candidates_tensor = torch.as_tensor(
            coordinate[candidate_indices][None], device=state.basis.device
        )
        score = model(
            context_tensor,
            torch.ones(context_tensor.shape[:2], dtype=torch.bool, device=state.basis.device),
            torch.as_tensor(coordinate[target_index][None], device=state.basis.device),
            candidates_tensor,
            torch.ones(
                candidates_tensor.shape[:2], dtype=torch.bool, device=state.basis.device
            ),
        )[0]
        selected.append(remaining.pop(int(torch.argmin(score).cpu())))
    return np.stack(selected)


@torch.inference_mode()
def hybrid_shortlist_schedule(
    model: ContinuousAcquisitionUtility,
    state: DomainState,
    target: np.ndarray,
    ordered: np.ndarray,
    maximum_budget: int,
    shortlist: int,
) -> np.ndarray:
    """Use learned continuous utility to propose and analytic risk to certify."""

    coordinate = action_coordinates(state).astype(np.float32)
    selected = [ordered[0]]
    remaining = [value for value in ordered[1:]]
    while len(selected) < maximum_budget:
        context_indices = np.asarray(
            [value[0] * state.frequency_count + value[1] for value in selected]
        )
        candidate = np.stack(remaining)
        candidate_indices = candidate[:, 0] * state.frequency_count + candidate[:, 1]
        target_index = int(target[0]) * state.frequency_count + int(target[1])
        context_tensor = torch.as_tensor(
            coordinate[context_indices][None], device=state.basis.device
        )
        candidates_tensor = torch.as_tensor(
            coordinate[candidate_indices][None], device=state.basis.device
        )
        score = model(
            context_tensor,
            torch.ones(context_tensor.shape[:2], dtype=torch.bool, device=state.basis.device),
            torch.as_tensor(coordinate[target_index][None], device=state.basis.device),
            candidates_tensor,
            torch.ones(
                candidates_tensor.shape[:2], dtype=torch.bool, device=state.basis.device
            ),
        )[0]
        proposal = torch.topk(
            score, min(shortlist, len(remaining)), largest=False
        ).indices.cpu().numpy()
        proposal_risk = candidate_target_risks(
            state,
            target,
            np.stack(selected),
            candidate[proposal],
        )
        choice = int(proposal[int(np.argmin(proposal_risk))])
        selected.append(remaining.pop(choice))
    return np.stack(selected)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def train_domain(
    domain: str,
    state: DomainState,
    setting: dict,
    output: Path,
    device: torch.device,
) -> dict:
    train_events = build_events(
        state,
        [int(value) for value in setting["train_seeds"]],
        int(setting["maximum_budget"]),
        int(setting["heldout_action_fraction"]),
        int(setting.get("heldout_action_offset", 0)),
        bool(setting.get("include_analytic_path", True)),
        str(setting.get("heldout_action_mode", "pair")),
    )
    validation_events = build_events(
        state,
        [int(value) for value in setting["validation_seeds"]],
        int(setting["maximum_budget"]),
        int(setting["heldout_action_fraction"]),
        int(setting.get("heldout_action_offset", 0)),
        bool(setting.get("include_analytic_path", True)),
        str(setting.get("heldout_action_mode", "pair")),
    )
    model = ContinuousAcquisitionUtility(
        len(train_events[0].target),
        int(setting["width"]),
        int(setting["fourier_frequencies"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(setting["learning_rate"]),
        weight_decay=float(setting["weight_decay"]),
    )
    generator = np.random.default_rng(int(setting["seed"]))
    maximum_context = int(setting["maximum_budget"]) - 1
    maximum_actions = state.source_count * state.frequency_count
    best = None
    best_regret = float("inf")
    for step in range(int(setting["steps"]) + 1):
        if step % int(setting["evaluation_interval"]) == 0:
            model.eval()
            metrics = evaluate_regret(
                model, validation_events, device, int(setting["batch_size"])
            )
            if metrics["selection_regret"] < best_regret:
                best_regret = metrics["selection_regret"]
                best = {
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "metrics": metrics,
                    "step": step,
                }
        if step == int(setting["steps"]):
            break
        model.train()
        indices = generator.integers(
            0, len(train_events), size=int(setting["batch_size"])
        )
        batch = collate_events(
            train_events, indices, maximum_context, maximum_actions, device
        )
        prediction = model(
            batch["context"],
            batch["context_mask"],
            batch["target"],
            batch["candidates"],
            batch["candidate_mask"],
        )
        loss = utility_loss(prediction, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if best is None:
        raise RuntimeError("training did not create a checkpoint")
    model.load_state_dict(best["model"])
    model.eval()
    parameters = sum(value.numel() for value in model.parameters())
    checkpoint = {
        **best,
        "parameters": parameters,
        "model_config": {
            "coordinate_dimensions": len(train_events[0].target),
            "width": int(setting["width"]),
            "frequencies": int(setting["fourier_frequencies"]),
        },
    }
    domain_directory = output / domain.lower()
    domain_directory.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, domain_directory / "best.pt")

    # Compare one exhaustive analytic ranking with one network ranking.
    benchmark_event = validation_events[0]
    target_pair = np.asarray(
        divmod(
            int(np.argmin(np.square(action_coordinates(state) - benchmark_event.target).sum(axis=1))),
            state.frequency_count,
        )
    )
    context_indices = [
        int(np.argmin(np.square(action_coordinates(state) - value).sum(axis=1)))
        for value in benchmark_event.context
    ]
    context_pair = np.asarray(
        [divmod(index, state.frequency_count) for index in context_indices]
    )
    candidate_indices = [
        int(np.argmin(np.square(action_coordinates(state) - value).sum(axis=1)))
        for value in benchmark_event.candidates
    ]
    candidate_pair = np.asarray(
        [divmod(index, state.frequency_count) for index in candidate_indices]
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    candidate_target_risks(state, target_pair, context_pair, candidate_pair)
    torch.cuda.synchronize() if device.type == "cuda" else None
    analytic_ms = 1e3 * (time.perf_counter() - start)
    batch = collate_events(
        validation_events, np.asarray([0]), maximum_context, maximum_actions, device
    )
    for _ in range(20):
        model(
            batch["context"],
            batch["context_mask"],
            batch["target"],
            batch["candidates"],
            batch["candidate_mask"],
        )
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(100):
        model(
            batch["context"],
            batch["context_mask"],
            batch["target"],
            batch["candidates"],
            batch["candidate_mask"],
        )
    torch.cuda.synchronize() if device.type == "cuda" else None
    neural_ms = 1e3 * (time.perf_counter() - start) / 100.0
    result = {
        "domain": domain,
        "parameters": parameters,
        "train_events": len(train_events),
        "validation_events": len(validation_events),
        "best_step": int(best["step"]),
        **best["metrics"],
        "analytic_planner_ms": analytic_ms,
        "utility_operator_ms": neural_ms,
        "speedup": analytic_ms / neural_ms,
    }
    (domain_directory / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result, model


@torch.inference_mode()
def export_schedules(
    models: dict[str, ContinuousAcquisitionUtility],
    states: dict[str, DomainState],
    seeds: list[int],
    maximum_budget: int,
    output: Path,
) -> dict:
    payload = {
        "method": "continuous learned acquisition utility",
        "maximum_budget": maximum_budget,
        "domains": {},
    }
    for domain, model in models.items():
        state = states[domain]
        domain_payload = payload["domains"].setdefault(domain, {})
        for seed in seeds:
            seed_payload = domain_payload.setdefault(str(seed), {})
            for source in range(state.source_count):
                for frequency in range(state.frequency_count):
                    target = np.asarray([source, frequency], dtype=np.int64)
                    ordered = random_order(state, target, seed)
                    seed_payload[f"{source}_{frequency}"] = learned_schedule(
                        model, state, target, ordered, maximum_budget
                    ).tolist()
    _atomic_json(output, payload)
    return payload


def run(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
    device = torch.device(device_name)
    states = {}
    models = {}
    results = {}
    for domain, domain_setting in config["domains"].items():
        setting = base["domains"][domain]
        selection_budget = int(base["selection_budget"])
        state = _build_state(
            domain,
            selection_budget,
            setting,
            setting["budgets"][str(selection_budget)],
            int(base["rank"]),
            int(base["value_rank"]),
            Path(base["adapter_checkpoint"]),
            device,
        )
        states[domain] = state
        results[domain], models[domain] = train_domain(
            domain, state, domain_setting, output, device
        )
    export_schedules(
        models,
        states,
        [int(value) for value in config["formal_schedule_seeds"]],
        int(config["maximum_budget"]),
        output / "schedules.json",
    )
    summary = {"domains": results}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
