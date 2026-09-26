"""Learn a functional prompt metric with listwise field-response supervision."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit_heterogeneous_prompt_headroom import _build_domain, _relative_error
from .probabilistic_response_operator import paired_summary
from .train_functional_prompt_reranker import (
    FunctionalPromptReranker,
    RetrievalEpisode,
    _cache_episodes,
    _decode,
    _model,
    _rank_correlation,
)


def _candidate_error(
    episode: RetrievalEpisode, index: torch.Tensor
) -> torch.Tensor:
    candidates = episode.candidate_targets[episode.candidate_order[index]]
    raw = episode.target_center.clone()[None, None] + episode.target_scale.clone()[
        None, None
    ] * candidates
    truth = episode.truth[index, None]
    return torch.linalg.vector_norm(raw - truth, dim=-1) / torch.linalg.vector_norm(
        truth, dim=-1
    ).clamp_min(1e-12)


def _listwise_loss(
    score: torch.Tensor,
    candidate_error: torch.Tensor,
    label_temperature: float,
) -> torch.Tensor:
    gap = candidate_error - candidate_error.min(dim=-1, keepdim=True).values
    scale = torch.quantile(gap, 0.5, dim=-1, keepdim=True).clamp_min(1e-3)
    label = torch.softmax(-gap / (label_temperature * scale), dim=-1)
    return -(label * torch.log_softmax(score, dim=-1)).sum(dim=-1).mean()


@torch.inference_mode()
def evaluate_grid(
    model: FunctionalPromptReranker,
    cached: dict[float, list[RetrievalEpisode]],
    batch_size: int,
    temperatures: tuple[float, ...],
    blends: tuple[float, ...],
) -> dict:
    model.eval()
    result = {}
    strategy_arrays: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    diagnostics = {}
    for fraction, episodes in cached.items():
        systems = len(episodes[0].truth)
        baseline_total = torch.zeros(systems, device=episodes[0].truth.device)
        errors = {
            f"t{temperature:g}_b{blend:g}": torch.zeros_like(baseline_total)
            for temperature in temperatures
            for blend in blends
        }
        entropy_total = torch.zeros_like(baseline_total)
        top1_total = torch.zeros_like(baseline_total)
        oracle_total = torch.zeros_like(baseline_total)
        for episode in episodes:
            baseline = _decode(episode.baseline_normalized, episode)
            baseline_total += _relative_error(baseline, episode.truth)
            for start in range(0, systems, batch_size):
                stop = min(start + batch_size, systems)
                index = torch.arange(start, stop, device=episode.truth.device)
                score, _ = model.score_candidates(
                    episode.prompt_values[index],
                    episode.bank_prompt_values,
                    episode.prompt_coordinates,
                    episode.candidate_order[index],
                    episode.candidate_logits[index],
                )
                candidate_error = _candidate_error(episode, index)
                top = score.argmax(dim=-1)
                top1_total[index] += candidate_error.gather(1, top[:, None]).squeeze(1)
                oracle_total[index] += candidate_error.min(dim=-1).values
                candidates = episode.candidate_targets[
                    episode.candidate_order[index]
                ]
                for temperature in temperatures:
                    weight = torch.softmax(score / temperature, dim=-1)
                    retrieved = (weight[..., None] * candidates).sum(dim=1)
                    if temperature == temperatures[0]:
                        entropy_total[index] += -(
                            weight * weight.clamp_min(1e-12).log()
                        ).sum(dim=-1) / np.log(weight.shape[-1])
                    for blend in blends:
                        normalized = episode.baseline_normalized[index] + blend * (
                            retrieved - episode.baseline_normalized[index]
                        )
                        prediction = _decode(normalized, episode)
                        errors[f"t{temperature:g}_b{blend:g}"][index] += (
                            _relative_error(prediction, episode.truth[index])
                        )
        baseline = (baseline_total / len(episodes)).cpu().numpy().astype(np.float64)
        entropy = (entropy_total / len(episodes)).cpu().numpy().astype(np.float64)
        key = f"r{round(100 * fraction):03d}"
        result[key] = {"baseline_mean": float(baseline.mean()), "strategies": {}}
        for strategy, total in errors.items():
            candidate = (total / len(episodes)).cpu().numpy().astype(np.float64)
            result[key]["strategies"][strategy] = {
                "candidate_mean": float(candidate.mean()),
                **paired_summary(baseline, candidate, 20260926),
            }
            strategy_arrays.setdefault(strategy, []).append((baseline, candidate))
        top1 = (top1_total / len(episodes)).cpu().numpy().astype(np.float64)
        oracle = (oracle_total / len(episodes)).cpu().numpy().astype(np.float64)
        diagnostics[key] = {
            "top1_candidate_error": float(top1.mean()),
            "oracle_candidate_error": float(oracle.mean()),
            "top1_oracle_regret": float((top1 - oracle).mean()),
            "uncertainty_error_spearman": _rank_correlation(entropy, top1),
        }
    selection = {}
    for strategy, arrays in strategy_arrays.items():
        ratios = [candidate.mean() / baseline.mean() for baseline, candidate in arrays]
        selection[strategy] = float(max(ratios) + 0.25 * np.mean(ratios))
    selected = min(selection, key=selection.get)
    return {
        "selected_strategy": selected,
        "selection_score": selection[selected],
        "fractions": result,
        "diagnostics": diagnostics,
    }


def _selected_summary(evaluation: dict) -> dict:
    selected = evaluation["selected_strategy"]
    return {
        key: {
            "baseline_mean": value["baseline_mean"],
            **value["strategies"][selected],
            **evaluation["diagnostics"][key],
        }
        for key, value in evaluation["fractions"].items()
    }


def train_domain(
    name: str,
    payload: dict,
    config: dict,
    output: Path,
    device: torch.device,
) -> dict:
    torch.manual_seed(int(config["seed"]))
    generator = np.random.default_rng(int(config["seed"]))
    domain = _build_domain(
        name,
        payload,
        int(config["budget"]),
        int(config["rank"]),
        int(config["value_rank"]),
        device,
    )
    fractions = tuple(float(value) for value in config["receiver_fractions"])
    temperatures = tuple(float(value) for value in config["retrieval_temperatures"])
    blends = tuple(float(value) for value in config["retrieval_blends"])
    started = time.perf_counter()
    train_cache = _cache_episodes(
        domain,
        "train",
        fractions,
        int(config["train_repeats"]),
        int(config["seed"]),
        int(config["layout_seed"]),
        int(config["candidates"]),
    )
    validation_cache = _cache_episodes(
        domain,
        "validation",
        fractions,
        int(config["validation_repeats"]),
        int(config["selection_seed"]),
        int(config["validation_layout_seed"]),
        int(config["candidates"]),
    )
    test_cache = _cache_episodes(
        domain,
        "test",
        fractions,
        int(config["test_repeats"]),
        int(config["test_seed"]),
        int(config["test_layout_seed"]),
        int(config["candidates"]),
    )
    model = _model(config, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_score = float("inf")
    best_state = None
    best_strategy = None
    history = []
    batch_size = int(config["batch_size"])
    for step in range(1, int(config["steps"]) + 1):
        fraction = fractions[int(generator.integers(len(fractions)))]
        episode = train_cache[fraction][
            int(generator.integers(len(train_cache[fraction])))
        ]
        index = torch.as_tensor(
            generator.integers(0, len(episode.truth), size=batch_size),
            dtype=torch.long,
            device=device,
        )
        score, _ = model.score_candidates(
            episode.prompt_values[index],
            episode.bank_prompt_values,
            episode.prompt_coordinates,
            episode.candidate_order[index],
            episode.candidate_logits[index],
            index,
        )
        loss = _listwise_loss(
            score,
            _candidate_error(episode, index),
            float(config["label_temperature"]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0:
            validation = evaluate_grid(
                model,
                validation_cache,
                int(config["eval_batch_size"]),
                temperatures,
                blends,
            )
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "selection_score": validation["selection_score"],
                    "selected_strategy": validation["selected_strategy"],
                }
            )
            if validation["selection_score"] < best_score:
                best_score = float(validation["selection_score"])
                best_strategy = str(validation["selected_strategy"])
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is None or best_strategy is None:
        raise RuntimeError("no checkpoint was evaluated")
    model.load_state_dict(best_state)
    validation = evaluate_grid(
        model,
        validation_cache,
        int(config["eval_batch_size"]),
        temperatures,
        blends,
    )
    test = evaluate_grid(
        model,
        test_cache,
        int(config["eval_batch_size"]),
        temperatures,
        blends,
    )
    test["selected_strategy"] = best_strategy
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": config, "domain": name}, output / "best.pt")
    summary = {
        "domain": name,
        "parameters": parameters,
        "best_selection_score": best_score,
        "selected_strategy": best_strategy,
        "seconds": time.perf_counter() - started,
        "validation": _selected_summary(validation),
        "test": _selected_summary(test),
        "history": history,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    result = {
        name: train_domain(name, payload, config, output / name.lower(), device)
        for name, payload in config["domains"].items()
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
