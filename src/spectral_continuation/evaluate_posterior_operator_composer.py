"""Evaluate a frozen posterior operator composer on unseen test systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .probabilistic_response_operator import paired_summary
from .train_posterior_operator_composer import (
    PosteriorOperatorComposer,
    _build_state,
    _cache_split,
    evaluate_cached,
)


def _new_model(config: dict, device: torch.device) -> PosteriorOperatorComposer:
    return PosteriorOperatorComposer(
        rank=int(config["rank"]),
        value_rank=int(config["value_rank"]),
        retrieval_dimensions=int(config["retrieval_dimensions"]),
        neighbors=int(config["neighbors"]),
        width=int(config["width"]),
        heads=int(config["heads"]),
        layers=int(config["layers"]),
        expansion=int(config["expansion"]),
        residual_bound=float(config["residual_bound"]),
    ).to(device)


def run(
    config_path: Path,
    checkpoint_path: Path,
    output: Path,
    device_name: str,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    states = {}
    for domain, setting in config["domains"].items():
        for budget_text, specification in setting["budgets"].items():
            key = (domain, int(budget_text))
            states[key] = _build_state(
                domain,
                key[1],
                setting,
                specification,
                int(config["rank"]),
                int(config["value_rank"]),
                Path(config["adapter_checkpoint"]),
                device,
            )
    learned = _new_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    learned.load_state_dict(checkpoint["model"])
    baseline = _new_model(config, device)
    seeds = [int(value) for value in config["formal_test_seeds"]]
    by_seed = {}
    arrays: dict[str, dict[str, list[np.ndarray]]] = {}
    for seed in seeds:
        cached = _cache_split(
            states, "test", int(config["formal_test_repeats"]), seed
        )
        learned_result = evaluate_cached(
            learned, cached, int(config["eval_batch_size"])
        )
        baseline_result = evaluate_cached(
            baseline, cached, int(config["eval_batch_size"])
        )
        seed_result = {}
        for key in learned_result:
            learned_error = learned_result[key]["_candidate"]
            baseline_error = baseline_result[key]["_candidate"]
            arrays.setdefault(key, {"learned": [], "baseline": []})
            arrays[key]["learned"].append(learned_error)
            arrays[key]["baseline"].append(baseline_error)
            seed_result[key] = {
                "baseline_mean": float(baseline_error.mean()),
                "learned_mean": float(learned_error.mean()),
                "gain": float(
                    (baseline_error.mean() - learned_error.mean())
                    / baseline_error.mean()
                ),
            }
        by_seed[str(seed)] = seed_result
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {}
    for offset, (key, values) in enumerate(sorted(arrays.items())):
        baseline_error = np.stack(values["baseline"]).mean(axis=0)
        learned_error = np.stack(values["learned"]).mean(axis=0)
        np.savez_compressed(
            output / f"{key.lower()}_test_systems.npz",
            baseline_error=baseline_error,
            learned_error=learned_error,
        )
        aggregate[key] = {
            "baseline_mean": float(baseline_error.mean()),
            "learned_mean": float(learned_error.mean()),
            **paired_summary(baseline_error, learned_error, 20260926 + offset),
        }
    result = {"seeds": seeds, "aggregate": aggregate, "by_seed": by_seed}
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, args.checkpoint, args.output, args.device), indent=2
        )
    )


if __name__ == "__main__":
    main()
