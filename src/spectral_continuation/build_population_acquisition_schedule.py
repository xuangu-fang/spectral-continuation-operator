"""Build a validation-only fixed acquisition schedule for each target query."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate_acquisition_baselines import random_order
from .evaluate_sequential_active_wave_query import adapter_error
from .train_posterior_operator_composer import _build_state


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@torch.inference_mode()
def greedy_population_schedule(
    state,
    target: np.ndarray,
    ordered: np.ndarray,
    maximum_budget: int,
) -> np.ndarray:
    """Minimize validation-system mean downstream error without test responses."""

    selected = [ordered[0]]
    remaining = [value for value in ordered[1:]]
    while len(selected) < maximum_budget:
        errors = []
        for candidate in remaining:
            context = np.stack((*selected, candidate))
            errors.append(
                float(adapter_error(state, "validation", context, target).mean().cpu())
            )
        selected.append(remaining.pop(int(np.argmin(errors))))
    return np.stack(selected)


@torch.inference_mode()
def run(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
    else:
        payload = {
            "method": "validation-only population-optimal greedy schedule",
            "selection_split": "validation",
            "maximum_budget": max(int(value) for value in config["active_budgets"]),
            "domains": {},
            "timing_seconds": {},
        }
    device = torch.device(device_name)
    maximum_budget = int(payload["maximum_budget"])
    for domain, setting in config["domains"].items():
        started = time.perf_counter()
        selection_budget = int(config["selection_budget"])
        state = _build_state(
            domain,
            selection_budget,
            setting,
            setting["budgets"][str(selection_budget)],
            int(config["rank"]),
            int(config["value_rank"]),
            Path(config["adapter_checkpoint"]),
            device,
        )
        domain_payload = payload["domains"].setdefault(domain, {})
        for seed in config["formal_test_seeds"]:
            seed_payload = domain_payload.setdefault(str(seed), {})
            for target_source in range(state.source_count):
                for target_frequency in range(state.frequency_count):
                    key = f"{target_source}_{target_frequency}"
                    if key in seed_payload:
                        continue
                    target = np.asarray([target_source, target_frequency], dtype=np.int64)
                    ordered = random_order(state, target, int(seed))
                    schedule = greedy_population_schedule(
                        state, target, ordered, maximum_budget
                    )
                    seed_payload[key] = schedule.tolist()
                    _atomic_json(output, payload)
        payload["timing_seconds"][domain] = float(time.perf_counter() - started)
        _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args.config, args.output, args.device)
    print(json.dumps({"timing_seconds": result["timing_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
