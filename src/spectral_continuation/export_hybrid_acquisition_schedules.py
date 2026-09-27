"""Export learned-proposal, analytic-certification acquisition schedules."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .evaluate_acquisition_baselines import random_order
from .train_continuous_acquisition_utility import (
    ContinuousAcquisitionUtility,
    _atomic_json,
    hybrid_shortlist_schedule,
)
from .train_posterior_operator_composer import _build_state


@torch.inference_mode()
def run(
    utility_config_path: Path,
    checkpoint_root: Path,
    output: Path,
    shortlist: int,
    device_name: str,
) -> dict:
    utility_config = json.loads(utility_config_path.read_text(encoding="utf-8"))
    base = json.loads(Path(utility_config["base_config"]).read_text(encoding="utf-8"))
    device = torch.device(device_name)
    payload = {
        "method": "continuous utility top-k proposal with analytic risk certification",
        "shortlist": shortlist,
        "maximum_budget": int(utility_config["maximum_budget"]),
        "domains": {},
        "mean_schedule_ms": {},
    }
    for domain in utility_config["domains"]:
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
        checkpoint = torch.load(
            checkpoint_root / domain.lower() / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model = ContinuousAcquisitionUtility(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        domain_payload = payload["domains"].setdefault(domain, {})
        timings = []
        for seed in utility_config["formal_schedule_seeds"]:
            seed_payload = domain_payload.setdefault(str(seed), {})
            for source in range(state.source_count):
                for frequency in range(state.frequency_count):
                    target = torch.tensor([source, frequency]).numpy()
                    ordered = random_order(state, target, int(seed))
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    schedule = hybrid_shortlist_schedule(
                        model,
                        state,
                        target,
                        ordered,
                        int(utility_config["maximum_budget"]),
                        shortlist,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    timings.append(1e3 * (time.perf_counter() - started))
                    seed_payload[f"{source}_{frequency}"] = schedule.tolist()
        payload["mean_schedule_ms"][domain] = float(sum(timings) / len(timings))
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shortlist", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.utility_config,
                args.checkpoint_root,
                args.output,
                args.shortlist,
                args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
