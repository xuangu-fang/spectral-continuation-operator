"""Benchmark the latent composer against its analytic posterior-kernel core."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .train_posterior_operator_composer import (
    PosteriorOperatorComposer,
    _build_state,
    _cache_split,
    _kernel_coefficient,
)


def _model(config: dict, checkpoint: Path, device: torch.device) -> PosteriorOperatorComposer:
    model = PosteriorOperatorComposer(
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
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def run(
    config_path: Path,
    checkpoint: Path,
    output: Path,
    domain: str,
    budget: int,
    repetitions: int,
    device_name: str,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    setting = config["domains"][domain]
    state = _build_state(
        domain,
        budget,
        setting,
        setting["budgets"][str(budget)],
        int(config["rank"]),
        int(config["value_rank"]),
        Path(config["adapter_checkpoint"]),
        device,
    )
    episode = _cache_split(
        {(domain, budget): state}, "test", 1, int(config["formal_test_seeds"][1])
    )[(domain, budget)][0]
    model = _model(config, checkpoint, device)

    def kernel() -> torch.Tensor:
        return _kernel_coefficient(
            episode.query_coefficient,
            episode.bank_coefficient,
            episode.memory_coefficients,
            episode.variance,
            episode.memory_dimensions,
            episode.memory_neighbors,
            episode.memory_temperature,
            episode.memory_blend,
        )

    def composer() -> torch.Tensor:
        return model(
            episode.query_coefficient,
            episode.posterior_variance,
            episode.bank_coefficient,
            episode.memory_coefficients,
            episode.variance,
            episode.target_features,
            episode.memory_dimensions,
            episode.memory_neighbors,
            episode.memory_temperature,
            episode.memory_blend,
            episode.neural_proposal,
            episode.prior_gate,
            episode.risk_features,
        )

    timings = {}
    with torch.inference_mode():
        for _ in range(20):
            kernel()
            composer()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for name, function in (("kernel", kernel), ("composer", composer)):
            started = time.perf_counter()
            for _ in range(repetitions):
                function()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings[name] = 1000.0 * (time.perf_counter() - started) / repetitions
    result = {
        "domain": domain,
        "budget": budget,
        "test_systems": len(episode.truth),
        "repetitions": repetitions,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "kernel_milliseconds": timings["kernel"],
        "composer_milliseconds": timings["composer"],
        "composer_over_kernel": timings["composer"] / timings["kernel"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", default="OpenFWI")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.checkpoint,
                args.output,
                args.domain,
                args.budget,
                args.repetitions,
                args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
