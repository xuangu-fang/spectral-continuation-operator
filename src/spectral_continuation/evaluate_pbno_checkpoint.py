"""Evaluate a trained PBNO checkpoint under Q2 or Q3 prompt protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import GlobalConfig, fit_global_subspace
from .train_prompted_bayesian_neural_operator import (
    PromptedBayesianNeuralOperator,
    evaluate,
    response_coordinates,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--regime", choices=("q2", "q3"), required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    base = GlobalConfig(**payload["base_config"])
    archive = load_cross_domain_archive(args.dataset)
    subspace = fit_global_subspace(archive, base.rank, base.normalization, args.device)
    model = PromptedBayesianNeuralOperator(
        response_coordinates(archive),
        subspace.basis[:, : base.rank],
        subspace.variance[: base.rank],
        base.strength,
        base.prior,
        int(config["hidden"]),
        bool(config.get("use_posterior_variance", False)),
        archive.responses.shape[-1],
        bool(config.get("use_prompt_set", False)),
        bool(config.get("use_receiver_residual", False)),
        bool(config["domains"][payload["domain"]].get("periodic_receiver", False)),
    ).to(args.device)
    model.load_state_dict(payload["model"])
    center = torch.as_tensor(subspace.center, device=args.device)
    scale = torch.as_tensor(subspace.scale, device=args.device)
    budgets = tuple(int(value) for value in args.budgets.split(","))
    result = {}
    for split in ("validation", "test", "ood"):
        if not np.any(archive.split == split):
            continue
        raw = torch.as_tensor(
            archive.responses[archive.split == split].reshape(
                int(np.sum(archive.split == split)), -1
            ),
            dtype=torch.complex64,
            device=args.device,
        )
        encoded = (raw - center[None]) / scale[None]
        values = evaluate(
            model,
            encoded,
            raw,
            center,
            scale,
            *archive.responses.shape[1:],
            budgets,
            int(
                config["validation_repeats"]
                if split == "validation"
                else config["test_repeats"]
            ),
            int(config["seed"]),
            regime=args.regime,
        )
        result[split] = {str(key): float(value.mean()) for key, value in values.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
