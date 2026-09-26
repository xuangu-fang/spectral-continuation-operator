"""Prompt-only leave-one-probe-out routing among PBNO prior adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import GlobalConfig, GlobalQ3Evaluator, fit_global_subspace
from .posterior_memory_response_operator import (
    MemoryConfig,
    predict_memory_config_batched,
)
from .probabilistic_response_operator import paired_summary
from .train_prompted_bayesian_neural_operator import (
    PromptedBayesianNeuralOperator,
    response_coordinates,
)
from .train_prompted_bayesian_neural_operator import (
    predict as predict_neural,
)


def relative_error(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
        truth, dim=-1
    ).clamp_min(1e-12)


def load_neural(
    checkpoint_path: Path,
    archive,
    device: str,
) -> tuple[PromptedBayesianNeuralOperator, torch.Tensor, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base_payload = checkpoint["base_config"]
    base = GlobalConfig(**base_payload)
    subspace = fit_global_subspace(archive, base.rank, base.normalization, device)
    config = checkpoint["config"]
    train_flat = archive.responses[archive.split == "train"].reshape(
        int(np.sum(archive.split == "train")), -1
    )
    residual_scale = np.sqrt(
        np.mean(np.abs(train_flat - subspace.center[None]) ** 2, axis=0)
    ).astype(np.float32)
    positive = residual_scale[residual_scale > 0]
    floor = 0.02 * float(np.median(positive)) if len(positive) else 1.0
    residual_scale = np.maximum(residual_scale, max(floor, 1e-12))
    receiver_count = archive.responses.shape[-1]
    domain_setting = config["domains"][checkpoint["domain"]]
    model = PromptedBayesianNeuralOperator(
        response_coordinates(archive),
        subspace.basis[:, : base.rank],
        subspace.variance[: base.rank],
        base.strength,
        base.prior,
        int(config["hidden"]),
        bool(config.get("use_posterior_variance", False)),
        receiver_count,
        bool(config.get("use_prompt_set", False)),
        bool(config.get("use_receiver_residual", False)),
        bool(domain_setting.get("periodic_receiver", False)),
        residual_scale,
        float(config.get("receiver_residual_bound", 0.05)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    center = torch.as_tensor(subspace.center, device=device)
    scale = torch.as_tensor(subspace.scale, device=device)
    return model, center, scale, torch.as_tensor(subspace.basis, device=device)


@torch.inference_mode()
def neural_event_predictions(
    model: PromptedBayesianNeuralOperator,
    encoded: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    observed: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    predictions = []
    for event in range(len(observed)):
        predictions.append(
            predict_neural(model, encoded, center, scale, observed[event], target[event])
        )
    return torch.stack(predictions)


@torch.inference_mode()
def route_split(
    evaluator: GlobalQ3Evaluator,
    operator: GlobalConfig,
    memory: MemoryConfig,
    model: PromptedBayesianNeuralOperator,
    neural_center: torch.Tensor,
    neural_scale: torch.Tensor,
    raw: torch.Tensor,
    holdout_count: int,
) -> dict[str, np.ndarray | float]:
    encoded_neural = (raw - neural_center[None]) / neural_scale[None]
    global_prediction = evaluator.predict(operator)
    kernel_prediction = predict_memory_config_batched(evaluator, operator, memory)
    neural_prediction = neural_event_predictions(
        model,
        encoded_neural,
        neural_center,
        neural_scale,
        evaluator.observed,
        evaluator.target,
    )
    branch_predictions = torch.stack(
        (global_prediction, kernel_prediction, neural_prediction), dim=0
    )
    branch_error = relative_error(branch_predictions, evaluator.truth[None])

    budget = evaluator.observed.shape[1] // evaluator.receiver_count
    if budget < 2:
        raise ValueError("leave-one-probe-out routing requires K >= 2")
    observed_group = evaluator.observed.reshape(
        len(evaluator.observed), budget, evaluator.receiver_count
    )
    context_group = evaluator.context.reshape(
        len(evaluator.observed), len(evaluator.raw), budget, evaluator.receiver_count
    )
    held_positions = np.linspace(
        0, budget - 1, min(holdout_count, budget), dtype=np.int64
    )
    self_error = torch.zeros(
        (3, len(evaluator.observed), len(evaluator.raw)),
        dtype=torch.float32,
        device=evaluator.raw.device,
    )
    for held in held_positions:
        keep = [position for position in range(budget) if position != held]
        observed = observed_group[:, keep].reshape(len(evaluator.observed), -1)
        context = context_group[:, :, keep].reshape(
            len(evaluator.observed), len(evaluator.raw), -1
        )
        target = observed_group[:, held]
        truth = evaluator.raw[:, target].permute(1, 0, 2)
        global_held = evaluator.predict(operator, observed, context, target)
        kernel_held = predict_memory_config_batched(
            evaluator,
            operator,
            memory,
            observed=observed,
            context=context,
            target=target,
        )
        neural_held = neural_event_predictions(
            model, encoded_neural, neural_center, neural_scale, observed, target
        )
        self_error += relative_error(
            torch.stack((global_held, kernel_held, neural_held), dim=0), truth[None]
        )
    self_error /= len(held_positions)
    route = self_error.argmin(dim=0)
    routed_prediction = torch.gather(
        branch_predictions.permute(1, 2, 0, 3),
        2,
        route[..., None, None].expand(-1, -1, 1, branch_predictions.shape[-1]),
    ).squeeze(2)
    routed_error = relative_error(routed_prediction, evaluator.truth)
    oracle_route = branch_error.argmin(dim=0)
    oracle_error = branch_error.min(dim=0).values
    return {
        "global": branch_error[0].mean(dim=0).cpu().numpy(),
        "kernel": branch_error[1].mean(dim=0).cpu().numpy(),
        "neural": branch_error[2].mean(dim=0).cpu().numpy(),
        "routed": routed_error.mean(dim=0).cpu().numpy(),
        "oracle": oracle_error.mean(dim=0).cpu().numpy(),
        "route_fraction_global": float((route == 0).float().mean().cpu()),
        "route_fraction_kernel": float((route == 1).float().mean().cpu()),
        "route_fraction_neural": float((route == 2).float().mean().cpu()),
        "route_oracle_agreement": float((route == oracle_route).float().mean().cpu()),
        "_self_error": self_error.permute(1, 2, 0).cpu().numpy(),
        "_branch_event_error": branch_error.permute(1, 2, 0).cpu().numpy(),
    }


def run(config_path: Path, output: Path, device: str) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for domain, setting in payload["domains"].items():
        archive = load_cross_domain_archive(Path(setting["dataset"]))
        model, neural_center, neural_scale, _ = load_neural(
            Path(setting["neural_checkpoint"]), archive, device
        )
        domain_result = {}
        for budget_text, specification in setting["budgets"].items():
            budget = int(budget_text)
            operator = GlobalConfig(
                int(specification["operator"]["rank"]),
                float(specification["operator"]["strength"]),
                "spectrum",
                str(specification["operator"]["normalization"]),
            )
            memory = MemoryConfig(**specification["memory"])
            subspace = fit_global_subspace(archive, operator.rank, operator.normalization, device)
            split_result = {}
            for split in ("validation", "test", "ood"):
                if not np.any(archive.split == split):
                    continue
                seed = int(payload["selection_seed"] if split == "validation" else payload["test_seed"])
                repeats = int(payload["validation_repeats"] if split == "validation" else payload["test_repeats"])
                evaluator = GlobalQ3Evaluator(
                    archive, subspace, split, budget, repeats, seed, device
                )
                raw = torch.as_tensor(
                    archive.responses[archive.split == split].reshape(len(evaluator.raw), -1),
                    dtype=torch.complex64,
                    device=device,
                )
                values = route_split(
                    evaluator, operator, memory, model, neural_center, neural_scale,
                    raw, int(payload["holdout_count"]),
                )
                branch = min(("global", "kernel", "neural"), key=lambda key: float(values[key].mean()))
                statistics = paired_summary(
                    values[branch], values["routed"], seed + budget
                )
                split_result[split] = {
                    "best_branch": branch,
                    "global_mean": float(values["global"].mean()),
                    "kernel_mean": float(values["kernel"].mean()),
                    "neural_mean": float(values["neural"].mean()),
                    "routed_mean": float(values["routed"].mean()),
                    "oracle_mean": float(values["oracle"].mean()),
                    **{
                        key: value
                        for key, value in values.items()
                        if key.startswith("route_") and not key.startswith("_route")
                    },
                    **statistics,
                }
                np.savez_compressed(
                    output / f"{domain.lower()}_k{budget}_{split}_systems.npz",
                    **{key: value for key, value in values.items() if isinstance(value, np.ndarray)},
                )
            domain_result[budget_text] = split_result
        result[domain] = domain_result
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
