"""Train a tiny shared risk head from prompt-only self-validation statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .calibrate_prompt_router import features
from .probabilistic_response_operator import paired_summary


class RiskHead(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def run(
    router_root: Path,
    output: Path,
    seed: int,
    steps: int,
    device_name: str,
) -> dict:
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    matrices = []
    targets = []
    for path in sorted(router_root.glob("*_validation_systems.npz")):
        archive = np.load(path)
        matrices.append(features(archive["_self_error"]).reshape(-1, 8))
        targets.append(
            np.log(np.maximum(archive["_branch_event_error"], 1e-8)).reshape(-1, 3)
        )
    matrix = np.concatenate(matrices).astype(np.float32)
    target = np.concatenate(targets).astype(np.float32)
    mean = matrix.mean(axis=0)
    standard_deviation = matrix.std(axis=0)
    standard_deviation = np.maximum(standard_deviation, 1e-5)
    matrix = (matrix - mean) / standard_deviation
    device = torch.device(device_name)
    x = torch.as_tensor(matrix, device=device)
    y = torch.as_tensor(target, device=device)
    model = RiskHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    batch_size = min(8192, len(x))
    for _ in range(steps):
        indices = torch.as_tensor(
            generator.integers(0, len(x), size=batch_size),
            dtype=torch.long,
            device=device,
        )
        prediction = model(x[indices])
        log_risk = y[indices]
        risk = torch.exp(log_risk)
        relative_regret = risk / risk.min(dim=-1, keepdim=True).values.clamp_min(1e-8) - 1.0
        probability = torch.softmax(-prediction, dim=-1)
        decision_loss = (probability * relative_regret).sum(dim=-1).mean()
        calibration_loss = (prediction - log_risk).square().mean()
        loss = decision_loss + 0.02 * calibration_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    result = {}
    model.eval()
    with torch.inference_mode():
        for path in sorted(router_root.glob("*_test_systems.npz")):
            stem = path.stem.removesuffix("_test_systems")
            domain, budget = stem.rsplit("_k", 1)
            archive = np.load(path)
            test_x = (features(archive["_self_error"]) - mean) / standard_deviation
            predicted = model(torch.as_tensor(test_x, dtype=torch.float32, device=device))
            route = predicted.argmin(dim=-1).cpu().numpy()
            branch_error = archive["_branch_event_error"]
            routed = np.take_along_axis(branch_error, route[..., None], axis=-1)[..., 0]
            per_system = routed.mean(axis=0)
            branch_system = branch_error.mean(axis=0)
            best_index = int(branch_system.mean(axis=0).argmin())
            best = branch_system[:, best_index]
            stats = paired_summary(best, per_system, seed + int(budget))
            result[f"{domain}_k{budget}"] = {
                "best_branch": ("global", "kernel", "neural")[best_index],
                "best_branch_mean": float(best.mean()),
                "risk_head_mean": float(per_system.mean()),
                "route_fraction_global": float(np.mean(route == 0)),
                "route_fraction_kernel": float(np.mean(route == 1)),
                "route_fraction_neural": float(np.mean(route == 2)),
                **stats,
            }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "feature_mean": mean, "feature_std": standard_deviation},
        output / "risk_head.pt",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.router_root, args.output, args.seed, args.steps, args.device), indent=2))


if __name__ == "__main__":
    main()
