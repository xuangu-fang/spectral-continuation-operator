"""Learn a shared continuous kernel--neural adapter from prompt-only diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .calibrate_prompt_router import features
from .probabilistic_response_operator import paired_summary


class PriorAdapter(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(value)[..., 0])


def run(router_root: Path, output: Path, seed: int, steps: int, device_name: str) -> dict:
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    matrices = []
    components = []
    for path in sorted(router_root.glob("*_validation_systems.npz")):
        archive = np.load(path)
        matrices.append(features(archive["_self_error"]).reshape(-1, 8))
        error = archive["_branch_event_error"]
        components.append(
            np.stack(
                (
                    np.square(error[..., 1]),
                    np.square(error[..., 2]),
                    archive["_kernel_neural_cross"],
                ),
                axis=-1,
            ).reshape(-1, 3)
        )
    matrix = np.concatenate(matrices).astype(np.float32)
    component = np.concatenate(components).astype(np.float32)
    mean = matrix.mean(axis=0)
    standard_deviation = np.maximum(matrix.std(axis=0), 1e-5)
    matrix = (matrix - mean) / standard_deviation
    device = torch.device(device_name)
    x = torch.as_tensor(matrix, device=device)
    values = torch.as_tensor(component, device=device)
    model = PriorAdapter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    batch_size = min(8192, len(x))
    for _ in range(steps):
        index = torch.as_tensor(
            generator.integers(0, len(x), size=batch_size),
            dtype=torch.long,
            device=device,
        )
        neural_weight = model(x[index])
        kernel_weight = 1.0 - neural_weight
        kernel_square, neural_square, cross = values[index].unbind(dim=-1)
        mixture_square = (
            kernel_weight.square() * kernel_square
            + neural_weight.square() * neural_square
            + 2.0 * kernel_weight * neural_weight * cross
        ).clamp_min(0.0)
        # A weak endpoint preference avoids an unidentifiable 0.5 blend when
        # the two predictions are numerically identical.
        loss = torch.sqrt(mixture_square + 1e-10).mean()
        loss = loss + 1e-5 * (neural_weight * kernel_weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    result = {}
    model.eval()
    with torch.inference_mode():
        for path in sorted(router_root.glob("*_test_systems.npz")):
            archive = np.load(path)
            stem = path.stem.removesuffix("_test_systems")
            domain, budget = stem.rsplit("_k", 1)
            test_x = (features(archive["_self_error"]) - mean) / standard_deviation
            neural_weight = model(
                torch.as_tensor(test_x, dtype=torch.float32, device=device)
            ).cpu().numpy()
            kernel_weight = 1.0 - neural_weight
            error = archive["_branch_event_error"]
            mixture_square = (
                np.square(kernel_weight) * np.square(error[..., 1])
                + np.square(neural_weight) * np.square(error[..., 2])
                + 2.0 * kernel_weight * neural_weight * archive["_kernel_neural_cross"]
            )
            mixture = np.sqrt(np.maximum(mixture_square, 0.0)).mean(axis=0)
            branch_system = error.mean(axis=0)
            best_index = int(branch_system.mean(axis=0).argmin())
            best = branch_system[:, best_index]
            stats = paired_summary(best, mixture, seed + int(budget))
            result[f"{domain}_k{budget}"] = {
                "best_branch": ("global", "kernel", "neural")[best_index],
                "best_branch_mean": float(best.mean()),
                "adapter_mean": float(mixture.mean()),
                "mean_neural_weight": float(neural_weight.mean()),
                **stats,
            }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "feature_mean": mean, "feature_std": standard_deviation},
        output / "adapter.pt",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.router_root, args.output, args.seed, args.steps, args.device), indent=2))


if __name__ == "__main__":
    main()
