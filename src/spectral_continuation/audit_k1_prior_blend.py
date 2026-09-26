"""Select one domain-agnostic kernel--neural blend for single-prompt episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probabilistic_response_operator import paired_summary


def mixture_error(archive: np.lib.npyio.NpzFile, neural_weight: float) -> np.ndarray:
    error = archive["_branch_event_error"]
    kernel_weight = 1.0 - neural_weight
    squared = (
        kernel_weight**2 * np.square(error[..., 1])
        + neural_weight**2 * np.square(error[..., 2])
        + 2.0 * kernel_weight * neural_weight * archive["_kernel_neural_cross"]
    )
    return np.sqrt(np.maximum(squared, 0.0)).mean(axis=0)


def run(router_root: Path, output: Path, seed: int) -> dict:
    validation = sorted(router_root.glob("*_k1_validation_systems.npz"))
    candidates = np.linspace(0.0, 1.0, 101)
    scores = []
    for weight in candidates:
        degradations = []
        for path in validation:
            archive = np.load(path)
            error = archive["_branch_event_error"].mean(axis=0)
            best = float(error.mean(axis=0).min())
            degradations.append(float(mixture_error(archive, weight).mean() / best - 1.0))
        scores.append((max(degradations), float(np.mean(np.maximum(degradations, 0.0))), weight))
    _, _, selected = min(scores)
    result = {"selected_neural_weight": float(selected), "domains": {}}
    for path in sorted(router_root.glob("*_k1_test_systems.npz")):
        archive = np.load(path)
        domain = path.stem.split("_k1_")[0]
        error = archive["_branch_event_error"].mean(axis=0)
        best_index = int(error.mean(axis=0).argmin())
        best = error[:, best_index]
        mixture = mixture_error(archive, selected)
        result["domains"][domain] = {
            "best_branch": ("global", "kernel", "neural")[best_index],
            "best_branch_mean": float(best.mean()),
            "blend_mean": float(mixture.mean()),
            **paired_summary(best, mixture, seed),
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    print(json.dumps(run(args.router_root, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
