"""Fit one shared validation-only risk calibrator for prompt adapter routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probabilistic_response_operator import paired_summary


def features(self_error: np.ndarray) -> np.ndarray:
    log_error = np.log(np.maximum(self_error, 1e-8))
    centered = log_error - log_error.mean(axis=-1, keepdims=True)
    return np.concatenate(
        (
            np.ones((*log_error.shape[:-1], 1)),
            log_error,
            centered,
            np.log(np.maximum(self_error.mean(axis=-1, keepdims=True), 1e-8)),
        ),
        axis=-1,
    )


def fit_risk_models(
    validation_files: list[Path], ridge: float
) -> np.ndarray:
    matrices = []
    targets = []
    for path in validation_files:
        archive = np.load(path)
        matrices.append(features(archive["_self_error"]).reshape(-1, 8))
        targets.append(
            np.log(np.maximum(archive["_branch_event_error"], 1e-8)).reshape(-1, 3)
        )
    matrix = np.concatenate(matrices)
    target = np.concatenate(targets)
    gram = matrix.T @ matrix + ridge * np.eye(matrix.shape[1])
    return np.linalg.solve(gram, matrix.T @ target)


def run(router_root: Path, output: Path, ridge: float, seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    validation_files = sorted(router_root.glob("*_validation_systems.npz"))
    weights = fit_risk_models(validation_files, ridge)
    result = {}
    for path in sorted(router_root.glob("*_test_systems.npz")):
        stem = path.stem.removesuffix("_test_systems")
        domain, budget = stem.rsplit("_k", 1)
        archive = np.load(path)
        self_error = archive["_self_error"]
        branch_error = archive["_branch_event_error"]
        predicted_log_risk = features(self_error) @ weights
        route = predicted_log_risk.argmin(axis=-1)
        routed = np.take_along_axis(branch_error, route[..., None], axis=-1)[..., 0]
        per_system = routed.mean(axis=0)
        branch_system = branch_error.mean(axis=0)
        best_index = int(branch_system.mean(axis=0).argmin())
        best = branch_system[:, best_index]
        stats = paired_summary(best, per_system, seed + int(budget))
        key = f"{domain}_k{budget}"
        result[key] = {
            "best_branch": ("global", "kernel", "neural")[best_index],
            "best_branch_mean": float(best.mean()),
            "calibrated_route_mean": float(per_system.mean()),
            "route_fraction_global": float(np.mean(route == 0)),
            "route_fraction_kernel": float(np.mean(route == 1)),
            "route_fraction_neural": float(np.mean(route == 2)),
            **stats,
        }
        np.savez_compressed(
            output / f"{key}_systems.npz", routed=per_system, best_branch=best
        )
    np.save(output / "risk_weights.npy", weights)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    print(json.dumps(run(args.router_root, args.output, args.ridge, args.seed), indent=2))


if __name__ == "__main__":
    main()
