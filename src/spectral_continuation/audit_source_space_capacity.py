"""Separate source-space parameter capacity from shared-encoder prediction.

Target-field optimization is explicitly an oracle. The physics-only objective
never accesses the target; both controls pay for the full 60-column cache.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .learned_source_space import load_cache, physics_coefficients, source_transform, squared_error


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    data, metadata = load_cache(
        config, config["systems"], torch.device(config.get("device", "cuda"))
    )
    rows, history = [], []
    for objective in ["query_truth_oracle", "physics_only"]:
        for bounded in [True, False]:
            started = time.perf_counter()
            raw = torch.zeros(
                len(metadata),
                9,
                12,
                device=data["H"].device,
                dtype=torch.float64,
                requires_grad=True,
            )
            optimizer = torch.optim.Adam([raw], lr=config["learning_rate"])

            def objective_value(coefficients, objective=objective):
                if objective == "query_truth_oracle":
                    return squared_error(data, coefficients)
                energy = (
                    (coefficients.conj() * (data["H"] @ coefficients[..., None]).squeeze(-1))
                    .sum(-1)
                    .real
                )
                linear = 2 * (coefficients.conj() * data["g"]).sum(-1).real
                # A target-free scale; omitted residual-RHS constant changes
                # neither the optimizer nor its selection of the best iterate.
                return (energy - linear) / data["g"].abs().square().sum(-1).sqrt().clamp_min(1e-20)

            with torch.no_grad():
                initial_coeff = physics_coefficients(
                    data["H"],
                    data["g"],
                    source_transform(raw, data["scales"], bounded),
                    config["ridge"],
                )
                best_objective = objective_value(initial_coeff)
                best_raw = raw.detach().clone()
            for step in range(config["steps"]):
                coefficients = physics_coefficients(
                    data["H"],
                    data["g"],
                    source_transform(raw, data["scales"], bounded),
                    config["ridge"],
                )
                value = objective_value(coefficients)
                optimizer.zero_grad()
                value.sum().backward()
                if not torch.isfinite(raw.grad).all():
                    raise FloatingPointError("nonfinite source-space capacity gradient")
                # Store the current (pre-update) parameters under this objective.
                with torch.no_grad():
                    improved = value < best_objective
                    best_objective[improved] = value.detach()[improved]
                    best_raw[improved] = raw.detach()[improved]
                optimizer.step()
                if step % 500 == 0:
                    history.append(
                        {
                            "objective": objective,
                            "bounded": bounded,
                            "step": step,
                            "raw_complex_error": float(
                                squared_error(data, coefficients).sqrt().mean().detach()
                            ),
                        }
                    )
            with torch.no_grad():
                coefficients = physics_coefficients(
                    data["H"],
                    data["g"],
                    source_transform(best_raw, data["scales"], bounded),
                    config["ridge"],
                )
                errors = squared_error(data, coefficients).sqrt().cpu().numpy()
            for info, error in zip(metadata, errors, strict=True):
                rows.append(
                    {
                        "system": info["system"],
                        "frequency": info["frequency"],
                        "objective": objective,
                        "bounded": bounded,
                        "complex_rel_l2": float(error),
                        "scope": "target truth used to fit representation"
                        if objective == "query_truth_oracle"
                        else "target-free per-instance fit; requires 60 background solves/cache, not six-solve deployment cost",
                    }
                )
            record = {
                "objective": objective,
                "bounded": bounded,
                "macro_error": float(np.mean(errors)),
                "seconds": time.perf_counter() - started,
            }
            print(json.dumps(record), flush=True)
            (output / "results.json").write_text(json.dumps(rows, indent=2))
            (output / "history.json").write_text(json.dumps(history, indent=2))
            np.savez_compressed(
                output / f"parameters_{objective}_{bounded}.npz", raw=best_raw.cpu().numpy()
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
