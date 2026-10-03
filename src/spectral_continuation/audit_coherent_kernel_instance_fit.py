"""Non-neural per-context hyperparameter fitting, with zero query-label access.

Compares amortized geometry to context-only LOO or profiled marginal likelihood.
Same kernel family, regularization, positivity constraints and observed points.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .coherent_representation import complex_metrics
from .learned_coherent_kernel import CoherentKernelEncoder


def fit_context(model, x, y, query, config, weights=None, oracle_target=None):
    raw = torch.zeros((len(x), 8), dtype=torch.float64, device=x.device)
    raw[:, [2, 3]] = np.arctanh(np.log(config["initial_epsilon"]) / 2)
    if config.get("checkpoint"):
        if weights is None:
            weights = torch.ones_like(y.real)
        with torch.no_grad():
            raw = model.raw_geometry(x, y, weights).double()
    raw = torch.nn.Parameter(raw)
    optimizer = torch.optim.Adam([raw], lr=config["learning_rate"])
    eye = torch.eye(x.shape[1], dtype=torch.complex128, device=x.device)
    best_score = torch.full((len(x),), float("inf"), device=x.device, dtype=torch.float64)
    best_raw = raw.detach().clone()
    for step in range(config["steps"] + 1):
        k, lower = model.geometry_from_raw(raw)
        kernel = model.covariance(x, x, k, lower) + config["ridge"] * eye
        inverse = torch.linalg.inv(kernel)
        alpha = (inverse @ y[..., None]).squeeze(-1)
        if config["objective"] == "query_oracle":
            if oracle_target is None:
                raise ValueError("explicit oracle target required for capacity audit")
            pred = (model.covariance(query, x, k, lower) @ alpha[..., None]).squeeze(-1)
            per_patch = (pred - oracle_target).abs().square().mean(
                -1
            ) / oracle_target.abs().square().mean(-1)
        elif config["objective"] == "loo":
            loo_error = alpha / inverse.diagonal(dim1=-2, dim2=-1).real
            per_patch = loo_error.abs().square().mean(-1) / y.abs().square().mean(-1)
        else:
            variance = (y.conj() * alpha).sum(-1).real / y.shape[1]
            per_patch = (
                y.shape[1] * variance.clamp_min(1e-30).log() + torch.linalg.slogdet(kernel)[1]
            )
        with torch.no_grad():
            keep = per_patch < best_score
            best_score[keep] = per_patch[keep]
            best_raw[keep] = raw[keep]
        if step == config["steps"]:
            break
        optimizer.zero_grad()
        # log-LOO balances difficult patches without using target errors.
        loss = (
            per_patch.clamp_min(1e-30).log().mean()
            if config["objective"] in ("loo", "query_oracle")
            else per_patch.mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 10)
        optimizer.step()
    with torch.no_grad():
        k, lower = model.geometry_from_raw(best_raw)
        kernel = model.covariance(x, x, k, lower) + config["ridge"] * eye
        coeff = torch.linalg.solve(kernel, y[..., None])
        return (model.covariance(query, x, k, lower) @ coeff).squeeze(-1), best_score


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(4)
    with np.load(config["pool"]) as z:
        arrays = dict(z)
    model = CoherentKernelEncoder(curvature=True, ridge=config["ridge"]).cuda()
    if config.get("checkpoint"):
        state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    predictions, criteria = [], []
    start = time.perf_counter()
    for first in range(0, len(arrays["x"]), config["batch"]):
        sl = slice(first, first + config["batch"])
        x = torch.tensor(arrays["x"][sl], device="cuda", dtype=torch.float64)
        y = torch.tensor(arrays["y"][sl], device="cuda", dtype=torch.complex128)
        q = torch.tensor(arrays["q"][sl], device="cuda", dtype=torch.float64)
        weights = torch.tensor(arrays["weight"][sl], device="cuda", dtype=torch.float64)
        oracle_target = None
        if config["objective"] == "query_oracle":
            oracle_target = torch.tensor(
                arrays["target"][sl], device="cuda", dtype=torch.complex128
            )
        p, criterion = fit_context(model, x, y, q, config, weights, oracle_target)
        predictions.append(p.cpu().numpy())
        criteria.append(criterion.cpu().numpy())
        if first % (4 * config["batch"]) == 0:
            print(
                json.dumps(
                    {
                        "completed": first + len(x),
                        "total": len(arrays["x"]),
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    predictions = np.concatenate(predictions)
    elapsed = time.perf_counter() - start
    rows = []
    for system in np.unique(arrays["system"]):
        for frequency in np.unique(arrays["frequency"]):
            mask = (arrays["system"] == system) & (arrays["frequency"] == frequency)
            if not mask.any():
                continue
            scale = arrays["scale"][mask, None]
            rows.append(
                {
                    "system": int(system),
                    "frequency": float(frequency),
                    **complex_metrics(predictions[mask] * scale, arrays["target"][mask] * scale),
                }
            )
    summary = {
        "macro_error": float(np.mean([r["complex_rel_l2"] for r in rows])),
        "seconds": elapsed,
        "patches": len(predictions),
        "selection": "query-truth fit: capacity audit only, not deployable"
        if config["objective"] == "query_oracle"
        else "context-only objective per patch; zero query label access",
    }
    print(json.dumps(summary), flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "results.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(
        output / "predictions.npz", prediction=predictions, fit_criterion=np.concatenate(criteria)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=["loo", "marginal_likelihood", "query_oracle"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.objective:
        config["objective"] = args.objective
    if args.steps is not None:
        config["steps"] = args.steps
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    run(config, args.output)


if __name__ == "__main__":
    main()
