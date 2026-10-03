"""Bounded component-count audit, with a matched single-component control.

Instance parameters are selected only by context marginal likelihood; optional
query-truth fits are explicitly labelled capacity oracles, never deployment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .coherent_representation import complex_metrics
from .mixture_coherent_kernel import MixtureCoherentKernelEncoder


def fit(model, x, y, query, config, target=None):
    raw = model.initial_raw.double()[None].expand(len(x), -1).clone()
    shaped = raw.reshape(len(x), model.components, 9)
    shaped[..., 2:4] = np.arctanh(np.log(config["initial_epsilon"]) / 2)
    raw = torch.nn.Parameter(raw)
    optimizer = torch.optim.Adam([raw], lr=config["learning_rate"])
    best_raw = raw.detach().clone()
    best_score = torch.full((len(x),), float("inf"), device=x.device, dtype=torch.float64)
    eye = torch.eye(x.shape[1], device=x.device, dtype=torch.complex128)
    for step in range(config["steps"] + 1):
        geometry = model.geometry_from_raw(raw)
        kernel = model.covariance(x, x, *geometry) + config["ridge"] * eye
        alpha = torch.linalg.solve(kernel, y[..., None]).squeeze(-1)
        if config.get("objective", "marginal_likelihood") == "query_oracle":
            if target is None:
                raise ValueError("oracle requires explicitly passed target")
            pred = (model.covariance(query, x, *geometry) @ alpha[..., None]).squeeze(-1)
            score = (pred - target).abs().square().mean(-1) / target.abs().square().mean(-1)
            loss = score.clamp_min(1e-30).log().mean()
        else:
            variance = (y.conj() * alpha).sum(-1).real / y.shape[1]
            score = y.shape[1] * variance.clamp_min(1e-30).log() + torch.linalg.slogdet(kernel)[1]
            loss = score.mean()
        with torch.no_grad():
            keep = score < best_score
            best_score[keep], best_raw[keep] = score[keep], raw[keep]
        if step == config["steps"]:
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 10)
        optimizer.step()
    with torch.no_grad():
        geometry = model.geometry_from_raw(best_raw)
        kernel = model.covariance(x, x, *geometry) + config["ridge"] * eye
        pred = (
            model.covariance(query, x, *geometry) @ torch.linalg.solve(kernel, y[..., None])
        ).squeeze(-1)
    return pred, best_raw


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(config.get("seed", 42))
    with np.load(config["pool"]) as z:
        arrays = dict(z)
    all_rows, summaries = [], []
    for components in config["components"]:
        model = MixtureCoherentKernelEncoder(components=components, ridge=config["ridge"]).cuda()
        predictions, latent = [], []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for first in range(0, len(arrays["x"]), config["batch"]):
            sl = slice(first, first + config["batch"])
            x = torch.as_tensor(arrays["x"][sl], dtype=torch.float64, device="cuda")
            y = torch.as_tensor(arrays["y"][sl], dtype=torch.complex128, device="cuda")
            q = torch.as_tensor(arrays["q"][sl], dtype=torch.float64, device="cuda")
            target = None
            if config.get("objective") == "query_oracle":
                target = torch.as_tensor(
                    arrays["target"][sl], dtype=torch.complex128, device="cuda"
                )
            pred, raw = fit(model, x, y, q, config, target)
            predictions.append(pred.cpu().numpy())
            latent.append(raw.cpu().numpy())
            print(
                json.dumps(
                    {
                        "components": components,
                        "completed": first + len(x),
                        "total": len(arrays["x"]),
                    }
                ),
                flush=True,
            )
        prediction = np.concatenate(predictions)
        torch.cuda.synchronize()
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
                        "components": components,
                        **complex_metrics(prediction[mask] * scale, arrays["target"][mask] * scale),
                    }
                )
        summary = {
            "components": components,
            "macro_error": float(np.mean([r["complex_rel_l2"] for r in rows])),
            "seconds": elapsed,
            "patches": len(prediction),
            "steps": config["steps"],
            "objective": config.get("objective", "marginal_likelihood"),
            "scope": "development per-instance audit; NOT shared learning or unseen-frequency generation; wall time under concurrent GPU jobs",
        }
        print(json.dumps(summary), flush=True)
        summaries.append(summary)
        all_rows.extend(rows)
        np.savez_compressed(
            output / f"components{components}.npz",
            prediction=prediction,
            latent=np.concatenate(latent),
        )
        (output / "summary.json").write_text(json.dumps(summaries, indent=2))
        (output / "results.json").write_text(json.dumps(all_rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
