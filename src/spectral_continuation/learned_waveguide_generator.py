"""Shared learned real generator correction with an analytic complex decoder.

Training-only real slowness supervision is explicit. At inference, the encoder
sees context-derived normal equations and background only; it does not see true
range-dependent speed or query labels. This is a development mechanism pilot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit_waveguide_generator_identification import (
    generator_geometry,
    identify,
    latent_from_coefficients,
    normal_statistics,
    true_projection,
)
from .audit_waveguide_spectral_transport import encode_field
from .coarse_waveguide_operator import query_field
from .coherent_representation import complex_metrics
from .learned_waveguide_spectral_codec import SpectralConv1d


class GeneratorEncoder(nn.Module):
    def __init__(self, channels, width=48, modes=16, layers=3, terms=16):
        super().__init__()
        self.lift = nn.Conv1d(channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, terms, 1)
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, features, baseline):
        count = features.shape[-1]
        hidden = nn.functional.pad(self.lift(features), (0, 16))
        for spectral, local in zip(self.spectral, self.local, strict=True):
            hidden = nn.functional.gelu(spectral(hidden) + local(hidden))
        # Small operator corrections; no frequency-dependent output head.
        return baseline + 0.002 * self.readout(hidden[..., :count]).transpose(1, 2)


def context_features(gram, rhs, baseline, geometry):
    cells = len(rhs)
    upper = np.triu_indices(rhs.shape[-1])
    background = (
        (geometry["background_q"] * geometry["weights"]) @ geometry["psi"] / geometry["depth"]
    )
    return np.concatenate(
        [
            gram[:, upper[0], upper[1]],
            rhs / 0.02,
            baseline / 0.02,
            np.tile(background, (cells, 1)),
            (geometry["middle"] / geometry["edges"][-1])[:, None],
        ],
        axis=-1,
    ).T


def prepare(config, systems, training):
    features, baselines, targets, metadata = [], [], [], []
    for system in systems:
        if training:
            with np.load(Path(config["training_modal_cache"]) / f"system{system}_modal.npz") as z:
                spec = json.loads(str(z["spec"]))
                frequencies, ranges = z["frequencies_hz"], z["ranges_m"]
                obs = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
                coefficients = z["modal_coefficients"][obs].astype(complex)
            fields, depths = None, None
        else:
            with np.load(Path(config["data"]) / f"system{system}_m32_r2049.npz") as z:
                spec = json.loads(str(z["spec"]))
                fields, depths, frequencies, ranges = (
                    z["field"],
                    z["depths_m"],
                    z["frequencies_hz"],
                    z["ranges_m"],
                )
            obs = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
            coefficients = np.stack(
                [encode_field(fields[i], depths, spec["depth_m"], 32) for i in obs]
            )
        take = np.linspace(0, len(ranges) - 1, config["context_range_samples"], dtype=int)
        geometry = generator_geometry(spec, config["cells"], config["depth_terms"])
        gram, rhs = normal_statistics(
            spec,
            coefficients[:, take],
            frequencies[obs],
            ranges[take],
            geometry,
            config["interpolation_order"],
        )
        baseline = identify(gram, rhs, config["ridge"], config["range_penalty"])
        features.append(context_features(gram, rhs, baseline, geometry))
        baselines.append(baseline)
        # These labels never enter features/decoder geometry.
        targets.append(true_projection(spec, geometry))
        metadata.append(
            {
                "system": system,
                "geometry": geometry,
                "initial": coefficients[0, 0],
                "fields": fields,
                "depths": depths,
                "ranges": ranges,
                "frequencies": frequencies,
            }
        )
    return {
        "features": np.array(features, dtype=np.float32),
        "baseline": np.array(baselines, dtype=np.float32),
        "target": np.array(targets, dtype=np.float32),
    }, metadata


def evaluate(coefficients, metadata, config):
    rows = []
    for beta, item in zip(coefficients, metadata, strict=True):
        latent, clip = latent_from_coefficients(beta, item["initial"], item["geometry"])
        roi = item["ranges"] >= config["roi_min_range_m"]
        for frequency in config["query_frequencies"]:
            fi = int(np.flatnonzero(item["frequencies"] == frequency)[0])
            prediction = query_field(latent, frequency, item["ranges"], item["depths"])
            rows.append(
                {
                    "system": item["system"],
                    "frequency": frequency,
                    "clipping_fraction": clip,
                    **complex_metrics(prediction[:, roi], item["fields"][fi][:, roi]),
                }
            )
    return rows


def error(rows):
    return float(np.mean([row["complex_rel_l2"] for row in rows]))


def kernel_predictions(train, val):
    x, q = (
        train["features"].reshape(len(train["features"]), -1).astype(float),
        val["features"].reshape(len(val["features"]), -1).astype(float),
    )
    variance = max(float(np.mean(np.sum((x - x.mean(0)) ** 2, axis=-1))), 1e-12)

    def distance(a, b):
        return np.maximum(
            ((a * a).sum(-1)[:, None] + (b * b).sum(-1)[None] - 2 * a @ b.T) / variance, 0
        )

    dxx, dqx = distance(x, x), distance(q, x)
    target = (train["target"] - train["baseline"]).reshape(len(x), -1)
    for bandwidth in [0.5, 1, 2]:
        for ridge in [0.0001, 0.01, 0.1]:
            coefficient = np.linalg.solve(
                np.exp(-dxx / bandwidth**2) + ridge * np.eye(len(x)), target
            )
            pred = val["baseline"] + (np.exp(-dqx / bandwidth**2) @ coefficient).reshape(
                val["baseline"].shape
            )
            yield {"bandwidth": bandwidth, "ridge": ridge}, pred


def run(config, output):
    if set(config["train_systems"]) & set(config["validation_systems"]):
        raise ValueError("system overlap")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    start = time.perf_counter()
    train, _ = prepare(config, config["train_systems"], True)
    val, validation = prepare(config, config["validation_systems"], False)
    mean = train["features"].mean((0, 2), keepdims=True)
    std = np.maximum(train["features"].std((0, 2), keepdims=True), 0.001)
    for pool in [train, val]:
        pool["features"] = (pool["features"] - mean) / std
    np.savez_compressed(output / "training_statistics.npz", **train, mean=mean, std=std)
    baseline = evaluate(val["baseline"], validation, config)
    oracle = evaluate(val["target"], validation, config)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2))
    (output / "oracle.json").write_text(json.dumps(oracle, indent=2))
    kernel_history, best_kernel = [], float("inf")
    for options, prediction in kernel_predictions(train, val):
        rows = evaluate(prediction, validation, config)
        kernel_history.append({**options, "macro_error": error(rows)})
        if error(rows) < best_kernel:
            best_kernel = error(rows)
            (output / "best_kernel.json").write_text(
                json.dumps({"options": options, "results": rows}, indent=2)
            )
    (output / "kernel_selection.json").write_text(json.dumps(kernel_history, indent=2))
    print(
        json.dumps(
            {
                "data_ready_seconds": time.perf_counter() - start,
                "baseline": error(baseline),
                "oracle": error(oracle),
                "kernel": best_kernel,
            }
        ),
        flush=True,
    )
    model = GeneratorEncoder(train["features"].shape[1], **config["model"]).cuda()
    train_gpu = {k: torch.as_tensor(v, device="cuda") for k, v in train.items()}
    val_gpu = {k: torch.as_tensor(v, device="cuda") for k, v in val.items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"])
    history, best = [], float("inf")
    for step in range(config["steps"] + 1):
        if step % config["eval_every"] == 0 or step == config["steps"]:
            with torch.no_grad():
                prediction = model(val_gpu["features"], val_gpu["baseline"]).cpu().numpy()
                train_mse = []
                for offset in range(0, len(train["features"]), 32):
                    batch = slice(offset, offset + 32)
                    fitted = model(train_gpu["features"][batch], train_gpu["baseline"][batch])
                    train_mse.extend(
                        (fitted - train_gpu["target"][batch]).square().mean((1, 2)).cpu().tolist()
                    )
            rows = evaluate(prediction, validation, config)
            row = {
                "step": step,
                "macro_error": error(rows),
                "train_coefficient_rmse": float(np.sqrt(np.mean(train_mse))),
                "validation_coefficient_rmse": float(
                    np.sqrt(np.mean((prediction - val["target"]) ** 2))
                ),
                "elapsed": time.perf_counter() - start,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if row["macro_error"] < best:
                best = row["macro_error"]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": config,
                        "mean": mean,
                        "std": std,
                        "step": step,
                    },
                    output / "best.pt",
                )
                (output / "best_results.json").write_text(json.dumps(rows, indent=2))
            (output / "history.json").write_text(json.dumps(history, indent=2))
        if step == config["steps"]:
            break
        index = torch.randint(len(train["features"]), (config["batch_size"],), device="cuda")
        prediction = model(train_gpu["features"][index], train_gpu["baseline"][index])
        loss = ((prediction - train_gpu["target"][index]) / 0.002).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "mean": mean,
            "std": std,
            "step": config["steps"],
        },
        output / "last.pt",
    )
    summary = {
        "best_macro_error": best,
        "baseline": error(baseline),
        "oracle": error(oracle),
        "kernel": best_kernel,
        "real_parameter_count": sum(
            p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
        ),
        "scope": "single-seed development supervised generator inference; real medium used only as training labels and labelled oracle",
        "seconds": time.perf_counter() - start,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["train_systems"] = list(range(20261300, 20261332)) + list(range(20261500, 20261980))
    run(config, args.output)


if __name__ == "__main__":
    main()
