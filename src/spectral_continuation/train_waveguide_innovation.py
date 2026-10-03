"""Controlled addition of coherent context residuals to the same generator FNO.

Only observed frequencies enter features. Extra channels are initially ignored;
the matched zero-channel model has identical real parameter count and updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit_waveguide_generator_identification import generator_geometry, latent_from_coefficients
from .audit_waveguide_spectral_transport import encode_field
from .coarse_waveguide_operator import query_coefficients
from .learned_waveguide_generator import (
    GeneratorEncoder,
    error,
    evaluate,
    kernel_predictions,
    prepare,
)
from .train_waveguide_generator_field_loss import DifferentiableGenerator, load_field_supervision


def stable_innovation(observed, predicted, initial, mass, kz2, frequencies, ranges):
    """Gauge-invariant modal residual; undo phase, never undo evanescent decay."""
    norm = float(np.linalg.norm(initial))
    if norm <= 1e-12:
        raise ValueError("nonzero known source required")
    anchor = initial[int(np.argmax(np.abs(initial)))]
    scale = norm * anchor / abs(anchor)
    columns = []
    for fi, frequency in enumerate(frequencies):
        values, vectors = np.linalg.eigh((2 * np.pi * frequency) ** 2 * mass - np.diag(kz2))
        wavenumber = np.sqrt(values.astype(complex)).real
        residual = (observed[fi] - predicted[fi]) @ vectors / scale
        residual *= np.exp(-1j * np.asarray(ranges)[:, None] * wavenumber[None])
        # Sampling coordinates coincide with the physical cell boundaries.
        centered = 0.5 * (residual[1:] + residual[:-1])
        columns.extend([centered.real.T, centered.imag.T])
    return np.concatenate(columns, axis=0)


def extra_features(config, systems, baseline, training):
    features, seconds = [], []
    for system, beta in zip(systems, baseline, strict=True):
        if training:
            path = Path(config["training_modal_cache"]) / f"system{system}_modal.npz"
            with np.load(path) as z:
                spec = json.loads(str(z["spec"]))
                frequencies, ranges = z["frequencies_hz"], z["ranges_m"]
                obs = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
                coefficients = z["modal_coefficients"][obs].astype(complex)
        else:
            path = Path(config["data"]) / f"system{system}_m32_r2049.npz"
            with np.load(path) as z:
                spec = json.loads(str(z["spec"]))
                frequencies, ranges = z["frequencies_hz"], z["ranges_m"]
                obs = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
                coefficients = np.stack(
                    [encode_field(z["field"][i], z["depths_m"], spec["depth_m"], 32) for i in obs]
                )
        take = np.linspace(0, len(ranges) - 1, config["context_range_samples"], dtype=int)
        observed = coefficients[:, take]
        initial = observed[0, 0]
        start = time.perf_counter()
        geometry = generator_geometry(spec, config["cells"], config["depth_terms"])
        if not np.allclose(ranges[take], geometry["edges"]):
            raise ValueError("context must coincide with cell boundaries")
        latent, _ = latent_from_coefficients(beta, initial, geometry)
        background, _ = latent_from_coefficients(np.zeros_like(beta), initial, geometry)
        predicted = np.stack(
            [query_coefficients(latent, f, ranges[take]) for f in frequencies[obs]]
        )
        features.append(
            stable_innovation(
                observed,
                predicted,
                initial,
                background["mass"][0],
                geometry["kz2"],
                frequencies[obs],
                ranges[take],
            )
        )
        seconds.append(time.perf_counter() - start)
    return np.array(features, dtype=np.float32), seconds


def expanded_model(state, old_channels, new_channels, model_options):
    model = GeneratorEncoder(old_channels + new_channels, **model_options)
    expanded = {key: value.clone() for key, value in state.items()}
    weight = torch.zeros_like(model.lift.weight)
    weight[:, :old_channels] = expanded["lift.weight"]
    expanded["lift.weight"] = weight
    model.load_state_dict(expanded)
    return model


def run(options, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise ValueError("do not overwrite an existing experiment")
    source = Path(options["coefficient_run"])
    config = json.loads((source / "config.json").read_text())
    if set(config["train_systems"]) & set(config["validation_systems"]):
        raise ValueError("train/validation overlap")
    checkpoint_path = Path(options["field_run"]) / "best.pt"
    record_config = {
        **options,
        "base": config,
        "initial_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    (output / "config.json").write_text(json.dumps(record_config, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(options["seed"])
    start = time.perf_counter()
    with np.load(source / "training_statistics.npz") as z:
        train = {key: z[key] for key in ["features", "baseline", "target"]}
        old_mean, old_std = z["mean"], z["std"]
    val, metadata = prepare(config, config["validation_systems"], False)
    val["features"] = (val["features"] - old_mean) / old_std
    old_channels = train["features"].shape[1]
    residual_train, train_cost = extra_features(
        config, config["train_systems"], train["baseline"], True
    )
    residual_val, val_cost = extra_features(
        config, config["validation_systems"], val["baseline"], False
    )
    mean = residual_train.mean((0, 2), keepdims=True)
    std = np.maximum(residual_train.std((0, 2), keepdims=True), 0.001)
    for pool, residual in [(train, residual_train), (val, residual_val)]:
        addition = (residual - mean) / std
        if options["feature_mode"] == "zero_control":
            addition = np.zeros_like(addition)
        elif options["feature_mode"] != "coherent_innovation":
            raise ValueError(options["feature_mode"])
        pool["features"] = np.concatenate([pool["features"], addition], axis=1)
    np.savez_compressed(
        output / "training_statistics.npz",
        **train,
        extra_mean=mean,
        extra_std=std,
        old_mean=old_mean,
        old_std=old_std,
    )
    (output / "encoding_cost.json").write_text(
        json.dumps(
            {
                "train_extra_seconds": train_cost,
                "dev_extra_seconds": val_cost,
                "scope": "additional geometry+3context-propagation+frame only; NOT total encoding latency; disk IO/modal projection excluded; sharedCPU",
            },
            indent=2,
        )
    )
    kernel_history, best_kernel = [], float("inf")
    for choices, prediction in kernel_predictions(train, val):
        rows = evaluate(prediction, metadata, config)
        kernel_history.append({**choices, "raw": error(rows)})
        if error(rows) < best_kernel:
            best_kernel = error(rows)
            (output / "best_kernel.json").write_text(
                json.dumps({"options": choices, "results": rows}, indent=2)
            )
    (output / "kernel_selection.json").write_text(json.dumps(kernel_history, indent=2))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = expanded_model(
        checkpoint["state_dict"], old_channels, residual_train.shape[1], config["model"]
    ).cuda()
    old_model = GeneratorEncoder(old_channels, **config["model"]).cuda()
    old_model.load_state_dict(checkpoint["state_dict"])
    train_gpu = {
        key: torch.as_tensor(train[key], device="cuda") for key in ["features", "baseline"]
    }
    val_gpu = {key: torch.as_tensor(val[key], device="cuda") for key in ["features", "baseline"]}
    with torch.no_grad():
        old_prediction = old_model(val_gpu["features"][:, :old_channels], val_gpu["baseline"])
        new_prediction = model(val_gpu["features"], val_gpu["baseline"])
        initialization_difference = float((old_prediction - new_prediction).abs().max())
    if initialization_difference > 1e-8:
        raise ValueError("new input changed the initial model")
    del old_model
    supervision, geometry = load_field_supervision(config)
    train_gpu.update(
        {key: torch.as_tensor(value, device="cuda") for key, value in supervision.items()}
    )
    decoder = DifferentiableGenerator(geometry).cuda()
    frequencies = torch.tensor(config["query_frequencies"], device="cuda", dtype=torch.float64)
    roi = torch.tensor(geometry["edges"] >= config["roi_min_range_m"], device="cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options["learning_rate"], weight_decay=0.00001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, options["steps"])
    # Both matched variants instantiate identical modules before sampling batches.
    torch.manual_seed(options["seed"] + 10000)
    history, best, batch_errors = [], float("inf"), []
    print(
        json.dumps(
            {
                "ready_seconds": time.perf_counter() - start,
                "kernel_raw": best_kernel,
                "initialization_max_difference": initialization_difference,
                "feature_mode": options["feature_mode"],
            }
        ),
        flush=True,
    )
    for step in range(options["steps"] + 1):
        if step % options["eval_every"] == 0 or step == options["steps"]:
            with torch.no_grad():
                beta = model(val_gpu["features"], val_gpu["baseline"]).cpu().numpy()
            rows = evaluate(beta, metadata, config)
            record = {
                "step": step,
                "raw": error(rows),
                "train_batch_raw": float(np.mean(batch_errors)) if batch_errors else None,
                "seconds": time.perf_counter() - start,
            }
            history.append(record)
            batch_errors = []
            if record["raw"] < best:
                best = record["raw"]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": record_config,
                        "old_mean": old_mean,
                        "old_std": old_std,
                        "extra_mean": mean,
                        "extra_std": std,
                        "step": step,
                    },
                    output / "best.pt",
                )
                (output / "best_results.json").write_text(json.dumps(rows, indent=2))
                np.savez_compressed(
                    output / "best_coefficients.npz",
                    coefficients=beta,
                    systems=config["validation_systems"],
                )
            (output / "history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(record), flush=True)
        if step == options["steps"]:
            break
        indices = torch.randint(len(train["features"]), (options["batch_size"],), device="cuda")
        fi = torch.randint(len(frequencies), (options["batch_size"],), device="cuda")
        beta = model(train_gpu["features"][indices], train_gpu["baseline"][indices])
        prediction = decoder(
            beta, train_gpu["background"][indices], train_gpu["initial"][indices], frequencies[fi]
        )[:, roi]
        target = train_gpu["target_field"][indices, fi][:, roi]
        relative = (prediction - target).abs().square().sum((1, 2)) / target.abs().square().sum(
            (1, 2)
        )
        loss = relative.mean() * 10000
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        batch_errors.append(float(relative.detach().sqrt().mean()))
    torch.save({"state_dict": model.state_dict(), "config": record_config}, output / "last.pt")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "best_raw": best,
                "kernel_raw": best_kernel,
                "initialization_max_difference": initialization_difference,
                "real_dof": sum(
                    p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
                ),
                "seconds": time.perf_counter() - start,
                "scope": "single-seed original8development only; NOT the independent64 evaluation",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=["coherent_innovation", "zero_control"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.feature_mode:
        config["feature_mode"] = args.feature_mode
    run(config, args.output)


if __name__ == "__main__":
    main()
