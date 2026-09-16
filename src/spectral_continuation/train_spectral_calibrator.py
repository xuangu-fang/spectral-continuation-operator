"""Train an amortized sparse spectral calibrator without future-query labels."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from .data import load_ocean_npz
from .model import parameter_count
from .spectral_calibrator import (
    InvariantSpectralCalibrator,
    endpoint_carrier_query,
)
from .train_representation_poc import build_model, coordinates, sha256_file


def _complex_fields(bundle, frequencies, positions, scale):
    return (
        torch.stack(
            [
                torch.complex(
                    bundle.fields[frequency][positions, 0],
                    bundle.fields[frequency][positions, 1],
                )
                for frequency in frequencies
            ],
            dim=1,
        )
        / scale
    )


def _relative_squared(prediction, target):
    numerator = (prediction - target).abs().square().flatten(1).sum(1)
    denominator = target.abs().square().flatten(1).sum(1).clamp_min(1e-8)
    return (numerator / denominator).mean()


def _sample_mask(batch, shape, budgets, generator, device):
    height, width = shape
    point_count = height * width
    budget_index = torch.randint(len(budgets), (1,), generator=generator).item()
    budget = budgets[budget_index]
    scores = torch.rand(batch, point_count, generator=generator)
    indices = scores.topk(budget, dim=1, largest=False).indices.to(device)
    mask = torch.zeros(batch, point_count, device=device)
    mask.scatter_(1, indices, 1.0)
    return mask.reshape(batch, 1, height, width), budget


@torch.no_grad()
def _base_predictions(base_model, visible_fields, context, frequencies):
    state = base_model.spectral_state(visible_fields, context)
    outputs = []
    for frequency in frequencies:
        query = torch.full(
            (visible_fields.shape[0],),
            frequency,
            device=visible_fields.device,
            dtype=visible_fields.real.dtype,
        )
        outputs.append(base_model.decode(*state[:3], visible_fields, query, state[3], state[4]))
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def _selection_error(
    model,
    visible,
    prior,
    targets,
    context,
    budget,
    seed,
    calibration_frequencies,
    rollout_offsets,
):
    generator = torch.Generator().manual_seed(seed)
    mask, _ = _sample_mask(len(visible), visible.shape[-2:], (budget,), generator, visible.device)
    sparse = targets * mask
    prediction = model(visible, prior, sparse, mask, context)
    losses = []
    for offset in rollout_offsets:
        query = calibration_frequencies[-1] + offset
        predicted_query = endpoint_carrier_query(prediction, calibration_frequencies, query)
        target_query = endpoint_carrier_query(targets, calibration_frequencies, query)
        losses.append(_relative_squared(predicted_query, target_query))
    return float(torch.stack(losses).mean().sqrt())


def run(args):
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    base_summary = json.loads(args.base_run_summary.read_text(encoding="utf-8"))
    base_checkpoint_path = args.base_run_summary.parent / "best.pt"
    base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    bundle = load_ocean_npz(args.dataset)
    normalization = base_checkpoint["normalization"]
    visible_frequencies = tuple(map(float, base_checkpoint["visible"]))
    calibration_frequencies = tuple(map(float, args.calibration_frequencies.split(",")))
    if len(calibration_frequencies) != 2:
        raise ValueError("the first POC requires exactly two calibration frequencies")
    train_positions = np.flatnonzero(bundle.split == "train")
    fit_positions = train_positions[: args.fit_environments]
    selection_positions = train_positions[args.fit_environments :]
    if len(selection_positions) == 0:
        raise ValueError("fit_environments must leave training environments for selection")

    base_model = build_model(
        Namespace(**base_summary["args"]),
        visible_frequencies,
        bundle.environment.shape[1],
        normalization.frequency_scale,
    ).to(device)
    base_model.load_state_dict(base_checkpoint["model"])
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    def prepare(positions):
        index = torch.as_tensor(positions, dtype=torch.long)
        visible = _complex_fields(bundle, visible_frequencies, index, normalization.field_scale).to(
            device
        )
        targets = _complex_fields(
            bundle, calibration_frequencies, index, normalization.field_scale
        ).to(device)
        environment = bundle.environment[index]
        environment = (
            environment - normalization.environment_mean[0]
        ) / normalization.environment_std[0]
        query = torch.full((len(index),), visible_frequencies[-1], device=device)
        context = coordinates(environment.to(device), query, normalization.frequency_scale)
        prior = _base_predictions(base_model, visible, context, calibration_frequencies)
        return visible, prior, targets, context

    fit = prepare(fit_positions)
    selection = prepare(selection_positions)
    model = InvariantSpectralCalibrator(
        len(visible_frequencies),
        bundle.environment.shape[1],
        calibration_count=2,
        width=args.width,
        sensor_token_grid=tuple(args.sensor_token_grid),
        max_log_amplitude_correction=args.max_log_amplitude_correction,
        max_phase_correction=args.max_phase_correction,
        max_slope_correction_per_hz=args.max_slope_correction_per_hz,
        calibration_frequency_step=(calibration_frequencies[1] - calibration_frequencies[0]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    budgets = tuple(map(int, args.sensor_budgets.split(",")))
    generator = torch.Generator().manual_seed(args.seed + 1_000)
    best_value = _selection_error(
        model,
        *selection,
        args.selection_budget,
        args.seed + 2_000,
        calibration_frequencies,
        tuple(map(float, args.rollout_offsets.split(","))),
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    rows = [{"epoch": 0, "train_rel_l2": None, "selection_rel_l2": best_value}]
    order_generator = np.random.default_rng(args.seed + 3_000)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = order_generator.permutation(len(fit_positions))
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = torch.as_tensor(order[start : start + args.batch_size], device=device)
            visible, prior, targets, context = (value[indices] for value in fit)
            mask, _ = _sample_mask(len(indices), visible.shape[-2:], budgets, generator, device)
            sparse = targets * mask
            optimizer.zero_grad(set_to_none=True)
            prediction = model(visible, prior, sparse, mask, context)
            rollout_offsets = tuple(map(float, args.rollout_offsets.split(",")))
            offset_index = torch.randint(len(rollout_offsets), (1,), generator=generator).item()
            query = calibration_frequencies[-1] + rollout_offsets[offset_index]
            predicted_query = endpoint_carrier_query(prediction, calibration_frequencies, query)
            target_query = endpoint_carrier_query(targets, calibration_frequencies, query)
            loss = _relative_squared(predicted_query, target_query)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().sqrt()))
        scheduler.step()
        model.eval()
        selection_value = _selection_error(
            model,
            *selection,
            args.selection_budget,
            args.seed + 2_000,
            calibration_frequencies,
            tuple(map(float, args.rollout_offsets.split(","))),
        )
        rows.append(
            {
                "epoch": epoch,
                "train_rel_l2": float(np.mean(losses)),
                "selection_rel_l2": selection_value,
            }
        )
        if selection_value < best_value:
            best_value = selection_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:03d}/{args.epochs} train={np.mean(losses):.5f} "
            f"select={selection_value:.5f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    torch.save({"model": best_state}, checkpoint_path)
    with (args.output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "status": "amortized sparse calibrator; no query labels used",
        "base_run_summary": str(args.base_run_summary),
        "base_checkpoint_sha256": sha256_file(base_checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "visible_hz": visible_frequencies,
        "calibration_hz": calibration_frequencies,
        "fit_environment_count": len(fit_positions),
        "selection_environment_count": len(selection_positions),
        "selected_epoch": best_epoch,
        "selection_rel_l2": best_value,
        "parameters": parameter_count(model),
        "elapsed_seconds": time.perf_counter() - started,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-summary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-frequencies", default="105,106")
    parser.add_argument("--fit-environments", type=int, default=96)
    parser.add_argument("--sensor-budgets", default="23,46,92,184,369")
    parser.add_argument("--selection-budget", type=int, default=92)
    parser.add_argument("--rollout-offsets", default="0,1,2,3,4,5,6")
    parser.add_argument("--sensor-token-grid", type=int, nargs=2, default=(4, 8))
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--max-log-amplitude-correction", type=float, default=1.5)
    parser.add_argument("--max-phase-correction", type=float, default=3.141592653589793)
    parser.add_argument("--max-slope-correction-per-hz", type=float, default=3.141592653589793)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
