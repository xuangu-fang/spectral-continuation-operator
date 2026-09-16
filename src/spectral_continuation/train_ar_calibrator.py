"""Train a sparse broadband calibrator for a stable complex spectral pole."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from .data import load_ocean_npz
from .model import parameter_count
from .spectral_calibrator import InvariantAR1Calibrator, ar1_query, fit_stable_ar1
from .train_representation_poc import build_model, coordinates, sha256_file
from .train_spectral_calibrator import _complex_fields, _relative_squared, _sample_mask


@torch.no_grad()
def _base_predictions(model, visible, context, frequencies):
    state = model.spectral_state(visible, context)
    outputs = []
    for frequency in frequencies:
        query = torch.full(
            (len(visible),), frequency, device=visible.device, dtype=visible.real.dtype
        )
        outputs.append(model.decode(*state[:3], visible, query, state[3], state[4]))
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def _selection_error(model, batch, budget, seed, reference_frequency, offsets, ridge):
    visible, prior, target, context = batch
    generator = torch.Generator().manual_seed(seed)
    mask, _ = _sample_mask(len(visible), visible.shape[-2:], (budget,), generator, visible.device)
    reference, pole = model(visible, prior, target * mask, mask, context)
    target_reference = target[:, -1]
    target_pole = fit_stable_ar1(target, ridge)
    losses = []
    for offset in offsets:
        prediction = ar1_query(reference, pole, reference_frequency, reference_frequency + offset)
        target_query = ar1_query(
            target_reference, target_pole, reference_frequency, reference_frequency + offset
        )
        losses.append(_relative_squared(prediction, target_query))
    return float(torch.stack(losses).mean().sqrt())


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    base_summary = json.loads(args.base_run_summary.read_text(encoding="utf-8"))
    base_checkpoint_path = args.base_run_summary.parent / "best.pt"
    base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    bundle = load_ocean_npz(args.dataset)
    normalization = base_checkpoint["normalization"]
    visible_frequencies = tuple(map(float, base_checkpoint["visible"]))
    calibration_frequencies = tuple(map(float, args.calibration_frequencies.split(",")))
    if len(calibration_frequencies) < 3:
        raise ValueError("broadband AR calibration requires at least three slices")
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
        target = _complex_fields(
            bundle, calibration_frequencies, index, normalization.field_scale
        ).to(device)
        environment = bundle.environment[index]
        environment = (
            environment - normalization.environment_mean[0]
        ) / normalization.environment_std[0]
        query = torch.full((len(index),), visible_frequencies[-1], device=device)
        context = coordinates(environment.to(device), query, normalization.frequency_scale)
        prior = _base_predictions(base_model, visible, context, calibration_frequencies)
        return visible, prior, target, context

    fit = prepare(fit_positions)
    selection = prepare(selection_positions)
    model = InvariantAR1Calibrator(
        len(visible_frequencies),
        bundle.environment.shape[1],
        calibration_count=len(calibration_frequencies),
        width=args.width,
        sensor_token_grid=tuple(args.sensor_token_grid),
        max_log_amplitude_correction=args.max_log_amplitude_correction,
        max_phase_correction=args.max_phase_correction,
        max_log_pole_correction=args.max_log_pole_correction,
        max_pole_angle_correction=args.max_pole_angle_correction,
        ar_ridge=args.ar_ridge,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed + 1_000)
    order_generator = np.random.default_rng(args.seed + 3_000)
    offsets = tuple(map(float, args.rollout_offsets.split(",")))
    reference_frequency = calibration_frequencies[-1]
    best_value = _selection_error(
        model,
        selection,
        args.sensor_budget,
        args.seed + 2_000,
        reference_frequency,
        offsets,
        args.ar_ridge,
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    rows = [{"epoch": 0, "train_rel_l2": None, "selection_rel_l2": best_value}]
    for epoch in range(1, args.epochs + 1):
        order = order_generator.permutation(len(fit_positions))
        model.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = torch.as_tensor(order[start : start + args.batch_size], device=device)
            visible, prior, target, context = (value[indices] for value in fit)
            mask, _ = _sample_mask(
                len(indices), visible.shape[-2:], (args.sensor_budget,), generator, device
            )
            optimizer.zero_grad(set_to_none=True)
            reference, pole = model(visible, prior, target * mask, mask, context)
            target_reference = target[:, -1]
            target_pole = fit_stable_ar1(target, args.ar_ridge)
            offset = offsets[torch.randint(len(offsets), (1,), generator=generator).item()]
            prediction = ar1_query(
                reference, pole, reference_frequency, reference_frequency + offset
            )
            target_query = ar1_query(
                target_reference,
                target_pole,
                reference_frequency,
                reference_frequency + offset,
            )
            loss = _relative_squared(prediction, target_query)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().sqrt()))
        scheduler.step()
        model.eval()
        value = _selection_error(
            model,
            selection,
            args.sensor_budget,
            args.seed + 2_000,
            reference_frequency,
            offsets,
            args.ar_ridge,
        )
        rows.append(
            {"epoch": epoch, "train_rel_l2": float(np.mean(losses)), "selection_rel_l2": value}
        )
        if value < best_value:
            best_value, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:03d}/{args.epochs} train={np.mean(losses):.5f} select={value:.5f}",
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
        "status": "sparse broadband AR calibration; no future-query labels used",
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
    parser.add_argument("--calibration-frequencies", default="101,102,103,104,105,106")
    parser.add_argument("--fit-environments", type=int, default=112)
    parser.add_argument("--sensor-budget", type=int, default=92)
    parser.add_argument("--sensor-token-grid", type=int, nargs=2, default=(2, 4))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-log-amplitude-correction", type=float, default=1.5)
    parser.add_argument("--max-phase-correction", type=float, default=3.141592653589793)
    parser.add_argument("--max-log-pole-correction", type=float, default=0.25)
    parser.add_argument("--max-pole-angle-correction", type=float, default=3.141592653589793)
    parser.add_argument("--ar-ridge", type=float, default=0.01)
    parser.add_argument("--rollout-offsets", default="0,1,2,3,4,5,6")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
