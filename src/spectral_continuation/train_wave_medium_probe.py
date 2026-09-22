"""Spatial sufficiency probe: recover hidden medium coefficients from response views."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .train_wave_operator_contrastive import GROUP_A, ContrastiveFieldBank


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.net(value))


class MediumUNet(nn.Module):
    def __init__(self, input_channels: int, width: int = 32):
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(input_channels, width, 3, padding=1), ResidualBlock(width))
        self.second = nn.Sequential(
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1), ResidualBlock(2 * width)
        )
        self.third = nn.Sequential(
            nn.Conv2d(2 * width, 4 * width, 4, stride=2, padding=1),
            ResidualBlock(4 * width), ResidualBlock(4 * width),
        )
        self.up2 = nn.ConvTranspose2d(4 * width, 2 * width, 4, stride=2, padding=1)
        self.decode2 = nn.Sequential(nn.Conv2d(4 * width, 2 * width, 3, padding=1), ResidualBlock(2 * width))
        self.up1 = nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1)
        self.decode1 = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1), ResidualBlock(width))
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.first(value)
        second = self.second(first)
        latent = self.third(second)
        value = self.decode2(torch.cat((self.up2(latent), second), dim=1))
        value = self.decode1(torch.cat((self.up1(value), first), dim=1))
        return 0.7 * torch.sigmoid(self.head(value))


def _input_tensor(bank: ContrastiveFieldBank, source_count: int = 6) -> torch.Tensor:
    views = tuple(view for view in GROUP_A if view[0] < source_count)
    values = np.concatenate(
        [bank.values[:, source, frequency] for source, frequency in views], axis=1
    )
    return torch.from_numpy(values.astype(np.float32))


@torch.no_grad()
def evaluate(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
    indices: np.ndarray, device: str, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        predictions.append(model(inputs[batch].to(device)).cpu())
    prediction = torch.cat(predictions)[:, 0].numpy()
    target = targets[indices, 0].numpy()
    numerator = np.linalg.norm((prediction - target).reshape(len(indices), -1), axis=1)
    denominator = np.linalg.norm(target.reshape(len(indices), -1), axis=1)
    return prediction, numerator / np.maximum(denominator, 1e-8)


def _iou(prediction: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    pred_mask = prediction > threshold
    target_mask = target > 1e-6
    intersection = (pred_mask & target_mask).sum(axis=(1, 2))
    union = (pred_mask | target_mask).sum(axis=(1, 2))
    return intersection / np.maximum(union, 1)


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.dataset, allow_pickle=True)
    split = data["split"].astype(str)
    train_indices = np.flatnonzero(split == "train")
    validation_indices = np.flatnonzero(split == "validation")
    test_indices = np.flatnonzero(split == "test")
    bank = ContrastiveFieldBank(
        data, args.representation, snr_db=args.train_snr_db, noise_seed=args.noise_seed
    )
    inputs = _input_tensor(bank, args.source_count)
    targets = torch.from_numpy((1.0 - data["speed"][:, None]).astype(np.float32))
    model = MediumUNet(inputs.shape[1], args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    best_validation = float("inf")
    best_state = None
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        batch = rng.choice(train_indices, args.batch_size, replace=False)
        value = inputs[batch].to(device)
        target = targets[batch].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(value)
        weight = 1.0 + 4.0 * (target > 1e-6)
        loss = ((prediction - target).square() * weight).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.evaluate_every == 0 or step == args.steps:
            validation_prediction, validation_error = evaluate(
                model, inputs, targets, validation_indices, device, args.batch_size
            )
            value = float(validation_error.mean())
            history.append({"step": step, "loss": float(loss.detach()), "validation_rel_l2": value})
            print(
                f"{args.representation} step {step:5d} loss {float(loss.detach()):.5f} "
                f"val {value:.4f}", flush=True,
            )
            if value < best_validation:
                best_validation = value
                best_state = {key: item.detach().cpu() for key, item in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    validation_prediction, _ = evaluate(
        model, inputs, targets, validation_indices, device, args.batch_size
    )
    validation_target = targets[validation_indices, 0].numpy()
    threshold_grid = np.linspace(0.02, 0.30, 29)
    threshold = max(
        threshold_grid,
        key=lambda current: float(_iou(validation_prediction, validation_target, current).mean()),
    )
    prediction, errors = evaluate(model, inputs, targets, test_indices, device, args.batch_size)
    target = targets[test_indices, 0].numpy()
    iou = _iou(prediction, target, float(threshold))
    correlation = np.asarray(
        [np.corrcoef(prediction[index].ravel(), target[index].ravel())[0, 1]
         for index in range(len(test_indices))]
    )
    rows = [
        {
            "system": int(system),
            "contrast_rel_l2": float(errors[position]),
            "inclusion_iou": float(iou[position]),
            "pixel_correlation": float(correlation[position]),
        }
        for position, system in enumerate(test_indices)
    ]
    summary = {
        "representation": args.representation,
        "source_count": args.source_count,
        "train_snr_db": args.train_snr_db,
        "seed": args.seed,
        "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selected_threshold": float(threshold),
        "test_contrast_rel_l2_mean": float(errors.mean()),
        "test_inclusion_iou_mean": float(iou.mean()),
        "test_pixel_correlation_mean": float(np.nanmean(correlation)),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state, "summary": summary, "args": vars(args)},
        args.output / f"medium_probe_{args.representation}.pt",
    )
    (args.output / f"medium_probe_{args.representation}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    np.savez_compressed(
        args.output / f"medium_probe_{args.representation}_predictions.npz",
        system=test_indices, prediction=prediction, target=target,
    )
    with (args.output / f"medium_probe_{args.representation}_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--representation", choices=(
            "raw", "homogeneous_carrier", "amplitude", "phase_space",
            "phase_gradient", "amplitude_phase_gradient",
            "amplitude_soft_phase_gradient",
        ),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--evaluate-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--source-count", type=int, choices=(1, 2, 3, 4, 5, 6), default=6)
    parser.add_argument("--train-snr-db", type=float)
    parser.add_argument("--noise-seed", type=int, default=9173)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
