"""Train a true-medium Fourier response operator before coupling a context encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .fourier_response_operator import FourierResponseOperator
from .response_query import complex_relative_l2, frozen_episodes, load_response_archive


def build_inputs(speed, source_xy, frequency, background, frequency_limits):
    batch, height, _ = speed.shape
    axis = np.linspace(0.0, 1.0, height, dtype=np.float32)
    row, column = np.meshgrid(axis, axis, indexing="ij")
    coordinates = np.stack((row, column))[None].repeat(batch, axis=0)
    source = np.exp(
        -(
            (row[None] - source_xy[:, 0, None, None]) ** 2
            + (column[None] - source_xy[:, 1, None, None]) ** 2
        )
        / (2.0 * 0.025**2)
    )
    normalized_frequency = 2.0 * (frequency - frequency_limits[0]) / (
        frequency_limits[1] - frequency_limits[0]
    ) - 1.0
    frequency_channel = np.broadcast_to(normalized_frequency[:, None, None], speed.shape)
    return np.concatenate(
        (
            (1.0 - speed)[:, None],
            source[:, None],
            frequency_channel[:, None],
            coordinates,
            background.real[:, None],
            background.imag[:, None],
            np.abs(background)[:, None],
        ),
        axis=1,
    ).astype(np.float32)


def field_statistics(field, split):
    train = field[split == "train"]
    mean = train.mean(axis=0).astype(np.complex64)
    residual = train - mean[None]
    scale = np.sqrt(np.mean(np.abs(residual) ** 2, axis=(0, 3, 4))).astype(np.float32)
    scale = np.maximum(scale, 0.02 * np.median(scale))
    return mean, scale


@torch.no_grad()
def evaluate(model, archive, field, speed, mean, scale, episodes, device):
    model.eval()
    errors = []
    rc = archive.receiver_rc
    for start in range(0, len(episodes), 32):
        batch = episodes[start : start + 32]
        systems = np.asarray([episode.system for episode in batch])
        sources = np.asarray([episode.target_source for episode in batch])
        frequencies = np.asarray([episode.target_frequency for episode in batch])
        background = mean[sources, frequencies]
        inputs = build_inputs(
            speed[systems],
            archive.sources[sources],
            archive.frequencies[frequencies],
            background,
            (archive.frequencies.min(), archive.frequencies.max()),
        )
        residual = model(torch.from_numpy(inputs).to(device)).cpu().numpy()
        residual = residual[:, 0] + 1j * residual[:, 1]
        prediction = background + scale[sources, frequencies, None, None] * residual
        for index, episode in enumerate(batch):
            pred_receivers = prediction[index, rc[:, 0], rc[:, 1]]
            truth = archive.responses[
                episode.system, episode.target_source, episode.target_frequency
            ]
            errors.append(float(complex_relative_l2(pred_receivers, truth)))
    return {"mean": float(np.mean(errors)), "median": float(np.median(errors))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    data = np.load(args.dataset)
    archive = load_response_archive(args.dataset, 64)
    field = data["field"].astype(np.complex64)
    speed = data["speed"].astype(np.float32)
    mean, scale = field_statistics(field, archive.split)
    train_systems = np.flatnonzero(archive.split == "train")
    validation = frozen_episodes(
        archive, split="validation", regimes=("q3",), episodes_per_system=8, seed=20260924
    )
    model = FourierResponseOperator(8, args.width, args.modes, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=2e-5)
    generator = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for step in range(1, args.steps + 1):
        systems = generator.choice(train_systems, args.batch_size)
        sources = generator.integers(archive.query_shape[0], size=args.batch_size)
        frequencies = generator.integers(archive.query_shape[1], size=args.batch_size)
        background = mean[sources, frequencies]
        inputs = build_inputs(
            speed[systems],
            archive.sources[sources],
            archive.frequencies[frequencies],
            background,
            (archive.frequencies.min(), archive.frequencies.max()),
        )
        target_complex = (field[systems, sources, frequencies] - background) / scale[
            sources, frequencies, None, None
        ]
        target = np.stack((target_complex.real, target_complex.imag), axis=1).astype(np.float32)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(torch.from_numpy(inputs).to(device))
        loss = F.smooth_l1_loss(prediction, torch.from_numpy(target).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 1000 == 0 or step == args.steps:
            metric = evaluate(model, archive, field, speed, mean, scale, validation, device)
            record = {"step": step, "loss": float(loss.detach()), **metric}
            history.append(record)
            print(json.dumps(record), flush=True)
            if metric["mean"] < best:
                best = metric["mean"]
                torch.save(model.state_dict(), args.output / "best.pt")
            model.train()
    (args.output / "result.json").write_text(
        json.dumps({"best_validation": best, "history": history, "args": vars(args)}, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
