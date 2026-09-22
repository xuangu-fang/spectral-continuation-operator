"""Minimal learned context-query model for latent wave-operator identification.

The model observes a rectangularly incomplete source-frequency response table:
all sources at low frequency and six probe sources at high frequency.  It must
answer queries in the missing three-source by four-frequency block for unseen
media.  ``raw`` and ``homogeneous_carrier`` use exactly the same network and
optimization; only the field coordinates differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

CONTEXT_SOURCES = tuple(range(6))
QUERY_SOURCES = tuple(range(6, 9))
LOW_FREQUENCIES = tuple(range(4))
HIGH_FREQUENCIES = tuple(range(4, 8))
CONTEXT_VIEWS = tuple(
    [(source, frequency) for source in range(9) for frequency in LOW_FREQUENCIES]
    + [
        (source, frequency)
        for source in CONTEXT_SOURCES
        for frequency in HIGH_FREQUENCIES
    ]
)
QUERY_VIEWS = tuple(
    (source, frequency)
    for source in QUERY_SOURCES
    for frequency in HIGH_FREQUENCIES
)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.layers(value))


class ContextQueryUNet(nn.Module):
    """A compact spatial latent baseline; the context stem is query independent."""

    def __init__(self, context_channels: int, width: int = 32):
        super().__init__()
        self.context_stem = nn.Sequential(
            nn.Conv2d(context_channels, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            ResidualBlock(width),
        )
        self.enc1 = nn.Sequential(nn.Conv2d(width + 7, width, 3, padding=1), ResidualBlock(width))
        self.enc2 = nn.Sequential(
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
            ResidualBlock(2 * width),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(2 * width, 4 * width, 4, stride=2, padding=1),
            ResidualBlock(4 * width),
            ResidualBlock(4 * width),
        )
        self.up2 = nn.ConvTranspose2d(4 * width, 2 * width, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(4 * width, 2 * width, 3, padding=1), ResidualBlock(2 * width)
        )
        self.up1 = nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(2 * width, width, 3, padding=1), ResidualBlock(width)
        )
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 2, 1)
        )

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        system = self.context_stem(context)
        first = self.enc1(torch.cat((system, query), dim=1))
        second = self.enc2(first)
        latent = self.enc3(second)
        value = self.dec2(torch.cat((self.up2(latent), second), dim=1))
        value = self.dec1(torch.cat((self.up1(value), first), dim=1))
        return self.head(value)


class CrossedQueryDataset(Dataset):
    def __init__(
        self,
        field: np.ndarray,
        indices: np.ndarray,
        sources: np.ndarray,
        frequencies: np.ndarray,
        scale: np.ndarray,
        representation: str,
    ):
        self.field = field
        self.indices = np.asarray(indices)
        self.sources = sources
        self.frequencies = frequencies
        self.scale = scale
        self.representation = representation
        grid = field.shape[-1]
        axis = np.linspace(0.0, 1.0, grid, dtype=np.float32)
        self.yy, self.xx = np.meshgrid(axis, axis, indexing="ij")
        self.carriers = np.stack(
            [
                np.exp(
                    1j
                    * 2.0
                    * np.pi
                    * frequencies[:, None, None]
                    * np.hypot(self.yy - source[0], self.xx - source[1])[None]
                ).astype(np.complex64)
                for source in sources
            ]
        )
        self.context_cache = np.stack(
            [self._context(system) for system in self.indices]
        )

    def _transform(self, value: np.ndarray, source: int, frequency: int) -> np.ndarray:
        value = value / self.scale[frequency]
        if self.representation == "homogeneous_carrier":
            value = value * self.carriers[source, frequency]
        return value

    def _context(self, system: int) -> np.ndarray:
        channels = []
        for source, frequency in CONTEXT_VIEWS:
            value = self._transform(self.field[system, source, frequency], source, frequency)
            channels.extend((value.real, value.imag))
        return np.stack(channels).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices) * len(QUERY_VIEWS)

    def __getitem__(self, index: int):
        local_system = index // len(QUERY_VIEWS)
        query_index = index % len(QUERY_VIEWS)
        system = int(self.indices[local_system])
        source, frequency = QUERY_VIEWS[query_index]
        source_coordinate = self.sources[source]
        dy = self.yy - source_coordinate[0]
        dx = self.xx - source_coordinate[1]
        distance = np.hypot(dy, dx)
        frequency_coordinate = (self.frequencies[frequency] - self.frequencies.min()) / (
            self.frequencies.max() - self.frequencies.min()
        )
        query = np.stack(
            (
                dy,
                dx,
                distance,
                np.exp(-0.5 * (distance / 0.035) ** 2),
                np.full_like(distance, frequency_coordinate),
                self.yy,
                self.xx,
            )
        ).astype(np.float32)
        target = self._transform(
            self.field[system, source, frequency], source, frequency
        )
        target = np.stack((target.real, target.imag)).astype(np.float32)
        return (
            torch.from_numpy(self.context_cache[local_system]),
            torch.from_numpy(query),
            torch.from_numpy(target),
            system,
            source,
            frequency,
        )


class MediumQueryDataset(Dataset):
    """Privileged coefficient-field input used to locate the active bottleneck."""

    def __init__(
        self,
        field: np.ndarray,
        speed: np.ndarray,
        indices: np.ndarray,
        sources: np.ndarray,
        frequencies: np.ndarray,
        scale: np.ndarray,
        representation: str,
        *,
        all_queries: bool,
    ):
        self.field = field
        self.speed = speed
        self.indices = np.asarray(indices)
        self.sources = sources
        self.frequencies = frequencies
        self.scale = scale
        self.representation = representation
        self.query_views = (
            tuple((source, frequency) for source in range(9) for frequency in range(8))
            if all_queries
            else QUERY_VIEWS
        )
        grid = field.shape[-1]
        axis = np.linspace(0.0, 1.0, grid, dtype=np.float32)
        self.yy, self.xx = np.meshgrid(axis, axis, indexing="ij")
        self.carriers = np.stack(
            [
                np.exp(
                    1j * 2.0 * np.pi * frequencies[:, None, None]
                    * np.hypot(self.yy - source[0], self.xx - source[1])[None]
                ).astype(np.complex64)
                for source in sources
            ]
        )

    def __len__(self) -> int:
        return len(self.indices) * len(self.query_views)

    def __getitem__(self, index: int):
        local_system = index // len(self.query_views)
        query_index = index % len(self.query_views)
        system = int(self.indices[local_system])
        source, frequency = self.query_views[query_index]
        source_coordinate = self.sources[source]
        dy = self.yy - source_coordinate[0]
        dx = self.xx - source_coordinate[1]
        distance = np.hypot(dy, dx)
        frequency_coordinate = (self.frequencies[frequency] - self.frequencies.min()) / (
            self.frequencies.max() - self.frequencies.min()
        )
        query = np.stack(
            (
                dy, dx, distance, np.exp(-0.5 * (distance / 0.035) ** 2),
                np.full_like(distance, frequency_coordinate), self.yy, self.xx,
            )
        ).astype(np.float32)
        target = self.field[system, source, frequency] / self.scale[frequency]
        if self.representation == "homogeneous_carrier":
            target = target * self.carriers[source, frequency]
        target = np.stack((target.real, target.imag)).astype(np.float32)
        return (
            torch.from_numpy(self.speed[system, None].astype(np.float32)),
            torch.from_numpy(query),
            torch.from_numpy(target),
            system,
            source,
            frequency,
        )


def _complex_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction - target).square().sum(dim=(1, 2, 3)).sqrt()
    denominator = target.square().sum(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
    return numerator / denominator


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    rows = []
    for context, query, target, system, source, frequency in loader:
        prediction = model(context.to(device), query.to(device))
        errors = _complex_relative_l2(prediction, target.to(device)).cpu().numpy()
        for position, error in enumerate(errors):
            rows.append(
                {
                    "system": int(system[position]),
                    "source_index": int(source[position]),
                    "frequency_index": int(frequency[position]),
                    "complex_rel_l2": float(error),
                }
            )
    return rows


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.dataset, allow_pickle=True)
    field = data["field"]
    sources = data["sources"]
    frequencies = data["frequencies"]
    split = data["split"].astype(str)
    train_indices = np.flatnonzero(split == "train")
    validation_indices = np.flatnonzero(split == "validation")
    test_indices = np.flatnonzero(split == "test")
    scale = np.sqrt(
        np.mean(np.abs(field[train_indices]) ** 2, axis=(0, 1, 3, 4))
    ).astype(np.float32)
    if args.input_mode == "response_context":
        datasets = {
            name: CrossedQueryDataset(
                field, indices, sources, frequencies, scale, args.representation
            )
            for name, indices in (
                ("train", train_indices),
                ("validation", validation_indices),
                ("test", test_indices),
            )
        }
        context_channels = 2 * len(CONTEXT_VIEWS)
    else:
        speed = data["speed"]
        datasets = {
            "train": MediumQueryDataset(
                field, speed, train_indices, sources, frequencies, scale,
                args.representation, all_queries=True,
            ),
            "validation": MediumQueryDataset(
                field, speed, validation_indices, sources, frequencies, scale,
                args.representation, all_queries=False,
            ),
            "test": MediumQueryDataset(
                field, speed, test_indices, sources, frequencies, scale,
                args.representation, all_queries=False,
            ),
        }
        context_channels = 1
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            num_workers=0, generator=generator, drop_last=True,
        ),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size),
    }
    model = ContextQueryUNet(context_channels, args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    iterator = iter(loaders["train"])
    best_validation = math.inf
    best_state = None
    history = []
    start = time.time()
    for step in range(1, args.steps + 1):
        try:
            context, query, target, *_ = next(iterator)
        except StopIteration:
            iterator = iter(loaders["train"])
            context, query, target, *_ = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context.to(device), query.to(device))
        target = target.to(device)
        loss = ((prediction - target) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.evaluate_every == 0 or step == args.steps:
            validation_rows = evaluate(model, loaders["validation"], device)
            validation_error = float(
                np.mean([row["complex_rel_l2"] for row in validation_rows])
            )
            history.append(
                {"step": step, "train_mse": float(loss), "validation_rel_l2": validation_error}
            )
            print(
                f"{args.representation} step {step:5d} mse {float(loss):.5f} "
                f"val {validation_error:.4f}",
                flush=True,
            )
            if validation_error < best_validation:
                best_validation = validation_error
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    test_rows = evaluate(model, loaders["test"], device)
    errors = np.asarray([row["complex_rel_l2"] for row in test_rows])
    summary = {
        "input_mode": args.input_mode,
        "representation": args.representation,
        "seed": args.seed,
        "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_complex_rel_l2": best_validation,
        "test_mean_complex_rel_l2": float(errors.mean()),
        "test_median_complex_rel_l2": float(np.median(errors)),
        "test_p90_complex_rel_l2": float(np.quantile(errors, 0.90)),
        "elapsed_seconds": time.time() - start,
        "field_scale": scale.tolist(),
        "history": history,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state, "summary": summary, "args": vars(args)},
        args.output / f"{args.representation}_checkpoint.pt",
    )
    (args.output / f"{args.representation}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    with (args.output / f"{args.representation}_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=test_rows[0].keys())
        writer.writeheader()
        writer.writerows(test_rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--representation", choices=("raw", "homogeneous_carrier"), required=True
    )
    parser.add_argument(
        "--input-mode", choices=("response_context", "oracle_medium"),
        default="response_context",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--evaluate-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
