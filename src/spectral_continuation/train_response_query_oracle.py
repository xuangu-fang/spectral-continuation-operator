"""Existence test: decode a receiver response from the true medium and query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .response_query import (
    ResponseNormalizer,
    complex_relative_l2,
    fourier_query_features,
    frozen_episodes,
    load_response_archive,
    normalized_query_coordinates,
)


class TrueMediumQueryDecoder(nn.Module):
    """Spatial decoder whose 8x8 output is aligned to the receiver lattice."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(21, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.medium = nn.Sequential(
            nn.Conv2d(3, hidden, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 2 * hidden, 3, stride=2, padding=1),
            nn.GroupNorm(8, 2 * hidden),
            nn.SiLU(),
            nn.Conv2d(2 * hidden, 2 * hidden, 3, stride=2, padding=1),
            nn.GroupNorm(8, 2 * hidden),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(3 * hidden, 2 * hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, 1),
        )

    def forward(self, medium: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = medium.shape
        source_x = 0.5 * (query[:, 0] + 1.0)
        source_y = 0.5 * (query[:, 1] + 1.0)
        axis = torch.linspace(0.0, 1.0, height, device=medium.device)
        row, column = torch.meshgrid(axis, axis, indexing="ij")
        source = torch.exp(
            -((row[None] - source_x[:, None, None]) ** 2 + (column[None] - source_y[:, None, None]) ** 2)
            / (2.0 * 0.025**2)
        )
        frequency = query[:, 2, None, None].expand(batch, height, width)
        features = self.medium(torch.cat((medium, source[:, None], frequency[:, None]), dim=1))
        conditioning = self.query(fourier_query_features(query))[:, :, None, None]
        conditioning = conditioning.expand(-1, -1, features.shape[-2], features.shape[-1])
        return self.output(torch.cat((features, conditioning), dim=1))


@torch.no_grad()
def evaluate(model, archive, normalizer, coordinates, episodes, device):
    model.eval()
    errors = []
    for start in range(0, len(episodes), 128):
        batch = episodes[start : start + 128]
        medium = np.stack([archive.speed[e.system] for e in batch])[:, None]
        query = np.stack([coordinates[e.target_source, e.target_frequency] for e in batch])
        prediction = model(
            torch.from_numpy(medium).to(device), torch.from_numpy(query).to(device)
        ).cpu().numpy().reshape(len(batch), 2, -1)
        for index, episode in enumerate(batch):
            decoded = normalizer.decode(
                prediction[index], np.asarray(episode.target_source), np.asarray(episode.target_frequency)
            )
            truth = archive.responses[
                episode.system, episode.target_source, episode.target_frequency
            ]
            errors.append(float(complex_relative_l2(decoded, truth)))
    return {"mean": float(np.mean(errors)), "median": float(np.median(errors))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    raw = np.load(args.dataset)
    archive = load_response_archive(args.dataset, 64)
    # Attach the true medium without expanding the shared archive data class.
    object.__setattr__(archive, "speed", raw["speed"].astype(np.float32))
    normalizer = ResponseNormalizer(archive)
    coordinates = normalized_query_coordinates(archive)
    train_systems = np.flatnonzero(archive.split == "train")
    validation = frozen_episodes(
        archive, split="validation", regimes=("q3",), episodes_per_system=8, seed=20260924
    )
    model = TrueMediumQueryDecoder(args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for step in range(1, args.steps + 1):
        systems = generator.choice(train_systems, args.batch_size)
        sources = generator.integers(archive.query_shape[0], size=args.batch_size)
        frequencies = generator.integers(archive.query_shape[1], size=args.batch_size)
        medium = torch.from_numpy(archive.speed[systems, None]).to(device)
        query = torch.from_numpy(coordinates[sources, frequencies]).to(device)
        target_complex = archive.responses[systems, sources, frequencies]
        target = normalizer.encode(target_complex, sources, frequencies).reshape(
            args.batch_size, 2, 8, 8
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(medium, query)
            loss = F.smooth_l1_loss(prediction, torch.from_numpy(target).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 1000 == 0 or step == args.steps:
            metric = evaluate(model, archive, normalizer, coordinates, validation, device)
            record = {"step": step, "loss": float(loss.detach()), **metric}
            history.append(record)
            print(json.dumps(record), flush=True)
            if metric["mean"] < best:
                best = metric["mean"]
                torch.save(model.state_dict(), args.output / "best.pt")
            model.train()
    result = {"best_validation": best, "history": history}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
