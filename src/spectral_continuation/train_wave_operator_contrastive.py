"""Cross-source, cross-frequency contrastive learning of wave-system identity."""

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

GROUP_A = tuple((source, frequency) for source in range(6) for frequency in range(4))
GROUP_B = tuple((source, frequency) for source in range(6, 9) for frequency in range(4, 8))


class ViewEncoder(nn.Module):
    def __init__(self, input_channels: int, dimension: int = 128, width: int = 32):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(input_channels, width, 5, stride=2, padding=2),
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.GroupNorm(8, 2 * width), nn.SiLU(),
            nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1),
            nn.GroupNorm(8, 4 * width), nn.SiLU(),
            nn.Conv2d(4 * width, 4 * width, 3, stride=2, padding=1),
            nn.GroupNorm(8, 4 * width), nn.SiLU(),
        )
        self.metadata = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.project = nn.Sequential(
            nn.Linear(4 * width + width, 2 * dimension), nn.SiLU(), nn.Linear(2 * dimension, dimension)
        )
        self.attention = nn.Linear(dimension, 1)

    def forward(self, views: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        batch, count, channels, height, width = views.shape
        image = self.image(views.reshape(batch * count, channels, height, width))
        image = image.mean(dim=(-2, -1))
        meta = self.metadata(metadata.reshape(batch * count, 3))
        token = self.project(torch.cat((image, meta), dim=1)).reshape(batch, count, -1)
        weight = torch.softmax(self.attention(token), dim=1)
        return F.normalize((weight * token).sum(dim=1), dim=1)


def _phase_space_channels(value: np.ndarray) -> np.ndarray:
    amplitude = np.abs(value)
    reference = np.sqrt(np.mean(amplitude**2)) + 1e-8
    log_amplitude = np.log1p(amplitude / reference)
    soft_unit = value / (amplitude + 0.1 * reference)
    horizontal = np.ones_like(value)
    vertical = np.ones_like(value)
    horizontal[:, 1:] = value[:, 1:] * np.conj(value[:, :-1])
    vertical[1:, :] = value[1:, :] * np.conj(value[:-1, :])
    horizontal /= np.abs(horizontal) + 1e-8
    vertical /= np.abs(vertical) + 1e-8
    return np.stack(
        (
            log_amplitude,
            soft_unit.real,
            soft_unit.imag,
            horizontal.real,
            horizontal.imag,
            vertical.real,
            vertical.imag,
        )
    ).astype(np.float32)


def _phase_gradient_channels(
    value: np.ndarray, *, include_amplitude: bool, soft_floor: float = 0.0
) -> np.ndarray:
    amplitude = np.abs(value)
    reference = np.sqrt(np.mean(amplitude**2)) + 1e-8
    # A fixed one-pixel difference is not discretization invariant.  Use the
    # physical offset of one cell on the 64-grid training mesh; e.g. a 128-grid
    # field therefore uses a two-pixel phase increment.
    stride = max(round((value.shape[-1] - 1) / 63.0), 1)
    horizontal = np.ones_like(value)
    vertical = np.ones_like(value)
    horizontal[:, stride:] = value[:, stride:] * np.conj(value[:, :-stride])
    vertical[stride:, :] = value[stride:, :] * np.conj(value[:-stride, :])
    floor = soft_floor * reference**2
    horizontal /= np.abs(horizontal) + floor + 1e-8
    vertical /= np.abs(vertical) + floor + 1e-8
    channels = [horizontal.real, horizontal.imag, vertical.real, vertical.imag]
    if include_amplitude:
        channels.insert(0, np.log1p(amplitude / reference))
    return np.stack(channels).astype(np.float32)


def _multiscale_phase_gradient_channels(
    value: np.ndarray,
    *,
    offsets: tuple[int, ...] = (1, 2, 4),
    soft_floor: float = 0.05,
) -> np.ndarray:
    """Amplitude plus confidence-valued phase advances at fixed physical offsets.

    ``offsets`` are expressed on the 64-grid reference mesh.  They are converted
    to pixel strides on the observed grid, so the represented distances stay
    fixed under a resolution change.
    """

    amplitude = np.abs(value)
    reference = np.sqrt(np.mean(amplitude**2)) + 1e-8
    channels: list[np.ndarray] = [np.log1p(amplitude / reference)]
    resolution_scale = max((value.shape[-1] - 1) / 63.0, 1.0)
    floor = soft_floor * reference**2
    for offset in offsets:
        stride = max(round(offset * resolution_scale), 1)
        horizontal = np.zeros_like(value)
        vertical = np.zeros_like(value)
        horizontal[:, stride:] = value[:, stride:] * np.conj(value[:, :-stride])
        vertical[stride:, :] = value[stride:, :] * np.conj(value[:-stride, :])
        horizontal /= np.abs(horizontal) + floor + 1e-8
        vertical /= np.abs(vertical) + floor + 1e-8
        channels.extend((horizontal.real, horizontal.imag, vertical.real, vertical.imag))
    return np.stack(channels).astype(np.float32)


def _wave_packet_gram_channels_batch(
    value: np.ndarray,
    *,
    frequency: float,
    reference_speed: float = 1.0,
    orientations: int = 8,
    radial_scales: tuple[float, ...] = (1.0,),
    batch_size: int = 32,
) -> np.ndarray:
    """Phase token plus gauge-invariant directional packet Gram features."""

    base = np.stack(
        [
            _phase_gradient_channels(item, include_amplitude=True, soft_floor=0.05)
            for item in value
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height, width = value.shape[-2:]
    ky = torch.fft.fftfreq(height, device=device)
    kx = torch.fft.fftfreq(width, device=device)
    grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
    spacing = 1.0 / (width - 1)
    center_frequency = float(frequency) * spacing / reference_speed
    bandwidth = max(0.45 * center_frequency, 1.0 / width)
    angles = torch.arange(orientations, device=device) * (2.0 * torch.pi / orientations)
    scale = torch.as_tensor(radial_scales, device=device)
    center_y = (center_frequency * scale[:, None] * torch.sin(angles)[None]).reshape(-1)
    center_x = (center_frequency * scale[:, None] * torch.cos(angles)[None]).reshape(-1)
    packet_bandwidth = (
        bandwidth * scale[:, None].expand(-1, orientations).reshape(-1)
    ).clamp_min(1.0 / width)
    filters = torch.exp(
        -0.5
        * (
            (grid_y[None] - center_y[:, None, None]).square()
            + (grid_x[None] - center_x[:, None, None]).square()
        )
        / packet_bandwidth[:, None, None] ** 2
    )
    orientation_edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7))
    edges = []
    for scale_index in range(len(radial_scales)):
        offset = scale_index * orientations
        edges.extend((offset + first, offset + second) for first, second in orientation_edges)
    for scale_index in range(len(radial_scales) - 1):
        first_offset = scale_index * orientations
        second_offset = (scale_index + 1) * orientations
        edges.extend(
            (first_offset + orientation, second_offset + orientation)
            for orientation in range(orientations)
        )
    extra = []
    for start in range(0, len(value), batch_size):
        wave = torch.from_numpy(value[start : start + batch_size]).to(device)
        components = torch.fft.ifft2(torch.fft.fft2(wave)[:, None] * filters[None])
        energy = components.abs().square()
        total = energy.sum(dim=1, keepdim=True).clamp_min(1e-10)
        channels = list((energy / total).unbind(dim=1))
        for first, second in edges:
            cross = components[:, second] * components[:, first].conj()
            denominator = torch.sqrt(energy[:, first] * energy[:, second])
            cross = cross / (denominator + 0.01 * total[:, 0] + 1e-8)
            channels.extend((cross.real, cross.imag))
        extra.append(torch.stack(channels, dim=1).float().cpu().numpy())
    return np.concatenate((base, np.concatenate(extra)), axis=1).astype(np.float32)


class ContrastiveFieldBank:
    def __init__(
        self, data: np.lib.npyio.NpzFile, representation: str,
        *, scale: np.ndarray | None = None, snr_db: float | None = None,
        noise_seed: int = 0,
    ):
        field = data["field"]
        if snr_db is not None:
            field = field.copy()
            generator = np.random.default_rng(noise_seed)
            signal_rms = np.sqrt(
                np.mean(np.abs(field) ** 2, axis=(-2, -1), keepdims=True)
            )
            noise = generator.standard_normal(field.shape) + 1j * generator.standard_normal(field.shape)
            field += noise / np.sqrt(2.0) * signal_rms * 10.0 ** (-snr_db / 20.0)
        self.sources = data["sources"]
        self.frequencies = data["frequencies"]
        self.representation = representation
        train = np.flatnonzero(data["split"].astype(str) == "train")
        if scale is None:
            if len(train) == 0:
                raise ValueError("a field scale is required when the dataset has no training split")
            scale = np.sqrt(np.mean(np.abs(field[train]) ** 2, axis=(0, 1, 3, 4)))
        self.scale = np.asarray(scale)
        grid = field.shape[-1]
        axis = np.linspace(0.0, 1.0, grid)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        values = []
        for source in range(field.shape[1]):
            source_values = []
            distance = np.hypot(yy - self.sources[source, 0], xx - self.sources[source, 1])
            for frequency in range(field.shape[2]):
                value = field[:, source, frequency] / self.scale[frequency]
                if representation == "homogeneous_carrier":
                    value = value * np.exp(
                        1j * 2.0 * np.pi * self.frequencies[frequency] * distance
                    )[None]
                    channels = np.stack((value.real, value.imag), axis=1).astype(np.float32)
                elif representation == "raw":
                    channels = np.stack((value.real, value.imag), axis=1).astype(np.float32)
                elif representation == "raw_normalized":
                    reference = np.sqrt(
                        np.mean(np.abs(value) ** 2, axis=(1, 2), keepdims=True)
                    )
                    normalized = value / (reference + 1e-8)
                    channels = np.stack(
                        (normalized.real, normalized.imag), axis=1
                    ).astype(np.float32)
                elif representation == "phase_space":
                    channels = np.stack([_phase_space_channels(item) for item in value])
                elif representation == "phase_gradient":
                    channels = np.stack(
                        [_phase_gradient_channels(item, include_amplitude=False) for item in value]
                    )
                elif representation == "amplitude_phase_gradient":
                    channels = np.stack(
                        [_phase_gradient_channels(item, include_amplitude=True) for item in value]
                    )
                elif representation == "amplitude_soft_phase_gradient":
                    channels = np.stack(
                        [
                            _phase_gradient_channels(
                                item, include_amplitude=True, soft_floor=0.05
                            )
                            for item in value
                        ]
                    )
                elif representation == "amplitude_multiscale_phase_gradient":
                    channels = np.stack(
                        [_multiscale_phase_gradient_channels(item) for item in value]
                    )
                elif representation == "amplitude_phase_wave_packet_gram":
                    channels = _wave_packet_gram_channels_batch(
                        value,
                        frequency=float(self.frequencies[frequency]),
                    )
                elif representation == "amplitude_phase_multiscale_packet_gram":
                    channels = _wave_packet_gram_channels_batch(
                        value,
                        frequency=float(self.frequencies[frequency]),
                        radial_scales=(0.75, 1.5),
                    )
                elif representation == "amplitude":
                    amplitude = np.abs(value)
                    reference = np.sqrt(np.mean(amplitude**2, axis=(1, 2), keepdims=True))
                    channels = np.log1p(amplitude / (reference + 1e-8))[:, None].astype(np.float32)
                else:
                    raise ValueError(representation)
                source_values.append(channels)
            values.append(np.stack(source_values, axis=1))
        # [source, system, frequency, channel, h, w] -> system major
        self.values = np.stack(values, axis=1)
        self.metadata = np.asarray(
            [
                [
                    self.sources[source, 0], self.sources[source, 1],
                    (self.frequencies[frequency] - self.frequencies.min())
                    / (self.frequencies.max() - self.frequencies.min()),
                ]
                for source in range(field.shape[1])
                for frequency in range(field.shape[2])
            ],
            dtype=np.float32,
        ).reshape(field.shape[1], field.shape[2], 3)

    @property
    def channels(self) -> int:
        return self.values.shape[3]

    def set_tensor(
        self, systems: np.ndarray, views: tuple[tuple[int, int], ...], device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = np.stack(
            [np.stack([self.values[system, source, frequency] for source, frequency in views])
             for system in systems]
        )
        metadata = np.broadcast_to(
            np.stack([self.metadata[source, frequency] for source, frequency in views]),
            (len(systems), len(views), 3),
        ).copy()
        return torch.from_numpy(value).to(device), torch.from_numpy(metadata).to(device)


def _sample_views(
    generator: np.random.Generator, group: tuple[tuple[int, int], ...], count: int
) -> tuple[tuple[int, int], ...]:
    positions = generator.choice(len(group), count, replace=False)
    return tuple(group[int(position)] for position in positions)


@torch.no_grad()
def embed(
    model: nn.Module,
    bank: ContrastiveFieldBank,
    systems: np.ndarray,
    views: tuple[tuple[int, int], ...],
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(systems), batch_size):
        batch = systems[start : start + batch_size]
        values, metadata = bank.set_tensor(batch, views, device)
        output.append(model(values, metadata).cpu().numpy())
    return np.concatenate(output)


def retrieval_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    similarity = first @ second.T
    order = np.argsort(-similarity, axis=1)
    ranks = np.asarray([np.flatnonzero(row == index)[0] + 1 for index, row in enumerate(order)])
    diagonal = np.diag(similarity)
    off_diagonal = similarity[~np.eye(len(similarity), dtype=bool)]
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= min(5, len(ranks)))),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "same_system_cosine": float(diagonal.mean()),
        "different_system_cosine": float(off_diagonal.mean()),
        "cosine_margin": float(diagonal.mean() - off_diagonal.mean()),
    }


def anti_collapse_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Variance and decorrelation regularizer for normalized set embeddings."""

    dimension = first.shape[1]
    target_std = 0.5 / math.sqrt(dimension)
    variance = 0.0
    covariance = 0.0
    for embedding in (first, second):
        centered = embedding - embedding.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.square().mean(dim=0) + 1e-4)
        variance = variance + F.relu(target_std - std).mean()
        gram = centered.T @ centered / max(len(embedding) - 1, 1)
        off_diagonal = gram - torch.diag(torch.diag(gram))
        covariance = covariance + off_diagonal.square().sum() / dimension
    return 10.0 * variance + 0.1 * covariance


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.dataset, allow_pickle=True)
    split = data["split"].astype(str)
    train_systems = np.flatnonzero(split == "train")
    validation_systems = np.flatnonzero(split == "validation")
    test_systems = np.flatnonzero(split == "test")
    bank = ContrastiveFieldBank(data, args.representation)
    model = ViewEncoder(bank.channels, args.dimension, args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    best_top1 = -1.0
    best_state = None
    history = []
    start_time = time.time()
    for step in range(1, args.steps + 1):
        systems = rng.choice(train_systems, args.batch_size, replace=False)
        views_a = _sample_views(rng, GROUP_A, args.views_per_set)
        views_b = _sample_views(rng, GROUP_B, args.views_per_set)
        first, first_metadata = bank.set_tensor(systems, views_a, device)
        second, second_metadata = bank.set_tensor(systems, views_b, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        first_embedding = model(first, first_metadata)
        second_embedding = model(second, second_metadata)
        logits = first_embedding @ second_embedding.T / args.temperature
        labels = torch.arange(len(systems), device=device)
        contrastive = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
        regularizer = anti_collapse_loss(first_embedding, second_embedding)
        loss = contrastive + regularizer
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.evaluate_every == 0 or step == args.steps:
            validation_first = embed(model, bank, validation_systems, GROUP_A, device)
            validation_second = embed(model, bank, validation_systems, GROUP_B, device)
            metrics = retrieval_metrics(validation_first, validation_second)
            history.append({"step": step, "loss": float(loss.detach()), **metrics})
            print(
                f"{args.representation} step {step:5d} loss {float(loss.detach()):.4f} "
                f"top1 {metrics['top1']:.3f} margin {metrics['cosine_margin']:.3f}",
                flush=True,
            )
            if metrics["top1"] > best_top1 or (
                metrics["top1"] == best_top1
                and (best_state is None or metrics["cosine_margin"] > best_state[0])
            ):
                best_top1 = metrics["top1"]
                best_state = (
                    metrics["cosine_margin"],
                    {key: value.detach().cpu() for key, value in model.state_dict().items()},
                )
    assert best_state is not None
    model.load_state_dict(best_state[1])
    first = embed(model, bank, test_systems, GROUP_A, device)
    second = embed(model, bank, test_systems, GROUP_B, device)
    metrics = retrieval_metrics(first, second)
    summary = {
        "representation": args.representation,
        "seed": args.seed,
        "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "test_systems": len(test_systems),
        "chance_top1": 1.0 / len(test_systems),
        **{f"test_{key}": value for key, value in metrics.items()},
        "elapsed_seconds": time.time() - start_time,
        "history": history,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state[1], "summary": summary, "args": vars(args)},
        args.output / f"contrastive_{args.representation}.pt",
    )
    (args.output / f"contrastive_{args.representation}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    similarity = first @ second.T
    with (args.output / f"contrastive_{args.representation}_similarity.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["anchor_system", *map(int, test_systems)])
        for system, row in zip(test_systems, similarity):
            writer.writerow([int(system), *map(float, row)])
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=(
            "raw", "raw_normalized", "homogeneous_carrier", "phase_space", "amplitude",
            "phase_gradient", "amplitude_phase_gradient",
            "amplitude_soft_phase_gradient", "amplitude_multiscale_phase_gradient",
            "amplitude_phase_wave_packet_gram",
            "amplitude_phase_multiscale_packet_gram",
        ),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--evaluate-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--views-per-set", type=int, default=6)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
