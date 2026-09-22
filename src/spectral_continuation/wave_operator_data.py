"""Controlled multi-source wave responses for latent-operator identification.

The dataset is deliberately organized by *physical system*.  Each random
wavespeed map is probed at the same source-frequency grid, so train/test splits
can be made by medium rather than by response view.  This prevents the common
leakage in which two columns of the same Green operator occur in different
splits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_SOURCES = (
    (0.50, 0.28),
    (0.30, 0.30),
    (0.70, 0.30),
    (0.40, 0.55),
    (0.60, 0.50),
    (0.25, 0.60),
    (0.75, 0.62),
    (0.50, 0.72),
    (0.35, 0.45),
)


def _build_medium(
    grid: int,
    scatterer_fraction: float,
    scatterer_contrast: float,
    seed: int,
    absorption: float,
    geometry_family: str = "circles",
) -> tuple[np.ndarray, np.ndarray]:
    """Create a random inclusion medium and an absorbing boundary layer."""

    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, grid)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    speed = np.ones((grid, grid), dtype=np.float32)
    radius0 = 0.045
    count = max(round(scatterer_fraction / (math.pi * radius0**2)), 1)
    for _ in range(count):
        cx, cy = rng.uniform(0.1, 0.9, size=2)
        radius = radius0 * rng.uniform(0.6, 1.4)
        if geometry_family == "circles":
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
        elif geometry_family == "ellipses":
            angle = rng.uniform(0.0, np.pi)
            first = np.cos(angle) * (xx - cx) + np.sin(angle) * (yy - cy)
            second = -np.sin(angle) * (xx - cx) + np.cos(angle) * (yy - cy)
            first_radius = radius * rng.uniform(0.45, 0.75)
            second_radius = radius * rng.uniform(1.6, 2.4)
            inside = (first / first_radius) ** 2 + (second / second_radius) ** 2 < 1.0
        else:
            raise ValueError(f"unknown geometry family: {geometry_family}")
        speed[inside] = 1.0 - scatterer_contrast
    edge = np.minimum(np.minimum(xx, 1.0 - xx), np.minimum(yy, 1.0 - yy))
    damping = absorption * np.clip(1.0 - edge / 0.22, 0.0, 1.0) ** 2
    return speed, damping.astype(np.float32)


def _ricker(times: np.ndarray, peak_frequency: float) -> np.ndarray:
    delay = 1.2 / peak_frequency
    argument = (np.pi * peak_frequency * (times - delay)) ** 2
    return ((1.0 - 2.0 * argument) * np.exp(-argument)).astype(np.float32)


def _simulate(
    speeds: np.ndarray,
    dampings: np.ndarray,
    source_rc: tuple[int, int],
    peak_frequency: float,
    duration: float,
    *,
    record_every: int = 4,
) -> tuple[np.ndarray, float, float]:
    """Batched differentiable-library-free FDTD; only the generated data are saved."""

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    c = torch.as_tensor(speeds, dtype=torch.float32, device=device)
    sigma = torch.as_tensor(dampings, dtype=torch.float32, device=device)
    grid = c.shape[-1]
    spacing = 1.0 / (grid - 1)
    dt = 0.4 * spacing / float(speeds.max())
    steps = round(duration / dt)
    wavelet = torch.as_tensor(
        _ricker(np.arange(steps) * dt, peak_frequency), device=device
    )
    previous = torch.zeros_like(c)
    current = torch.zeros_like(c)
    coefficient = (c * dt / spacing) ** 2
    half = 0.5 * dt * sigma
    frames = []
    row, column = source_rc
    for step in range(steps):
        laplacian = torch.zeros_like(current)
        laplacian[:, 1:-1, 1:-1] = (
            current[:, 2:, 1:-1]
            + current[:, :-2, 1:-1]
            + current[:, 1:-1, 2:]
            + current[:, 1:-1, :-2]
            - 4.0 * current[:, 1:-1, 1:-1]
        )
        update = 2.0 * current - (1.0 - half) * previous + coefficient * laplacian
        update[:, row, column] += (dt**2) * wavelet[step]
        update /= 1.0 + half
        previous, current = current, update
        if step % record_every == 0:
            frames.append(current)
    return torch.stack(frames, dim=1).cpu().numpy(), dt * record_every, spacing


def generate_dataset(config: dict, output: Path) -> None:
    data_config = config["data"]
    count = int(data_config["media"])
    grid = int(data_config["grid"])
    rng = np.random.default_rng(int(config["seed"]))
    fractions = rng.uniform(*data_config["scatterer_fraction"], count)
    contrasts = rng.uniform(*data_config["scatterer_contrast"], count)
    speeds, dampings = zip(
        *[
            _build_medium(
                grid,
                float(fractions[index]),
                float(contrasts[index]),
                int(config["seed"]) + index,
                float(data_config["boundary_absorption"]),
                str(data_config.get("geometry_family", "circles")),
            )
            for index in range(count)
        ]
    )
    speeds = np.stack(speeds)
    dampings = np.stack(dampings)
    fields = []
    frequencies = None
    for source_index, source in enumerate(data_config["sources"]):
        rc = tuple(np.rint(np.asarray(source) * (grid - 1)).astype(int))
        frames, record_dt, spacing = _simulate(
            speeds,
            dampings,
            rc,
            float(data_config["peak_frequency"]),
            float(data_config["duration"]),
        )
        padded = 2 * frames.shape[1]
        spectrum = np.fft.rfft(frames.astype(np.float64), n=padded, axis=1)
        full_frequencies = np.fft.rfftfreq(padded, record_dt)
        low, high = data_config["frequency_band"]
        keep = np.linspace(
            int(np.searchsorted(full_frequencies, low)),
            int(np.searchsorted(full_frequencies, high)) - 1,
            int(data_config["frequencies"]),
        ).astype(int)
        frequencies = full_frequencies[keep]
        fields.append(spectrum[:, keep].astype(np.complex64))
        print(f"source {source_index + 1}/{len(data_config['sources'])}", flush=True)
    # [source, medium, frequency, row, column] -> system-major layout
    field = np.stack(fields, axis=1)
    split = data_config["split"]
    labels = np.asarray(
        ["train"] * int(split["train"])
        + ["validation"] * int(split["validation"])
        + ["test"] * int(split["test"])
    )
    if len(labels) != count:
        raise ValueError("train/validation/test counts do not sum to media")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        field=field,
        speed=speeds,
        damping=dampings,
        sources=np.asarray(data_config["sources"], dtype=np.float32),
        frequencies=frequencies,
        split=labels,
        spacing=spacing,
        config=json.dumps(config),
    )
    print(f"saved {output} with field shape {field.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    generate_dataset(config, args.output)


if __name__ == "__main__":
    main()
