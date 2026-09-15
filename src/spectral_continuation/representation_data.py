"""Datasets for the oscillatory representation proof of concept."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import FieldBundle, load_ocean_npz


@dataclass(frozen=True)
class RepresentationNormalization:
    field_scale: float
    environment_mean: torch.Tensor
    environment_std: torch.Tensor
    frequency_scale: float


def fit_representation_normalization(
    bundle: FieldBundle,
    train_positions: np.ndarray,
    visible_frequencies: tuple[float, ...],
    train_queries: tuple[float, ...],
) -> RepresentationNormalization:
    frequencies = visible_frequencies + train_queries
    energies = [
        bundle.fields[frequency][train_positions].double().square().mean()
        for frequency in frequencies
    ]
    environment = bundle.environment[train_positions].double()
    if environment.shape[1] == 0:
        environment_mean = torch.empty(1, 0, 1, 1)
        environment_std = torch.empty(1, 0, 1, 1)
    else:
        environment_mean = environment.mean(dim=(0, 2, 3), keepdim=True).float()
        environment_std = (
            environment.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8).float()
        )
    return RepresentationNormalization(
        field_scale=max(float(torch.stack(energies).mean().sqrt()), 1e-8),
        environment_mean=environment_mean,
        environment_std=environment_std,
        frequency_scale=max(bundle.frequencies),
    )


class FrequencySetDataset(Dataset):
    """Cartesian product of physical instances and target query frequencies."""

    def __init__(
        self,
        bundle: FieldBundle,
        positions: np.ndarray,
        visible_frequencies: tuple[float, ...],
        query_frequencies: tuple[float, ...],
        normalization: RepresentationNormalization,
        *,
        field_only: bool = False,
    ):
        self.bundle = bundle
        self.positions = np.asarray(positions, dtype=np.int64)
        self.visible = visible_frequencies
        self.queries = query_frequencies
        self.normalization = normalization
        self.field_only = field_only

    @property
    def environment_channels(self) -> int:
        return self.bundle.environment.shape[1]

    def __len__(self) -> int:
        return len(self.positions) * len(self.queries)

    def __getitem__(self, index: int):
        query_index = index % len(self.queries)
        position = int(self.positions[index // len(self.queries)])
        frequency = self.queries[query_index]
        visible = torch.stack(
            [
                torch.complex(
                    self.bundle.fields[current][position, 0],
                    self.bundle.fields[current][position, 1],
                )
                for current in self.visible
            ]
        ) / self.normalization.field_scale
        target_field = self.bundle.fields[frequency][position]
        target = torch.complex(target_field[0], target_field[1])
        target = target / self.normalization.field_scale
        environment = (
            self.bundle.environment[position] - self.normalization.environment_mean[0]
        ) / self.normalization.environment_std[0]
        if self.field_only:
            environment = torch.zeros_like(environment)
        return visible, environment, target, torch.tensor(frequency), query_index


def _synthetic_parameters(count: int, seed: int, paths: int = 3) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    return {
        "amplitude": generator.uniform(0.35, 1.0, (count, paths)),
        "phase": generator.uniform(-np.pi, np.pi, (count, paths)),
        "delay": generator.uniform(0.20, 0.34, (count, paths)),
        "slowness": generator.uniform(0.12, 0.25, (count, paths)),
        "direction": generator.uniform(0.0, 2.0 * np.pi, (count, paths)),
        "center_x": generator.uniform(-0.25, 0.25, (count, paths)),
        "center_y": generator.uniform(-0.25, 0.25, (count, paths)),
        "width": generator.uniform(0.35, 0.65, (count, paths)),
        "curvature": generator.uniform(-0.025, 0.025, (count, paths)),
        "decay": generator.uniform(0.0, 0.02, (count, paths)),
    }


def make_synthetic_bundle(
    *,
    train_count: int,
    validation_count: int,
    height: int,
    width: int,
    frequencies: tuple[float, ...],
    seed: int,
) -> FieldBundle:
    """Generate localized multipath fields on a continuous physical domain."""

    count = train_count + validation_count
    parameters = _synthetic_parameters(count, seed)
    y = np.linspace(-0.5, 0.5, height, dtype=np.float64)
    x = np.linspace(-0.5, 0.5, width, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    fields = {}
    for frequency in frequencies:
        value = np.zeros((count, height, width), dtype=np.complex128)
        for path in range(parameters["amplitude"].shape[1]):
            direction = parameters["direction"][:, path, None, None]
            projection = np.cos(direction) * xx + np.sin(direction) * yy
            radial = (xx - parameters["center_x"][:, path, None, None]) ** 2
            radial += (yy - parameters["center_y"][:, path, None, None]) ** 2
            envelope = np.exp(
                -radial / (2.0 * parameters["width"][:, path, None, None] ** 2)
            )
            delay = parameters["delay"][:, path, None, None]
            delay = delay + parameters["slowness"][:, path, None, None] * projection
            delay = delay + parameters["curvature"][:, path, None, None] * radial
            coefficient = parameters["amplitude"][:, path, None, None]
            coefficient = coefficient * np.exp(
                -parameters["decay"][:, path, None, None] * frequency
            )
            coefficient = coefficient * np.exp(
                1j * parameters["phase"][:, path, None, None]
            )
            value += coefficient * envelope * np.exp(1j * 2.0 * np.pi * frequency * delay)
        tensor = np.stack((value.real, value.imag), axis=1).astype(np.float32)
        fields[float(frequency)] = torch.from_numpy(tensor)
    environment = torch.zeros(count, 0, height, width)
    return FieldBundle(
        fields=fields,
        environment=environment,
        ids=np.arange(count),
        split=np.asarray(["train"] * train_count + ["val"] * validation_count),
        frequencies=frequencies,
        lineage={
            "kind": "localized-multipath-synthetic",
            "seed": seed,
            "shape": [height, width],
            "paths": parameters["amplitude"].shape[1],
        },
    )


def load_representation_bundle(domain: str, dataset: Path | None, **synthetic_kwargs):
    if domain == "ocean":
        if dataset is None:
            raise ValueError("--dataset is required for the ocean domain")
        return load_ocean_npz(dataset)
    if domain == "synthetic":
        return make_synthetic_bundle(**synthetic_kwargs)
    raise ValueError(f"unknown domain: {domain}")
