"""Shared set encoders and frozen probes for complex wave response views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .train_wave_medium_probe import ResidualBlock
from .train_wave_operator_contrastive import ContrastiveFieldBank

GROUP_LOW = tuple((source, frequency) for source in range(6) for frequency in range(4))
GROUP_HIGH = tuple((source, frequency) for source in range(6, 9) for frequency in range(4, 8))


def sample_rectangular_views(
    generator: np.random.Generator,
    *,
    sources: tuple[int, ...],
    frequencies: tuple[int, ...],
    source_count: int,
    frequency_count: int,
) -> tuple[tuple[int, int], ...]:
    """Sample a source-frequency rectangle without imposing an ordering."""

    if not 1 <= source_count <= len(sources):
        raise ValueError("invalid source_count")
    if not 1 <= frequency_count <= len(frequencies):
        raise ValueError("invalid frequency_count")
    selected_sources = generator.choice(sources, source_count, replace=False)
    selected_frequencies = generator.choice(frequencies, frequency_count, replace=False)
    views = [
        (int(source), int(frequency))
        for source in selected_sources
        for frequency in selected_frequencies
    ]
    generator.shuffle(views)
    return tuple(views)


def apply_independent_phase_rotation(
    views: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Rotate every raw complex response view by an independent global phase."""

    if views.shape[2] != 2:
        raise ValueError("phase rotation requires two real/imag channels")
    phase = 2.0 * torch.pi * torch.rand(
        views.shape[0], views.shape[1], 1, 1,
        device=views.device, generator=generator,
    )
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    real = views[:, :, 0] * cosine - views[:, :, 1] * sine
    imaginary = views[:, :, 0] * sine + views[:, :, 1] * cosine
    return torch.stack((real, imaginary), dim=2)


class SharedViewBackbone(nn.Module):
    """Encode every response with exactly the same convolutional weights."""

    def __init__(
        self,
        input_channels: int,
        width: int = 24,
        metadata_conditioning: str = "deep_add",
    ):
        super().__init__()
        self.metadata_conditioning = metadata_conditioning
        self.first = nn.Sequential(
            nn.Conv2d(input_channels, width, 5, padding=2),
            ResidualBlock(width),
        )
        self.second = nn.Sequential(
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
            ResidualBlock(2 * width),
        )
        self.third = nn.Sequential(
            nn.Conv2d(2 * width, 4 * width, 4, stride=2, padding=1),
            ResidualBlock(4 * width),
            ResidualBlock(4 * width),
        )
        if metadata_conditioning == "deep_add":
            self.metadata = nn.Sequential(
                nn.Linear(3, 2 * width), nn.SiLU(), nn.Linear(2 * width, 4 * width)
            )
        elif metadata_conditioning == "multiscale_film":
            self.metadata_first = self._film_network(3, width)
            self.metadata_second = self._film_network(3, 2 * width)
            self.metadata_third = self._film_network(3, 4 * width)
        else:
            raise ValueError(f"unknown metadata conditioning: {metadata_conditioning}")

    @staticmethod
    def _film_network(input_channels: int, channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_channels, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, 2 * channels),
        )

    @staticmethod
    def _film(value: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = conditioning.chunk(2, dim=1)
        return value * (1.0 + 0.1 * torch.tanh(scale[:, :, None, None])) + shift[
            :, :, None, None
        ]

    def forward(
        self, views: torch.Tensor, metadata: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, count, channels, height, width = views.shape
        value = views.reshape(batch * count, channels, height, width)
        flattened_metadata = metadata.reshape(batch * count, 3)
        first = self.first(value)
        if self.metadata_conditioning == "multiscale_film":
            first = self._film(first, self.metadata_first(flattened_metadata))
        second = self.second(first)
        if self.metadata_conditioning == "multiscale_film":
            second = self._film(second, self.metadata_second(flattened_metadata))
        third = self.third(second)
        if self.metadata_conditioning == "multiscale_film":
            third = self._film(third, self.metadata_third(flattened_metadata))
        else:
            conditioning = self.metadata(flattened_metadata)
            third = third + conditioning[:, :, None, None]
        return tuple(
            feature.reshape(batch, count, *feature.shape[1:])
            for feature in (first, second, third)
        )


class MaskedMomentAggregation(nn.Module):
    """Permutation-invariant mean/std aggregation with a learned projection."""

    def __init__(self, channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
        )

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(value.shape[:2], dtype=value.dtype, device=value.device)
        weight = mask.to(value.dtype)[:, :, None, None, None]
        denominator = weight.sum(dim=1).clamp_min(1.0)
        mean = (value * weight).sum(dim=1) / denominator
        second = (value.square() * weight).sum(dim=1) / denominator
        standard_deviation = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-5)
        return self.project(torch.cat((mean, standard_deviation), dim=1))


class MaskedSlotAggregation(nn.Module):
    """Spatial multi-slot attention over an unordered response-view set."""

    def __init__(self, channels: int, slots: int = 4):
        super().__init__()
        self.slots = slots
        self.score = nn.Conv2d(channels, slots, 1)
        self.project = nn.Sequential(
            nn.Conv2d((slots + 2) * channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
        )

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, count, channels, height, width = value.shape
        if mask is None:
            mask = torch.ones(value.shape[:2], dtype=value.dtype, device=value.device)
        valid = mask.to(value.dtype)[:, :, None, None, None]
        denominator = valid.sum(dim=1).clamp_min(1.0)
        mean = (value * valid).sum(dim=1) / denominator
        second = (value.square() * valid).sum(dim=1) / denominator
        standard_deviation = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-5)
        score = self.score(value.reshape(batch * count, channels, height, width))
        score = score.reshape(batch, count, self.slots, height, width)
        score = score.masked_fill(valid == 0, -torch.finfo(score.dtype).max)
        attention = torch.softmax(score, dim=1)
        attended = torch.einsum("bvshw,bvchw->bschw", attention, value)
        attended = attended.reshape(batch, self.slots * channels, height, width)
        return self.project(torch.cat((mean, standard_deviation, attended), dim=1))


class MaskedSlotStackAggregation(nn.Module):
    """Keep moment and attended propagation slots as separate feature channels."""

    def __init__(self, channels: int, slots: int = 4):
        super().__init__()
        self.slots = slots
        self.score = nn.Conv2d(channels, slots, 1)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, count, channels, height, width = value.shape
        if mask is None:
            mask = torch.ones(value.shape[:2], dtype=value.dtype, device=value.device)
        valid = mask.to(value.dtype)[:, :, None, None, None]
        denominator = valid.sum(dim=1).clamp_min(1.0)
        mean = (value * valid).sum(dim=1) / denominator
        second = (value.square() * valid).sum(dim=1) / denominator
        standard_deviation = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-5)
        score = self.score(value.reshape(batch * count, channels, height, width))
        score = score.reshape(batch, count, self.slots, height, width)
        score = score.masked_fill(valid == 0, -torch.finfo(score.dtype).max)
        attention = torch.softmax(score, dim=1)
        attended = torch.einsum("bvshw,bvchw->bschw", attention, value)
        attended = attended.reshape(batch, self.slots * channels, height, width)
        return torch.cat((mean, standard_deviation, attended), dim=1)


class MetadataBasisAggregation(nn.Module):
    """Project an unordered acquisition set onto learned coordinate basis functions."""

    def __init__(self, channels: int, basis_dimension: int = 4):
        super().__init__()
        if basis_dimension < 2:
            raise ValueError("basis_dimension must include a constant and a learned basis")
        self.basis_dimension = basis_dimension
        self.basis = nn.Sequential(
            nn.Linear(3, 2 * basis_dimension),
            nn.SiLU(),
            nn.Linear(2 * basis_dimension, basis_dimension - 1),
        )
        self.normalization = nn.GroupNorm(
            min(8, basis_dimension * channels), basis_dimension * channels
        )

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if metadata is None:
            raise ValueError("metadata basis aggregation requires acquisition coordinates")
        batch, _, channels, height, width = value.shape
        if mask is None:
            mask = torch.ones(value.shape[:2], dtype=value.dtype, device=value.device)
        valid = mask.to(value.dtype)
        learned = torch.tanh(self.basis(metadata))
        basis = torch.cat((torch.ones_like(learned[..., :1]), learned), dim=-1)
        basis = basis * valid[..., None]
        denominator = valid.sum(dim=1).clamp_min(1.0)[:, None, None, None, None]
        projected = torch.einsum("bvr,bvchw->brchw", basis, value) / denominator
        projected = projected.reshape(
            batch, self.basis_dimension * channels, height, width
        )
        return F.silu(self.normalization(projected))


class WaveSetEncoder(nn.Module):
    """Produce spatial and global system representations from an unordered view set."""

    def __init__(
        self,
        input_channels: int,
        width: int = 24,
        dimension: int = 128,
        aggregation: str = "moments",
        slots: int = 4,
        metadata_conditioning: str = "deep_add",
        basis_dimension: int = 4,
    ):
        super().__init__()
        self.width = width
        self.backbone = SharedViewBackbone(input_channels, width, metadata_conditioning)
        if aggregation == "moments":
            aggregation_class = MaskedMomentAggregation
            aggregation_kwargs = {}
        elif aggregation == "slots":
            aggregation_class = MaskedSlotAggregation
            aggregation_kwargs = {"slots": slots}
            feature_multiplier = 1
        elif aggregation == "slot_stack":
            aggregation_class = MaskedSlotStackAggregation
            aggregation_kwargs = {"slots": slots}
            feature_multiplier = slots + 2
        elif aggregation == "coordinate_basis":
            aggregation_class = MetadataBasisAggregation
            aggregation_kwargs = {"basis_dimension": basis_dimension}
            feature_multiplier = basis_dimension
        else:
            raise ValueError(f"unknown aggregation: {aggregation}")
        if aggregation == "moments":
            feature_multiplier = 1
        self.feature_width = feature_multiplier * width
        self.aggregate_first = aggregation_class(width, **aggregation_kwargs)
        self.aggregate_second = aggregation_class(2 * width, **aggregation_kwargs)
        self.aggregate_third = aggregation_class(4 * width, **aggregation_kwargs)
        self.project = nn.Sequential(
            nn.Linear(4 * self.feature_width, 2 * dimension),
            nn.SiLU(),
            nn.Linear(2 * dimension, dimension),
        )

    def spatial(
        self,
        views: torch.Tensor,
        metadata: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, second, third = self.backbone(views, metadata)
        return (
            self.aggregate_first(first, mask, metadata),
            self.aggregate_second(second, mask, metadata),
            self.aggregate_third(third, mask, metadata),
        )

    def forward(
        self,
        views: torch.Tensor,
        metadata: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.global_embedding(self.spatial(views, metadata, mask))

    def global_embedding(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        third = features[-1]
        return F.normalize(self.project(third.mean(dim=(-2, -1))), dim=1)


class FrozenMediumDecoder(nn.Module):
    """Small spatial probe operating on the encoder's aggregated feature pyramid."""

    def __init__(self, width: int = 24):
        super().__init__()
        self.deep = ResidualBlock(4 * width)
        self.up_second = nn.ConvTranspose2d(4 * width, 2 * width, 4, stride=2, padding=1)
        self.decode_second = nn.Sequential(
            nn.Conv2d(4 * width, 2 * width, 3, padding=1), ResidualBlock(2 * width)
        )
        self.up_first = nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1)
        self.decode_first = nn.Sequential(
            nn.Conv2d(2 * width, width, 3, padding=1), ResidualBlock(width)
        )
        self.head = nn.Conv2d(width, 1, 1)

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        first, second, third = features
        value = self.decode_second(torch.cat((self.up_second(self.deep(third)), second), dim=1))
        value = self.decode_first(torch.cat((self.up_first(value), first), dim=1))
        return 0.7 * torch.sigmoid(self.head(value))


class QueryViewDecoder(nn.Module):
    """Decode a held-out response token from a system pyramid and query metadata."""

    def __init__(self, output_channels: int, width: int = 24):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(3, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, 7 * width),
        )
        self.deep = ResidualBlock(4 * width)
        self.up_second = nn.ConvTranspose2d(4 * width, 2 * width, 4, stride=2, padding=1)
        self.decode_second = nn.Sequential(
            nn.Conv2d(4 * width, 2 * width, 3, padding=1), ResidualBlock(2 * width)
        )
        self.up_first = nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1)
        self.decode_first = nn.Sequential(
            nn.Conv2d(2 * width, width, 3, padding=1), ResidualBlock(width)
        )
        self.head = nn.Conv2d(width, output_channels, 1)

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        query_metadata: torch.Tensor,
    ) -> torch.Tensor:
        first, second, third = features
        conditioning = self.query(query_metadata)
        first_query, second_query, third_query = torch.split(
            conditioning, (first.shape[1], second.shape[1], third.shape[1]), dim=1
        )
        first = first + first_query[:, :, None, None]
        second = second + second_query[:, :, None, None]
        third = third + third_query[:, :, None, None]
        value = self.decode_second(torch.cat((self.up_second(self.deep(third)), second), dim=1))
        value = self.decode_first(torch.cat((self.up_first(value), first), dim=1))
        return self.head(value)


@dataclass(frozen=True)
class ViewBudget:
    source_count: int
    frequency_count: int


class WaveSetBank(ContrastiveFieldBank):
    """Typed alias documenting that one bank feeds variable-size view sets."""
