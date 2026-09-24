"""Data protocol and compact models for in-context wave response queries.

The physical system is the statistical unit.  A query is a source-frequency
pair and its response is a complex vector on a fixed receiver array.  This
module intentionally keeps episode construction independent of model code so
that every baseline sees exactly the same information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ResponseArchive:
    responses: np.ndarray
    sources: np.ndarray
    frequencies: np.ndarray
    split: np.ndarray
    receiver_rc: np.ndarray
    receiver_xy: np.ndarray

    @property
    def query_shape(self) -> tuple[int, int]:
        return int(self.responses.shape[1]), int(self.responses.shape[2])


@dataclass(frozen=True)
class QueryEpisode:
    system: int
    target_source: int
    target_frequency: int
    context: tuple[tuple[int, int], ...]
    regime: str


def cartesian_receiver_array(grid: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic interior Cartesian receiver array.

    ``count=32`` is a checkerboard subset of the same 8x8 lattice used for
    ``count=64``.  This makes the receiver-count comparison nested.
    """

    if count not in (32, 64):
        raise ValueError("receiver count must be 32 or 64")
    axis = np.rint(np.linspace(4, grid - 5, 8)).astype(np.int64)
    row, column = np.meshgrid(axis, axis, indexing="ij")
    rc = np.stack((row.ravel(), column.ravel()), axis=-1)
    if count == 32:
        keep = (np.indices((8, 8)).sum(axis=0).ravel() % 2) == 0
        rc = rc[keep]
    xy = rc.astype(np.float32) / float(grid - 1)
    return rc, xy


def load_response_archive(
    path: str | Path,
    receiver_count: int = 64,
    receiver_mode: str = "cartesian",
) -> ResponseArchive:
    archive = np.load(path)
    field = archive["field"]
    if receiver_mode == "cartesian":
        rc, xy = cartesian_receiver_array(int(field.shape[-1]), receiver_count)
    elif receiver_mode == "source_array":
        xy = archive["sources"].astype(np.float32)
        rc = np.rint(xy * (field.shape[-1] - 1)).astype(np.int64)
        if receiver_count != len(rc):
            raise ValueError("source_array receiver count must equal the source count")
    else:
        raise ValueError(f"unknown receiver mode: {receiver_mode}")
    responses = field[..., rc[:, 0], rc[:, 1]].astype(np.complex64)
    return ResponseArchive(
        responses=responses,
        sources=archive["sources"].astype(np.float32),
        frequencies=archive["frequencies"].astype(np.float32),
        split=archive["split"].astype(str),
        receiver_rc=rc,
        receiver_xy=xy,
    )


def normalized_query_coordinates(archive: ResponseArchive) -> np.ndarray:
    """Continuous source and log-frequency coordinates in roughly [-1, 1]."""

    sources = 2.0 * archive.sources - 1.0
    log_frequency = np.log(archive.frequencies)
    log_frequency = 2.0 * (log_frequency - log_frequency.min()) / (
        log_frequency.max() - log_frequency.min()
    ) - 1.0
    source_grid = np.repeat(sources[:, None, :], len(log_frequency), axis=1)
    frequency_grid = np.repeat(log_frequency[None, :, None], len(sources), axis=0)
    return np.concatenate((source_grid, frequency_grid), axis=-1).astype(np.float32)


class ResponseNormalizer:
    """Train-only querywise complex centering and RMS scaling."""

    def __init__(self, archive: ResponseArchive, floor_fraction: float = 0.02):
        train = archive.responses[archive.split == "train"]
        self.mean = train.mean(axis=0).astype(np.complex64)
        scale = np.sqrt(np.mean(np.abs(train - self.mean[None]) ** 2, axis=0))
        floor = floor_fraction * float(np.median(scale))
        self.scale = np.maximum(scale, floor).astype(np.float32)

    def encode(self, value: np.ndarray, source: np.ndarray, frequency: np.ndarray) -> np.ndarray:
        centered = value - self.mean[source, frequency]
        normalized = centered / self.scale[source, frequency]
        return np.stack((normalized.real, normalized.imag), axis=-2).astype(np.float32)

    def decode(self, value: np.ndarray, source: np.ndarray, frequency: np.ndarray) -> np.ndarray:
        complex_value = value[..., 0, :] + 1j * value[..., 1, :]
        return (
            self.mean[source, frequency]
            + self.scale[source, frequency] * complex_value
        ).astype(np.complex64)


def _context_candidates(
    source_count: int,
    frequency_count: int,
    target_source: int,
    target_frequency: int,
    regime: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    all_queries = [
        (source, frequency)
        for source in range(source_count)
        for frequency in range(frequency_count)
        if (source, frequency) != (target_source, target_frequency)
    ]
    if regime == "q1":
        required = [(target_source, f) for f in range(frequency_count) if f != target_frequency]
        allowed = [(s, f) for s, f in all_queries if f != target_frequency]
    elif regime == "q2":
        required = [(s, target_frequency) for s in range(source_count) if s != target_source]
        allowed = [(s, f) for s, f in all_queries if s != target_source]
    elif regime == "q3":
        required = []
        allowed = [
            (s, f)
            for s, f in all_queries
            if s != target_source and f != target_frequency
        ]
    else:
        raise ValueError(f"unknown query regime: {regime}")
    return required, allowed


def sample_episode(
    generator: np.random.Generator,
    *,
    system: int,
    source_count: int,
    frequency_count: int,
    budget: int,
    regime: str,
    target: tuple[int, int] | None = None,
) -> QueryEpisode:
    if target is None:
        target = (int(generator.integers(source_count)), int(generator.integers(frequency_count)))
    target_source, target_frequency = target
    required, allowed = _context_candidates(
        source_count, frequency_count, target_source, target_frequency, regime
    )
    if budget > len(allowed):
        raise ValueError("context budget exceeds the admissible query set")
    generator.shuffle(required)
    context: list[tuple[int, int]] = required[:budget]
    remaining = [query for query in allowed if query not in context]
    generator.shuffle(remaining)
    context.extend(remaining[: budget - len(context)])
    return QueryEpisode(
        system=system,
        target_source=target_source,
        target_frequency=target_frequency,
        context=tuple(context),
        regime=regime,
    )


def frozen_episodes(
    archive: ResponseArchive,
    *,
    split: str,
    regimes: tuple[str, ...] = ("q1", "q2", "q3"),
    episodes_per_system: int = 8,
    max_budget: int = 16,
    seed: int = 20260924,
) -> list[QueryEpisode]:
    """Build nested episodes; smaller budgets use a prefix of the same context."""

    generator = np.random.default_rng(seed)
    systems = np.flatnonzero(archive.split == split)
    source_count, frequency_count = archive.query_shape
    episodes = []
    for regime in regimes:
        for system in systems:
            for _ in range(episodes_per_system):
                episodes.append(
                    sample_episode(
                        generator,
                        system=int(system),
                        source_count=source_count,
                        frequency_count=frequency_count,
                        budget=max_budget,
                        regime=regime,
                    )
                )
    return episodes


def validate_episode(episode: QueryEpisode) -> None:
    target_source = episode.target_source
    target_frequency = episode.target_frequency
    if (target_source, target_frequency) in episode.context:
        raise ValueError("target query leaked into context")
    sources = {source for source, _ in episode.context}
    frequencies = {frequency for _, frequency in episode.context}
    if episode.regime == "q1":
        if target_source not in sources or target_frequency in frequencies:
            raise ValueError("invalid Q1 episode")
    elif episode.regime == "q2":
        if target_frequency not in frequencies or target_source in sources:
            raise ValueError("invalid Q2 episode")
    elif episode.regime == "q3":
        if target_source in sources or target_frequency in frequencies:
            raise ValueError("invalid Q3 episode")
    else:
        raise ValueError(f"unknown regime: {episode.regime}")


def fourier_query_features(coordinates: torch.Tensor) -> torch.Tensor:
    """Small continuous embedding without turning discrete IDs into lookup tables."""

    features = [coordinates]
    for scale in (1.0, 2.0, 4.0):
        features.extend((torch.sin(torch.pi * scale * coordinates), torch.cos(torch.pi * scale * coordinates)))
    return torch.cat(features, dim=-1)


class ProbeEncoder(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.response = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.metadata = nn.Sequential(nn.Linear(21, 64), nn.SiLU(), nn.Linear(64, 64))
        self.fuse = nn.Sequential(
            nn.Linear(160, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, response: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch, count = response.shape[:2]
        value = self.response(response.reshape(batch * count, *response.shape[2:]))
        meta = self.metadata(fourier_query_features(query.reshape(batch * count, 3)))
        return self.fuse(torch.cat((value, meta), dim=-1)).reshape(batch, count, -1)


class ResponseQueryCNP(nn.Module):
    """Minimal variable-budget DeepSets conditional neural process."""

    def __init__(self, receiver_count: int = 64, hidden: int = 256, attention: bool = False):
        super().__init__()
        side = round(receiver_count**0.5)
        if side * side != receiver_count:
            raise ValueError("the convolutional model requires a square receiver array")
        self.side = side
        self.hidden = hidden
        self.attention = attention
        self.encoder = ProbeEncoder(hidden)
        self.query = nn.Sequential(nn.Linear(21, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        if attention:
            self.cross_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
            decoder_input = 2 * hidden
        else:
            decoder_input = 3 * hidden
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, 2 * hidden),
            nn.LayerNorm(2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, 2 * receiver_count),
        )

    def forward(
        self,
        context_response: torch.Tensor,
        context_query: torch.Tensor,
        context_mask: torch.Tensor,
        target_query: torch.Tensor,
    ) -> torch.Tensor:
        token = self.encoder(context_response, context_query)
        target = self.query(fourier_query_features(target_query))
        if self.attention:
            attended, _ = self.cross_attention(
                target[:, None], token, token, key_padding_mask=~context_mask.bool()
            )
            latent = torch.cat((target, attended[:, 0]), dim=-1)
        else:
            weight = context_mask.to(token.dtype)[..., None]
            denominator = weight.sum(dim=1).clamp_min(1.0)
            mean = (token * weight).sum(dim=1) / denominator
            second = (token.square() * weight).sum(dim=1) / denominator
            standard_deviation = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-5)
            latent = torch.cat((target, mean, standard_deviation), dim=-1)
        output = self.decoder(latent)
        return output.reshape(output.shape[0], 2, self.side, self.side)


class SpatialResponseQueryOperator(nn.Module):
    """Query-aware set operator that never collapses the receiver lattice.

    The scalar CNP above is deliberately minimal, but its global pooling can
    erase receiver-to-receiver phase relations.  This operator keeps a shared
    spatial feature map for each probe and only aggregates across probes.
    """

    def __init__(self, width: int = 64):
        super().__init__()
        self.width = width
        self.response = nn.Sequential(
            nn.Conv2d(2, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.context_metadata = nn.Sequential(
            nn.Linear(21, width), nn.SiLU(), nn.Linear(width, 2 * width)
        )
        self.target_metadata = nn.Sequential(
            nn.Linear(21, width), nn.SiLU(), nn.Linear(width, 2 * width)
        )
        self.key = nn.Linear(width, width)
        self.decoder = nn.Sequential(
            nn.Conv2d(3 * width, 2 * width, 3, padding=1),
            nn.GroupNorm(8, 2 * width),
            nn.SiLU(),
            nn.Conv2d(2 * width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, 2, 1),
        )

    def forward(
        self,
        context_response: torch.Tensor,
        context_query: torch.Tensor,
        context_mask: torch.Tensor,
        target_query: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, _, height, width = context_response.shape
        feature = self.response(context_response.reshape(batch * count, 2, height, width))
        feature = feature.reshape(batch, count, self.width, height, width)
        scale, shift = self.context_metadata(
            fourier_query_features(context_query).reshape(batch * count, -1)
        ).chunk(2, dim=-1)
        scale = scale.reshape(batch, count, self.width, 1, 1)
        shift = shift.reshape(batch, count, self.width, 1, 1)
        feature = feature * (1.0 + 0.1 * torch.tanh(scale)) + shift

        valid = context_mask.to(feature.dtype)[..., None, None, None]
        denominator = valid.sum(dim=1).clamp_min(1.0)
        mean = (feature * valid).sum(dim=1) / denominator
        second = (feature.square() * valid).sum(dim=1) / denominator
        standard_deviation = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-5)

        target_scale, target_shift = self.target_metadata(
            fourier_query_features(target_query)
        ).chunk(2, dim=-1)
        pooled = feature.mean(dim=(-2, -1))
        score = torch.einsum(
            "bkh,bh->bk", self.key(pooled), target_shift
        ) / self.width**0.5
        score = score.masked_fill(~context_mask.bool(), -torch.finfo(score.dtype).max)
        attention = torch.softmax(score, dim=1)
        attended = torch.einsum("bk,bkchw->bchw", attention, feature)
        attended = attended * (1.0 + 0.1 * torch.tanh(target_scale[:, :, None, None]))
        attended = attended + target_shift[:, :, None, None]
        return self.decoder(torch.cat((mean, standard_deviation, attended), dim=1))


class MaskedResponseTransformer(nn.Module):
    """Jointly reason over observed probe tokens and one masked query token."""

    def __init__(
        self,
        receiver_count: int = 64,
        hidden: int = 256,
        layers: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.receiver_count = receiver_count
        self.response = nn.Sequential(
            nn.Linear(2 * receiver_count, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.query = nn.Sequential(nn.Linear(21, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.observed_token = nn.Parameter(torch.zeros(hidden))
        self.masked_token = nn.Parameter(torch.randn(hidden) / hidden**0.5)
        layer = nn.TransformerEncoderLayer(
            hidden,
            heads,
            dim_feedforward=4 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden))
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, 2 * receiver_count)
        )

    def forward(
        self,
        context_response: torch.Tensor,
        context_query: torch.Tensor,
        context_mask: torch.Tensor,
        target_query: torch.Tensor,
    ) -> torch.Tensor:
        batch, count = context_response.shape[:2]
        response = self.response(context_response.reshape(batch, count, -1))
        context = response + self.query(fourier_query_features(context_query))
        context = context + self.observed_token[None, None]
        target = self.query(fourier_query_features(target_query)) + self.masked_token[None]
        sequence = torch.cat((context, target[:, None]), dim=1)
        padding = torch.cat(
            (
                ~context_mask.bool(),
                torch.zeros(batch, 1, dtype=torch.bool, device=context_mask.device),
            ),
            dim=1,
        )
        encoded = self.transformer(sequence, src_key_padding_mask=padding)
        output = self.decoder(encoded[:, -1])
        side = round(self.receiver_count**0.5)
        return output.reshape(batch, 2, side, side)


def complex_relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(prediction - target, axis=-1)
    denominator = np.linalg.norm(target, axis=-1).clip(min=1e-12)
    return numerator / denominator
