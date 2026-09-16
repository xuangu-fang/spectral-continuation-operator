"""Leakage-safe paired-fidelity and episodic support/query data protocol.

This module intentionally contains no training code.  A low-fidelity archive
provides full context-frequency fields, while a separately generated
high-fidelity archive provides sparse hardware observations at support
frequencies and full query targets.  The archive guard binds both fidelities
to the same ordered physical instances, coordinates, and frequency lattice.

The low-fidelity provenance contract is fail-closed: it must explicitly state
that the fields were produced independently of the high-fidelity target.  A
downsampled, filtered, interpolated, or otherwise transformed high target is
not accepted as low fidelity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE_SCHEMA = "spectral-continuation-paired-fidelity-v1"
ALLOWED_SPLITS = ("train", "validation", "test")
ALLOWED_LOW_FIELD_ORIGINS = {
    "independent_low_fidelity_solver",
    "independent_coarse_mesh_solver",
    "independent_reduced_order_solver",
    "independent_measurement",
}
PSEUDO_LOW_ORIGINS = {
    "high_target_derived",
    "downsampled_high_target",
    "resampled_high_target",
    "filtered_high_target",
    "interpolated_high_target",
    "cropped_high_target",
    "teacher_target_transform",
}
IDENTITY_ARRAY_KEYS = (
    "environment_parameters",
    "environment_parameter_names",
    "source_parameters",
    "source_parameter_names",
    "source_coordinates",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_array_hash(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    if array.dtype.kind in "US":
        for item in array.astype(str).ravel():
            encoded = item.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        digest.update(np.ascontiguousarray(array).view(np.uint8))


def _identity_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        _update_array_hash(digest, name, arrays[name])
    return digest.hexdigest()


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar string")
    return str(array.item())


def _load_provenance(value: np.ndarray, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_scalar_text(value, "provenance_json"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid provenance_json in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"provenance_json must encode an object: {path}")
    return payload


def _parent_rows(provenance: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    parents = provenance.get("parents", [])
    if not isinstance(parents, list) or not all(isinstance(row, dict) for row in parents):
        raise ValueError("provenance parents must be a list of objects")
    return parents


def _guard_low_provenance(provenance: Mapping[str, Any], high_archive_sha256: str) -> None:
    if provenance.get("fidelity_level") != "low":
        raise ValueError("low archive provenance must declare fidelity_level='low'")
    if provenance.get("independent_of_high_target") is not True:
        raise ValueError("low archive must explicitly certify independent_of_high_target=true")
    if provenance.get("derived_from_high_target") is not False:
        raise ValueError("low archive must explicitly certify derived_from_high_target=false")
    origin = str(provenance.get("field_origin", ""))
    if origin in PSEUDO_LOW_ORIGINS:
        raise ValueError(f"pseudo low fidelity is forbidden: field_origin={origin!r}")
    if origin not in ALLOWED_LOW_FIELD_ORIGINS:
        raise ValueError(
            f"unrecognized low-fidelity origin {origin!r}; expected an independently generated source"
        )
    forbidden_operations = {
        "downsample_high_target",
        "resample_high_target",
        "filter_high_target",
        "interpolate_high_target",
        "crop_high_target",
        "transform_teacher_target",
    }
    operations = provenance.get("derivation_operations", [])
    if not isinstance(operations, list):
        raise TypeError("derivation_operations must be a list")
    present = forbidden_operations.intersection(map(str, operations))
    if present:
        raise ValueError(
            f"pseudo low-fidelity derivation operations are forbidden: {sorted(present)}"
        )
    direct_parent_hashes = {
        str(provenance.get(key, ""))
        for key in ("parent_archive_sha256", "source_archive_sha256", "teacher_archive_sha256")
        if provenance.get(key)
    }
    if high_archive_sha256 in direct_parent_hashes:
        raise ValueError("low archive provenance directly references the high-target archive")
    for parent in _parent_rows(provenance):
        parent_hash = str(parent.get("sha256", ""))
        parent_role = str(parent.get("role", "")).lower()
        if parent_hash == high_archive_sha256 or parent_role in {
            "high_target",
            "high_fidelity_target",
            "teacher_target",
        }:
            raise ValueError("low archive provenance contains a high-target parent")


def _guard_high_provenance(provenance: Mapping[str, Any]) -> None:
    if provenance.get("fidelity_level") != "high":
        raise ValueError("high archive provenance must declare fidelity_level='high'")
    origin = provenance.get("field_origin")
    if not isinstance(origin, str) or not origin:
        raise ValueError("high archive provenance requires a non-empty field_origin")
    _parent_rows(provenance)


@dataclass(frozen=True)
class FidelityArchive:
    path: Path
    role: str
    fields: np.ndarray
    environment_ids: np.ndarray
    source_ids: np.ndarray
    split: np.ndarray
    frequencies_hz: np.ndarray
    depths_m: np.ndarray
    ranges_m: np.ndarray
    provenance: dict[str, Any]
    optional_identity_arrays: dict[str, np.ndarray]
    archive_sha256: str

    @property
    def instance_count(self) -> int:
        return len(self.environment_ids)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.fields.shape[2]), int(self.fields.shape[3])


@dataclass(frozen=True)
class PairedFidelityArchives:
    low: FidelityArchive
    high: FidelityArchive
    identity_sha256: str

    @property
    def instance_ids(self) -> np.ndarray:
        return np.asarray(
            [
                f"{environment}::{source}"
                for environment, source in zip(
                    self.low.environment_ids.astype(str),
                    self.low.source_ids.astype(str),
                    strict=True,
                )
            ]
        )


def load_fidelity_archive(path: Path, expected_role: str) -> FidelityArchive:
    """Load one archive and validate its internal metadata contract."""
    if expected_role not in {"low", "high"}:
        raise ValueError("expected_role must be 'low' or 'high'")
    path = Path(path).resolve()
    archive_sha = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema_version",
            "fidelity_role",
            "complex_field",
            "environment_ids",
            "source_ids",
            "split",
            "frequencies_hz",
            "depths_m",
            "ranges_m",
            "provenance_json",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
        schema = _scalar_text(archive["schema_version"], "schema_version")
        role = _scalar_text(archive["fidelity_role"], "fidelity_role")
        fields = np.asarray(archive["complex_field"])
        environment_ids = np.asarray(archive["environment_ids"]).astype(str)
        source_ids = np.asarray(archive["source_ids"]).astype(str)
        split = np.asarray(archive["split"]).astype(str)
        frequencies = np.asarray(archive["frequencies_hz"], dtype=np.float64)
        depths = np.asarray(archive["depths_m"], dtype=np.float64)
        ranges = np.asarray(archive["ranges_m"], dtype=np.float64)
        provenance = _load_provenance(archive["provenance_json"], path)
        optional = {
            key: np.asarray(archive[key]) for key in IDENTITY_ARRAY_KEYS if key in archive.files
        }
    if schema != ARCHIVE_SCHEMA:
        raise ValueError(f"unexpected archive schema {schema!r}: {path}")
    if role != expected_role:
        raise ValueError(f"archive role {role!r} does not match expected {expected_role!r}: {path}")
    if fields.ndim != 5 or fields.shape[-1] != 2:
        raise ValueError("complex_field must have shape [instance,frequency,depth,range,2]")
    if not np.isfinite(fields).all():
        raise ValueError("complex_field contains non-finite values")
    instance_count, frequency_count, depth_count, range_count, _ = fields.shape
    if environment_ids.shape != (instance_count,) or source_ids.shape != (instance_count,):
        raise ValueError("environment_ids/source_ids must have one value per physical instance")
    if split.shape != (instance_count,):
        raise ValueError("split must have shape [instance], never [instance,frequency] or [patch]")
    if (
        frequencies.shape != (frequency_count,)
        or depths.shape != (depth_count,)
        or ranges.shape != (range_count,)
    ):
        raise ValueError("frequency/depth/range coordinates do not match complex_field axes")
    if (
        not np.all(np.diff(frequencies) > 0)
        or not np.all(np.diff(depths) > 0)
        or not np.all(np.diff(ranges) > 0)
    ):
        raise ValueError("frequencies and spatial coordinates must be strictly increasing")
    validate_instance_split(environment_ids, source_ids, split)
    for key, value in optional.items():
        if key.endswith("_names"):
            if value.ndim != 1:
                raise ValueError(f"{key} must be one-dimensional")
        elif value.shape[0] != instance_count:
            raise ValueError(f"{key} must have physical instance as its first axis")
    if expected_role == "high":
        _guard_high_provenance(provenance)
    return FidelityArchive(
        path=path,
        role=role,
        fields=fields,
        environment_ids=environment_ids,
        source_ids=source_ids,
        split=split,
        frequencies_hz=frequencies,
        depths_m=depths,
        ranges_m=ranges,
        provenance=provenance,
        optional_identity_arrays=optional,
        archive_sha256=archive_sha,
    )


def validate_paired_fidelity_archives(low_path: Path, high_path: Path) -> PairedFidelityArchives:
    """Validate physical identity and independent low-fidelity provenance."""
    high = load_fidelity_archive(high_path, "high")
    low = load_fidelity_archive(low_path, "low")
    _guard_low_provenance(low.provenance, high.archive_sha256)
    comparisons = {
        "environment_ids": (low.environment_ids, high.environment_ids),
        "source_ids": (low.source_ids, high.source_ids),
        "split": (low.split, high.split),
        "frequencies_hz": (low.frequencies_hz, high.frequencies_hz),
        "depths_m": (low.depths_m, high.depths_m),
        "ranges_m": (low.ranges_m, high.ranges_m),
    }
    for name, (left, right) in comparisons.items():
        if left.dtype.kind in "f" or right.dtype.kind in "f":
            equal = np.array_equal(left, right)
        else:
            equal = np.array_equal(left.astype(str), right.astype(str))
        if not equal:
            raise ValueError(f"paired-fidelity identity mismatch: {name}")
    if low.fields.shape != high.fields.shape:
        raise ValueError(
            f"paired-fidelity field shape mismatch: {low.fields.shape} vs {high.fields.shape}"
        )
    if set(low.optional_identity_arrays) != set(high.optional_identity_arrays):
        raise ValueError(
            "paired archives expose different optional environment/source identity arrays"
        )
    for name in low.optional_identity_arrays:
        if not np.array_equal(
            low.optional_identity_arrays[name], high.optional_identity_arrays[name]
        ):
            raise ValueError(f"paired-fidelity identity mismatch: {name}")
    identity_arrays = {
        "environment_ids": low.environment_ids,
        "source_ids": low.source_ids,
        "split": low.split,
        "frequencies_hz": low.frequencies_hz,
        "depths_m": low.depths_m,
        "ranges_m": low.ranges_m,
        **low.optional_identity_arrays,
    }
    return PairedFidelityArchives(
        low=low, high=high, identity_sha256=_identity_sha256(identity_arrays)
    )


def validate_instance_split(
    environment_ids: np.ndarray,
    source_ids: np.ndarray,
    split: np.ndarray,
) -> dict[str, Any]:
    """Require one split label for each unique environment/source instance."""
    environment_ids = np.asarray(environment_ids).astype(str)
    source_ids = np.asarray(source_ids).astype(str)
    split = np.asarray(split).astype(str)
    if environment_ids.ndim != 1 or source_ids.shape != environment_ids.shape:
        raise ValueError("environment_ids and source_ids must be matched one-dimensional arrays")
    if split.shape != environment_ids.shape:
        raise ValueError(
            "split must be instance-level with exactly one label per environment/source"
        )
    if np.any(environment_ids == "") or np.any(source_ids == ""):
        raise ValueError("environment/source IDs cannot be empty")
    unknown = set(split.tolist()).difference(ALLOWED_SPLITS)
    if unknown:
        raise ValueError(f"unknown instance split labels: {sorted(unknown)}")
    pairs = np.asarray(
        [
            f"{environment}::{source}"
            for environment, source in zip(environment_ids, source_ids, strict=True)
        ]
    )
    if len(np.unique(pairs)) != len(pairs):
        raise ValueError("environment/source pairs must be unique archive rows")
    return {
        "instance_count": len(pairs),
        "counts": {name: int(np.sum(split == name)) for name in ALLOWED_SPLITS},
        "instance_identity_sha256": _identity_sha256({"instance_ids": pairs, "split": split}),
        "policy": "physical environment/source instance; never frequency or spatial patch",
    }


def make_instance_split(
    environment_ids: np.ndarray,
    source_ids: np.ndarray,
    counts: Mapping[str, int],
    *,
    seed: int,
) -> np.ndarray:
    """Create a deterministic split over physical instances only."""
    if set(counts) != set(ALLOWED_SPLITS):
        raise ValueError(f"counts must define exactly {ALLOWED_SPLITS}")
    if any(int(value) < 0 for value in counts.values()):
        raise ValueError("split counts must be nonnegative")
    count = len(np.asarray(environment_ids))
    if sum(map(int, counts.values())) != count:
        raise ValueError("split counts must sum to the physical instance count")
    # Validate identity shape/uniqueness without imposing pre-existing labels.
    validate_instance_split(environment_ids, source_ids, np.asarray(["train"] * count))
    order = np.random.default_rng(seed).permutation(count)
    result = np.empty(count, dtype="<U10")
    start = 0
    for name in ALLOWED_SPLITS:
        stop = start + int(counts[name])
        result[order[start:stop]] = name
        start = stop
    validate_instance_split(environment_ids, source_ids, result)
    return result


def _inclusive(start: int, stop: int) -> tuple[float, ...]:
    return tuple(float(value) for value in range(start, stop + 1))


@dataclass(frozen=True)
class EpisodeSpec:
    name: str
    context_frequencies_hz: tuple[float, ...]
    support_frequencies_hz: tuple[float, ...]
    query_frequencies_hz: tuple[float, ...]
    evaluation_only: bool = False

    def validate(self) -> None:
        bands = (
            self.context_frequencies_hz,
            self.support_frequencies_hz,
            self.query_frequencies_hz,
        )
        if any(not band for band in bands):
            raise ValueError(f"episode {self.name} contains an empty frequency band")
        if any(tuple(sorted(set(band))) != band for band in bands):
            raise ValueError(f"episode {self.name} frequency bands must be sorted and unique")
        if (
            set(bands[0]) & set(bands[1])
            or set(bands[0]) & set(bands[2])
            or set(bands[1]) & set(bands[2])
        ):
            raise ValueError(f"episode {self.name} context/support/query bands overlap")
        if not (max(bands[0]) < min(bands[1]) < max(bands[1]) < min(bands[2])):
            raise ValueError(
                f"episode {self.name} bands are not ordered context -> support -> query"
            )


def canonical_episode_specs() -> tuple[EpisodeSpec, ...]:
    """Return the frozen A/B/C development and strict sliding windows."""
    specs = (
        EpisodeSpec("A", _inclusive(74, 84), _inclusive(85, 90), _inclusive(91, 96)),
        EpisodeSpec("B", _inclusive(80, 90), _inclusive(91, 96), _inclusive(97, 102)),
        EpisodeSpec("C", _inclusive(84, 94), _inclusive(95, 100), _inclusive(101, 106)),
        EpisodeSpec(
            "strict",
            _inclusive(90, 100),
            _inclusive(101, 106),
            _inclusive(107, 112),
            evaluation_only=True,
        ),
    )
    for spec in specs:
        spec.validate()
    return specs


@dataclass(frozen=True)
class ObservationBudget:
    hardware_k: int
    frequency_m: int
    total_k_times_m: int
    spatial_points: int
    support_tensor_points: int
    episode_high_band_points: int
    spatial_coverage: float
    support_coverage: float
    episode_high_band_coverage: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "hardware_k": self.hardware_k,
            "frequency_m": self.frequency_m,
            "total_k_times_m": self.total_k_times_m,
            "spatial_points": self.spatial_points,
            "support_tensor_points": self.support_tensor_points,
            "episode_high_band_points": self.episode_high_band_points,
            "spatial_coverage": self.spatial_coverage,
            "support_coverage": self.support_coverage,
            "episode_high_band_coverage": self.episode_high_band_coverage,
            "counting_rule": "K fixed hardware locations x m support frequencies",
        }


def observation_budget(
    hardware_k: int,
    frequency_m: int,
    spatial_shape: tuple[int, int],
    *,
    query_frequency_count: int,
) -> ObservationBudget:
    """Describe hardware/frequency budget and unambiguous coverage denominators."""
    depth, ranges = map(int, spatial_shape)
    spatial_points = depth * ranges
    if depth <= 0 or ranges <= 0 or not 0 < hardware_k <= spatial_points:
        raise ValueError("hardware K must be in [1, depth*range]")
    if frequency_m <= 0 or query_frequency_count <= 0:
        raise ValueError("frequency counts must be positive")
    total = int(hardware_k) * int(frequency_m)
    support_points = spatial_points * int(frequency_m)
    high_band_points = spatial_points * (int(frequency_m) + int(query_frequency_count))
    return ObservationBudget(
        hardware_k=int(hardware_k),
        frequency_m=int(frequency_m),
        total_k_times_m=total,
        spatial_points=spatial_points,
        support_tensor_points=support_points,
        episode_high_band_points=high_band_points,
        spatial_coverage=hardware_k / spatial_points,
        support_coverage=total / support_points,
        episode_high_band_coverage=total / high_band_points,
    )


@dataclass(frozen=True)
class EpisodeIndex:
    spec: EpisodeSpec
    context_indices: np.ndarray
    support_indices: np.ndarray
    query_indices: np.ndarray
    hardware_indices: np.ndarray
    instance_positions_by_split: dict[str, np.ndarray]
    budget: ObservationBudget

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "evaluation_only": self.spec.evaluation_only,
            "context_frequencies_hz": list(self.spec.context_frequencies_hz),
            "support_frequencies_hz": list(self.spec.support_frequencies_hz),
            "query_frequencies_hz": list(self.spec.query_frequencies_hz),
            "instance_counts": {
                name: len(values) for name, values in self.instance_positions_by_split.items()
            },
            "split_unit": "environment/source instance",
            "hardware_indices": self.hardware_indices.tolist(),
            "hardware_indices_sha256": _identity_sha256(
                {"hardware_indices": self.hardware_indices}
            ),
            "budget": self.budget.as_dict(),
        }


def _frequency_indices(available: np.ndarray, requested: Sequence[float], name: str) -> np.ndarray:
    indices = []
    for frequency in requested:
        matches = np.flatnonzero(np.isclose(available, frequency, rtol=0.0, atol=1e-9))
        if len(matches) != 1:
            raise ValueError(f"{name} frequency {frequency:g} must occur exactly once in archives")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def build_episode_index(
    pair: PairedFidelityArchives,
    spec: EpisodeSpec,
    hardware_indices: np.ndarray,
) -> EpisodeIndex:
    """Bind one frequency window to instance splits and a fixed hardware mask."""
    spec.validate()
    hardware = np.asarray(hardware_indices)
    if hardware.ndim != 1 or hardware.dtype.kind not in "iu" or len(hardware) == 0:
        raise ValueError("hardware_indices must be a non-empty integer vector")
    hardware = hardware.astype(np.int64, copy=False)
    spatial_points = int(np.prod(pair.high.spatial_shape))
    if (
        len(np.unique(hardware)) != len(hardware)
        or np.any(hardware < 0)
        or np.any(hardware >= spatial_points)
    ):
        raise ValueError("hardware indices must be unique and inside the shared spatial grid")
    context = _frequency_indices(pair.low.frequencies_hz, spec.context_frequencies_hz, "context")
    support = _frequency_indices(pair.high.frequencies_hz, spec.support_frequencies_hz, "support")
    query = _frequency_indices(pair.high.frequencies_hz, spec.query_frequencies_hz, "query")
    positions = {
        name: np.flatnonzero(pair.low.split == name).astype(np.int64) for name in ALLOWED_SPLITS
    }
    budget = observation_budget(
        len(hardware),
        len(support),
        pair.high.spatial_shape,
        query_frequency_count=len(query),
    )
    return EpisodeIndex(
        spec=spec,
        context_indices=context,
        support_indices=support,
        query_indices=query,
        hardware_indices=hardware.copy(),
        instance_positions_by_split=positions,
        budget=budget,
    )


def build_canonical_episode_indices(
    pair: PairedFidelityArchives,
    hardware_indices: np.ndarray,
) -> tuple[EpisodeIndex, ...]:
    return tuple(
        build_episode_index(pair, spec, hardware_indices) for spec in canonical_episode_specs()
    )


def episode_protocol_manifest(
    pair: PairedFidelityArchives,
    hardware_indices: np.ndarray,
) -> dict[str, Any]:
    """Return a JSON-serializable identity, split, episode, and budget manifest."""
    split_audit = validate_instance_split(
        pair.low.environment_ids, pair.low.source_ids, pair.low.split
    )
    episodes = build_canonical_episode_indices(pair, hardware_indices)
    return {
        "schema_version": "spectral-continuation-episode-protocol-v1",
        "paired_identity_sha256": pair.identity_sha256,
        "low_archive": {
            "path": str(pair.low.path),
            "sha256": pair.low.archive_sha256,
            "fidelity_role": pair.low.role,
            "field_origin": pair.low.provenance["field_origin"],
            "independent_of_high_target": pair.low.provenance["independent_of_high_target"],
        },
        "high_archive": {
            "path": str(pair.high.path),
            "sha256": pair.high.archive_sha256,
            "fidelity_role": pair.high.role,
            "field_origin": pair.high.provenance["field_origin"],
        },
        "frequency_lattice_hz": pair.low.frequencies_hz.tolist(),
        "spatial_shape": list(pair.low.spatial_shape),
        "instance_split": split_audit,
        "episodes": [episode.metadata() for episode in episodes],
    }


class PairedFidelityEpisodeDataset:
    """Expose only low context, sparse high support, and full high query targets."""

    def __init__(
        self,
        pair: PairedFidelityArchives,
        episode: EpisodeIndex,
        split: str,
        hardware_indices: np.ndarray | None = None,
    ) -> None:
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"unknown instance split {split!r}")
        if episode.spec.evaluation_only and split == "train":
            raise ValueError(
                "evaluation-only strict episode cannot be constructed on train instances"
            )
        hardware = (
            episode.hardware_indices
            if hardware_indices is None
            else np.asarray(hardware_indices, dtype=np.int64)
        )
        if not np.array_equal(hardware, episode.hardware_indices):
            raise ValueError("hardware_indices differ from the frozen episode hardware identity")
        self.pair = pair
        self.episode = episode
        self.split = split
        self.hardware_indices = hardware.copy()
        self.positions = episode.instance_positions_by_split[split]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item: int) -> dict[str, Any]:
        position = int(self.positions[item])
        low_context = self.pair.low.fields[position, self.episode.context_indices]
        high_support_full = self.pair.high.fields[position, self.episode.support_indices]
        support_flat = high_support_full.reshape(len(self.episode.support_indices), -1, 2)
        high_support = support_flat[:, self.hardware_indices]
        high_query = self.pair.high.fields[position, self.episode.query_indices]
        return {
            "instance_id": self.pair.instance_ids[position],
            "environment_id": self.pair.low.environment_ids[position],
            "source_id": self.pair.low.source_ids[position],
            "low_context": low_context,
            "high_support": high_support,
            "high_query": high_query,
            "hardware_indices": self.hardware_indices.copy(),
            "budget": self.episode.budget.as_dict(),
        }
