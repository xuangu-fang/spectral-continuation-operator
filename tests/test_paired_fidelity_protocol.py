from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectral_continuation.paired_fidelity_protocol import (
    ARCHIVE_SCHEMA,
    PairedFidelityEpisodeDataset,
    build_canonical_episode_indices,
    canonical_episode_specs,
    episode_protocol_manifest,
    make_instance_split,
    observation_budget,
    sha256_file,
    validate_instance_split,
    validate_paired_fidelity_archives,
)


def provenance(role: str) -> dict:
    if role == "low":
        return {
            "fidelity_level": "low",
            "field_origin": "independent_coarse_mesh_solver",
            "independent_of_high_target": True,
            "derived_from_high_target": False,
            "derivation_operations": [],
            "parents": [],
        }
    return {
        "fidelity_level": "high",
        "field_origin": "independent_high_fidelity_solver",
        "parents": [],
    }


def write_archive(
    path: Path,
    role: str,
    *,
    changed: str | None = None,
    provenance_override: dict | None = None,
) -> None:
    frequencies = np.arange(74.0, 113.0)
    depths = np.asarray([5.0, 10.0, 15.0])
    ranges = np.asarray([100.0, 200.0, 300.0, 400.0])
    environments = np.asarray(["env-0", "env-1", "env-2", "env-3", "env-4", "env-5"])
    sources = np.asarray(["src-0", "src-1", "src-2", "src-3", "src-4", "src-5"])
    split = np.asarray(["train", "train", "train", "validation", "validation", "test"])
    environment_parameters = np.arange(12, dtype=np.float64).reshape(6, 2)
    source_parameters = np.arange(6, dtype=np.float64)[:, None]
    if changed == "environment":
        environments[2] = "other"
    elif changed == "source":
        sources[2] = "other"
    elif changed == "frequency":
        frequencies[3] += 0.1
    elif changed == "depth":
        depths[1] += 0.1
    elif changed == "range":
        ranges[1] += 0.1
    elif changed == "split":
        split[2] = "test"
    elif changed == "environment_parameters":
        environment_parameters[1, 0] += 0.5
    generator = np.random.default_rng(11 if role == "low" else 12)
    field = generator.normal(size=(6, len(frequencies), len(depths), len(ranges), 2)).astype(
        np.float32
    )
    np.savez(
        path,
        schema_version=np.asarray(ARCHIVE_SCHEMA),
        fidelity_role=np.asarray(role),
        complex_field=field,
        environment_ids=environments,
        source_ids=sources,
        split=split,
        frequencies_hz=frequencies,
        depths_m=depths,
        ranges_m=ranges,
        environment_parameters=environment_parameters,
        environment_parameter_names=np.asarray(["c", "depth"]),
        source_parameters=source_parameters,
        source_parameter_names=np.asarray(["source_depth"]),
        provenance_json=np.asarray(json.dumps(provenance_override or provenance(role))),
    )


def paired_fixture(tmp_path: Path):
    low, high = tmp_path / "low.npz", tmp_path / "high.npz"
    write_archive(low, "low")
    write_archive(high, "high")
    return low, high


def test_identity_guard_accepts_independently_generated_matched_archives(tmp_path: Path):
    low, high = paired_fixture(tmp_path)
    pair = validate_paired_fidelity_archives(low, high)
    assert pair.low.fields.shape == pair.high.fields.shape == (6, 39, 3, 4, 2)
    assert pair.low.archive_sha256 != pair.high.archive_sha256
    assert len(pair.identity_sha256) == 64
    assert pair.instance_ids.tolist() == [f"env-{index}::src-{index}" for index in range(6)]


@pytest.mark.parametrize(
    "changed,expected",
    [
        ("environment", "environment_ids"),
        ("source", "source_ids"),
        ("frequency", "frequencies_hz"),
        ("depth", "depths_m"),
        ("range", "ranges_m"),
        ("split", "split"),
        ("environment_parameters", "environment_parameters"),
    ],
)
def test_identity_guard_rejects_mismatched_archives(tmp_path: Path, changed: str, expected: str):
    low, high = paired_fixture(tmp_path)
    write_archive(low, "low", changed=changed)
    with pytest.raises(ValueError, match=expected):
        validate_paired_fidelity_archives(low, high)


@pytest.mark.parametrize(
    "update,expected",
    [
        ({"derived_from_high_target": True}, "derived_from_high_target"),
        ({"independent_of_high_target": False}, "independent_of_high_target"),
        ({"field_origin": "downsampled_high_target"}, "pseudo low fidelity"),
        ({"derivation_operations": ["filter_high_target"]}, "derivation operations"),
    ],
)
def test_low_provenance_rejects_pseudo_fidelity(tmp_path: Path, update: dict, expected: str):
    low, high = paired_fixture(tmp_path)
    payload = provenance("low")
    payload.update(update)
    write_archive(low, "low", provenance_override=payload)
    with pytest.raises(ValueError, match=expected):
        validate_paired_fidelity_archives(low, high)


def test_low_provenance_rejects_high_archive_parent(tmp_path: Path):
    low, high = paired_fixture(tmp_path)
    payload = provenance("low")
    payload["parents"] = [{"sha256": sha256_file(high), "role": "high_target"}]
    write_archive(low, "low", provenance_override=payload)
    with pytest.raises(ValueError, match="high-target parent"):
        validate_paired_fidelity_archives(low, high)


def test_canonical_sliding_windows_are_exact_and_disjoint():
    specs = {spec.name: spec for spec in canonical_episode_specs()}
    assert specs["A"].context_frequencies_hz == tuple(map(float, range(74, 85)))
    assert specs["A"].support_frequencies_hz == tuple(map(float, range(85, 91)))
    assert specs["A"].query_frequencies_hz == tuple(map(float, range(91, 97)))
    assert specs["B"].context_frequencies_hz == tuple(map(float, range(80, 91)))
    assert specs["B"].support_frequencies_hz == tuple(map(float, range(91, 97)))
    assert specs["B"].query_frequencies_hz == tuple(map(float, range(97, 103)))
    assert specs["C"].context_frequencies_hz == tuple(map(float, range(84, 95)))
    assert specs["C"].support_frequencies_hz == tuple(map(float, range(95, 101)))
    assert specs["C"].query_frequencies_hz == tuple(map(float, range(101, 107)))
    assert specs["strict"].context_frequencies_hz == tuple(map(float, range(90, 101)))
    assert specs["strict"].support_frequencies_hz == tuple(map(float, range(101, 107)))
    assert specs["strict"].query_frequencies_hz == tuple(map(float, range(107, 113)))
    assert specs["strict"].evaluation_only
    for spec in specs.values():
        assert not set(spec.context_frequencies_hz) & set(spec.support_frequencies_hz)
        assert not set(spec.support_frequencies_hz) & set(spec.query_frequencies_hz)


def test_budget_metadata_reports_k_m_product_and_coverages():
    budget = observation_budget(4, 6, (3, 4), query_frequency_count=6)
    assert budget.hardware_k == 4
    assert budget.frequency_m == 6
    assert budget.total_k_times_m == 24
    assert budget.spatial_coverage == pytest.approx(4 / 12)
    assert budget.support_coverage == pytest.approx(24 / (12 * 6))
    assert budget.episode_high_band_coverage == pytest.approx(24 / (12 * 12))


def test_episode_dataset_preserves_instance_split_and_hides_unsampled_support(tmp_path: Path):
    low, high = paired_fixture(tmp_path)
    pair = validate_paired_fidelity_archives(low, high)
    hardware = np.asarray([0, 3, 5, 11])
    episodes = {
        episode.spec.name: episode for episode in build_canonical_episode_indices(pair, hardware)
    }
    assert [
        len(episodes["A"].instance_positions_by_split[name])
        for name in ("train", "validation", "test")
    ] == [3, 2, 1]
    sample = PairedFidelityEpisodeDataset(pair, episodes["A"], "validation", hardware)[0]
    assert sample["instance_id"] == "env-3::src-3"
    assert sample["low_context"].shape == (11, 3, 4, 2)
    assert sample["high_support"].shape == (6, 4, 2)
    assert sample["high_query"].shape == (6, 3, 4, 2)
    assert sample["budget"]["hardware_k"] == 4
    assert sample["budget"]["frequency_m"] == 6
    assert sample["budget"]["total_k_times_m"] == 24
    with pytest.raises(ValueError, match="hardware identity"):
        PairedFidelityEpisodeDataset(pair, episodes["A"], "validation", np.asarray([0, 2, 5, 11]))
    with pytest.raises(ValueError, match="evaluation-only"):
        PairedFidelityEpisodeDataset(pair, episodes["strict"], "train")
    manifest = episode_protocol_manifest(pair, hardware)
    assert [row["name"] for row in manifest["episodes"]] == ["A", "B", "C", "strict"]
    assert all(row["budget"]["total_k_times_m"] == 24 for row in manifest["episodes"])
    assert json.loads(json.dumps(manifest))["instance_split"]["counts"]["test"] == 1


def test_split_is_deterministic_and_rejects_frequency_or_duplicate_instance_splits():
    environments = np.asarray([f"e-{index}" for index in range(8)])
    sources = np.asarray([f"s-{index}" for index in range(8)])
    first = make_instance_split(
        environments, sources, {"train": 4, "validation": 2, "test": 2}, seed=19
    )
    second = make_instance_split(
        environments, sources, {"train": 4, "validation": 2, "test": 2}, seed=19
    )
    assert np.array_equal(first, second)
    assert {name: int(np.sum(first == name)) for name in ("train", "validation", "test")} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    with pytest.raises(ValueError, match="instance-level"):
        validate_instance_split(environments, sources, np.full((8, 3), "train"))
    duplicated = environments.copy()
    duplicated[1] = duplicated[0]
    duplicate_sources = sources.copy()
    duplicate_sources[1] = duplicate_sources[0]
    with pytest.raises(ValueError, match="unique archive rows"):
        validate_instance_split(duplicated, duplicate_sources, first)
