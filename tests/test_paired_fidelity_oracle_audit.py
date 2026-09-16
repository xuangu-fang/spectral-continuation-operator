from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from spectral_continuation.paired_fidelity_oracle_audit import (
    dense_calibration_ar1_residual,
    evaluate,
    global_rational_residual,
    load_paired_inputs,
    write_outputs,
)
from spectral_continuation.paired_fidelity_protocol import ARCHIVE_SCHEMA

CALIBRATION = (101.0, 102.0, 103.0, 104.0, 105.0, 106.0)
QUERIES = (107.0, 108.0, 109.0, 110.0, 111.0, 112.0)


def _write_fixture(path, *, high: bool, reverse: bool = False) -> None:
    generator = np.random.default_rng(8)
    count, height, width = 6, 3, 4
    frequencies = np.arange(101.0, 113.0)
    low = generator.normal(size=(count, len(frequencies), height, width)) + 1j * generator.normal(
        size=(count, len(frequencies), height, width)
    )
    initial = 0.8 * (
        generator.normal(size=(count, height, width))
        + 1j * generator.normal(size=(count, height, width))
    )
    pole = 0.82 + 0.11j
    discrepancy = np.stack([initial * pole**index for index in range(len(frequencies))], axis=1)
    field = low + discrepancy if high else low
    ids = np.asarray([f"environment_{index:02d}" for index in range(count)])
    if reverse:
        field, ids = field[::-1], ids[::-1]
    two_channel = np.stack((field.real, field.imag), axis=-1).astype(np.float32)
    np.savez_compressed(
        path,
        complex_field=two_channel,
        frequencies_hz=frequencies,
        environment_ids=ids,
        split=np.asarray(["val"] * count),
    )


def _write_formal_fixture(path, *, role: str) -> None:
    _write_fixture(path, high=role == "high")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays.pop("split")
    count, _, height, width, _ = arrays["complex_field"].shape
    if role == "low":
        provenance = {
            "fidelity_level": "low",
            "field_origin": "independent_low_fidelity_solver",
            "independent_of_high_target": True,
            "derived_from_high_target": False,
            "derivation_operations": [],
            "parents": [],
        }
    else:
        provenance = {
            "fidelity_level": "high",
            "field_origin": "independent_high_fidelity_solver",
            "parents": [],
        }
    np.savez_compressed(
        path,
        **arrays,
        schema_version=np.asarray(ARCHIVE_SCHEMA),
        fidelity_role=np.asarray(role),
        source_ids=np.asarray([f"source_{index:02d}" for index in range(count)]),
        split=np.asarray(["validation"] * count),
        depths_m=np.arange(height, dtype=np.float64),
        ranges_m=np.arange(width, dtype=np.float64),
        provenance_json=np.asarray(json.dumps(provenance)),
    )


def test_independent_synthetic_ar1_oracle_passes_gate_and_pairs_by_id(tmp_path) -> None:
    low_path, high_path = tmp_path / "low.npz", tmp_path / "high.npz"
    _write_fixture(low_path, high=False)
    _write_fixture(high_path, high=True, reverse=True)
    inputs = load_paired_inputs(low_path, high_path, CALIBRATION, QUERIES)
    result = evaluate(inputs)

    cheap = result["methods"]["cheap_query"]["macro_107_112"]["complex_rel_l2"]
    ar1 = result["methods"]["dense_calibration_ar1"]["macro_107_112"]["complex_rel_l2"]
    assert inputs.fidelity_independent
    assert ar1 < cheap * 1e-3
    assert result["task_gate"]["passed"]
    assert set(result["task_gate"]["per_query_relative_improvement_percent"]) == {
        "107",
        "108",
        "109",
        "110",
    }


def test_formal_archives_use_strict_protocol_and_composite_instance_identity(tmp_path) -> None:
    low_path, high_path = tmp_path / "formal_low.npz", tmp_path / "formal_high.npz"
    _write_formal_fixture(low_path, role="low")
    _write_formal_fixture(high_path, role="high")
    inputs = load_paired_inputs(low_path, high_path, CALIBRATION, QUERIES, split="validation")
    assert inputs.identity_guard["formal_protocol_validated"]
    assert len(inputs.identity_guard["paired_identity_sha256"]) == 64
    assert inputs.ids[0] == "environment_00::source_00"
    assert evaluate(inputs)["task_gate"]["passed"]


def test_query_target_cannot_change_oracle_prediction(tmp_path) -> None:
    low_path, high_path = tmp_path / "low.npz", tmp_path / "high.npz"
    _write_fixture(low_path, high=False)
    _write_fixture(high_path, high=True)
    inputs = load_paired_inputs(low_path, high_path, CALIBRATION, QUERIES)
    prediction = dense_calibration_ar1_residual(
        inputs.low_calibration, inputs.high_calibration, inputs.low_query
    )
    corrupted = replace(inputs, high_query_target=inputs.high_query_target + (1000.0 + 500.0j))
    repeated = dense_calibration_ar1_residual(
        corrupted.low_calibration, corrupted.high_calibration, corrupted.low_query
    )
    np.testing.assert_array_equal(prediction, repeated)


def test_global_rational_oracle_has_expected_shape_and_no_query_target_access(tmp_path) -> None:
    low_path, high_path = tmp_path / "low.npz", tmp_path / "high.npz"
    _write_fixture(low_path, high=False)
    _write_fixture(high_path, high=True)
    inputs = load_paired_inputs(low_path, high_path, CALIBRATION, QUERIES)
    prediction = global_rational_residual(
        inputs.low_calibration, inputs.high_calibration, inputs.low_query, order=4
    )
    assert prediction.shape == inputs.high_query_target.shape
    corrupted = replace(inputs, high_query_target=np.full_like(inputs.high_query_target, np.nan))
    repeated = global_rational_residual(
        corrupted.low_calibration, corrupted.high_calibration, corrupted.low_query, order=4
    )
    np.testing.assert_array_equal(prediction, repeated)


def test_query_frequency_sensor_overlap_is_rejected(tmp_path) -> None:
    low_path, high_path = tmp_path / "low.npz", tmp_path / "high.npz"
    _write_fixture(low_path, high=False)
    _write_fixture(high_path, high=True)
    with pytest.raises(ValueError, match="sensor leakage"):
        load_paired_inputs(low_path, high_path, CALIBRATION + (107.0,), QUERIES)


def test_same_archive_smoke_is_explicit_and_cannot_pass_gate(tmp_path) -> None:
    archive = tmp_path / "legacy.npz"
    _write_fixture(archive, high=True)
    with pytest.raises(ValueError, match="same-archive"):
        load_paired_inputs(archive, archive, CALIBRATION, QUERIES)
    inputs = load_paired_inputs(
        archive, archive, CALIBRATION, QUERIES, allow_same_archive_smoke=True
    )
    result = evaluate(inputs)
    assert not inputs.fidelity_independent
    assert not result["task_gate"]["passed"]
    assert result["task_gate"]["forced_failure_reason"]


def test_json_and_csv_outputs_include_lineage_metrics_cost_and_gate(tmp_path) -> None:
    low_path, high_path = tmp_path / "low.npz", tmp_path / "high.npz"
    _write_fixture(low_path, high=False)
    _write_fixture(high_path, high=True)
    inputs = load_paired_inputs(low_path, high_path, CALIBRATION, QUERIES)
    result = evaluate(inputs)
    artifacts = write_outputs(tmp_path / "output", inputs, result, None, "synthetic-test")

    summary = (tmp_path / "output" / "summary.json").read_text(encoding="utf-8")
    curve = (tmp_path / "output" / "query_curve.csv").read_text(encoding="utf-8")
    assert artifacts["summary_sha256"] and artifacts["query_curve_sha256"]
    assert '"high_fidelity_query_sensor_access": false' in summary
    assert '"solver_cost"' in summary
    assert '"oracle_gap"' in summary
    assert '"task_gate"' in summary
    assert "macro_107_112" in curve and "weighted_phase_mae_rad" in curve
