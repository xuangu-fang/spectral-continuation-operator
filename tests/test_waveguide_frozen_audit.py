import json

import pytest

from spectral_continuation.benchmark_waveguide_generator import completed_systems, run
from spectral_continuation.evaluate_frozen_generator_queries import (
    completed_systems as completed_query_systems,
)


def test_frozen_cohort_cannot_be_selected_after_generation(tmp_path):
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps({"systems": [1, 2]}))
    config = {"freeze_manifest": str(manifest), "systems": [1]}
    with pytest.raises(ValueError, match="complete frozen system cohort"):
        run(config, tmp_path / "output")


def test_changed_candidate_is_rejected_before_model_loading(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"changed content")
    manifest = tmp_path / "freeze.json"
    manifest.write_text(
        json.dumps({"systems": [1], "frozen_files_sha256": {str(checkpoint): "wrong"}})
    )
    config = {"freeze_manifest": str(manifest), "systems": [1]}
    with pytest.raises(ValueError, match="changed frozen candidate"):
        run(config, tmp_path / "output")


def test_resume_preserves_only_complete_registered_systems():
    rows = [{"system": 1, "method": "a", "repeat": 0}, {"system": 1, "method": "b", "repeat": 0}]
    expected = {("a", 0), ("b", 0)}
    assert completed_systems(rows, [1, 2], expected) == {1}
    with pytest.raises(ValueError, match="partial/duplicate"):
        completed_systems(rows[:1], [1, 2], expected)
    with pytest.raises(ValueError, match="unregistered"):
        completed_systems(rows, [2], expected)


def test_query_resume_never_drops_a_frequency_or_seed():
    row = {"system": 1, "training_seed": "42", "method": "a", "frequency": 400.0}
    expected = {("42", "a", 400.0)}
    assert completed_query_systems([row], [1, 2], expected) == {1}
    with pytest.raises(ValueError, match="partial/duplicate"):
        completed_query_systems([row, row], [1, 2], expected)
    with pytest.raises(ValueError, match="partial/duplicate"):
        completed_query_systems([row], [1, 2], expected | {("42", "a", 140.0)})
