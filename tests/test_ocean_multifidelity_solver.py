import json

import numpy as np
import pytest

from spectral_continuation.audit_ocean_multifidelity_solver import passes_gate
from spectral_continuation.generate_ocean_multifidelity_pilot import atomic_savez
from spectral_continuation.ocean_multifidelity_solver import (
    field_metrics,
    load_config,
    parse_environments,
    parse_grid,
    parse_solver,
)


def test_field_metrics_exact_and_global_phase_rotation():
    reference = np.ones((3, 5), dtype=np.complex128)
    exact = field_metrics(reference, reference)
    assert exact["complex_rel_l2"] == pytest.approx(0.0)
    assert exact["phase_coherence"] == pytest.approx(1.0)

    rotated = reference * np.exp(1j * np.pi / 3)
    metrics = field_metrics(rotated, reference)
    assert metrics["phase_coherence"] == pytest.approx(0.5)
    assert metrics["weighted_phase_mae_rad"] == pytest.approx(np.pi / 3)


def test_config_contract_rejects_postprocessed_or_invalid_fidelity(tmp_path):
    config = {
        "schema_version": 1,
        "high_fidelity": {"family": "array_downsample"},
        "low_fidelity_candidates": [],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="reference"):
        load_config(path)
    with pytest.raises(ValueError, match="ngrid"):
        parse_solver({"solver_id": "bad", "family": "pekeris_mode", "ngrid": 2})


def test_native_grid_archive_and_go_gate(tmp_path):
    path = tmp_path / "pair.npz"
    low = np.zeros((2, 4, 8, 2), dtype=np.float32)
    high = np.ones_like(low)
    atomic_savez(path, low=low, high=high)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["low"].shape == archive["high"].shape == (2, 4, 8, 2)
    summary = {
        "median_wall_time_speedup": 2.0,
        "median_allocation_reduction": 1.5,
        "median_complex_rel_l2": 0.5,
        "median_phase_coherence": 0.8,
    }
    gate = {
        "median_wall_time_speedup_min": 1.5,
        "median_allocation_reduction_min": 1.2,
        "median_complex_rel_l2_max": 0.8,
        "median_phase_coherence_min": 0.65,
    }
    assert passes_gate(summary, gate)


def test_repository_config_is_small_and_independently_solved():
    config = load_config("configs/ocean_multifidelity_pilot_20260916.json")
    grid = parse_grid(config)
    cases = parse_environments(config, "pilot_environments")
    assert grid.depth_steps * grid.range_steps <= 4096
    assert len(cases) == 3
    assert all(item["family"].startswith("pekeris_") for item in config["low_fidelity_candidates"])
