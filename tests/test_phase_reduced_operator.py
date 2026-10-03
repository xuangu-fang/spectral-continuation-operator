import numpy as np

from spectral_continuation.audit_phase_reduced_operator import (
    discrete_operator,
    projected_query,
    travel_coordinates,
)
from spectral_continuation.coherent_helmholtz_data import environment, solve


def test_teacher_operator_identity_and_snapshot_projection():
    spec = environment(20261003)
    fields, _, _, _ = solve(spec, 33, [3.0, 4.0])
    lap, mass, rhs = discrete_operator(spec, 33)
    u = fields[0, 1:-1, 1:-1].ravel()
    assert np.linalg.norm(lap @ u + (6 * np.pi) ** 2 * mass * u - rhs) / np.linalg.norm(rhs) < 1e-10
    basis = fields[:, 1:-1, 1:-1].reshape(2, -1).T
    for rule in ("galerkin", "least_residual"):
        pred, _, residual = projected_query(lap, mass, rhs, basis, 3.0, rule)
        assert np.linalg.norm(pred - u) / np.linalg.norm(u) < 1e-10
        assert residual < 1e-10


def test_eikonal_source_correction_recovers_homogeneous_phase():
    import pytest

    pytest.importorskip("skfmm")
    spec = environment(20261003)
    spec["contrasts"] = [0.0] * 4
    phases = travel_coordinates(spec, np.linspace(0, 1, 65), include_eikonal=True)
    np.testing.assert_allclose(phases["eikonal_phase"], phases["reference_phase"], atol=1e-13)
