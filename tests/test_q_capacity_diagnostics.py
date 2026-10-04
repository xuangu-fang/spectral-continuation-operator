import numpy as np
import pytest

from spectral_continuation.audit_q768_map_dev import coordinate_solver
from spectral_continuation.plot_q_capacity_audit import mean_raw


def test_map_coordinate_solver_recovers_exact_full_rank_code():
    rng = np.random.default_rng(8)
    factor = rng.normal(size=(5, 4, 6))
    mean = rng.normal(size=(5, 4))
    expected = rng.normal(size=6)
    beta = mean + np.einsum("rdk,k->rd", factor, expected)
    np.testing.assert_allclose(coordinate_solver(factor)(beta, mean), expected,
                               rtol=1e-12, atol=1e-12)


def test_capacity_plot_rejects_incomplete_pairing():
    rows = [{"source": "original", "method": "q", "frequency": 800,
             "complex_rel_l2": 0.1}]
    with pytest.raises(ValueError, match="sixteen"):
        mean_raw(rows, source="original", method="q")
