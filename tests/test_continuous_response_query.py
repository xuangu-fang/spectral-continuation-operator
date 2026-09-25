import numpy as np

from spectral_continuation.audit_continuous_response_query import (
    interpolation_matrix,
    receiver_split,
)


def test_interpolation_matrices_preserve_constant_response():
    observed, query = receiver_split(64, periodic=True)
    for method in ("periodic_linear", "periodic_cubic", "periodic_fourier"):
        matrix = interpolation_matrix(64, observed, query, method)
        assert np.allclose(matrix @ np.ones(len(observed)), 1.0, atol=1e-6)


def test_periodic_fourier_recovers_resolved_complex_mode():
    observed, query = receiver_split(64, periodic=True)
    matrix = interpolation_matrix(64, observed, query, "periodic_fourier")
    mode = 7
    response = np.exp(2j * np.pi * mode * observed / 64)
    truth = np.exp(2j * np.pi * mode * query / 64)
    assert np.allclose(matrix @ response, truth, atol=1e-5)


def test_nonperiodic_split_keeps_both_endpoints_observed():
    observed, query = receiver_split(48, periodic=False)
    assert observed[0] == 0
    assert observed[-1] == 47
    assert not np.intersect1d(observed, query).size
    assert len(observed) + len(query) == 48
