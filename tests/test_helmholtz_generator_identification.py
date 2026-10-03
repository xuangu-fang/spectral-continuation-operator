import numpy as np

from spectral_continuation.audit_helmholtz_generator_identification import (
    known_terms,
    laplacian,
    query,
)
from spectral_continuation.coherent_helmholtz_data import environment, solve


def test_known_terms_ignore_hidden_speed_contrast():
    spec = environment(42)
    axis = np.linspace(0, 1, 33)
    first = known_terms(spec, axis)
    second = known_terms({**spec, "contrasts": [999] * 4}, axis)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_real_generator_query_matches_original_discrete_solver():
    spec = environment(42)
    expected, speed, _, _ = solve(spec, 33, [3])
    loss, source = known_terms(spec, np.linspace(0, 1, 33))
    prediction = query(1 / speed**2, source, loss, 3, laplacian(33))
    np.testing.assert_allclose(prediction, expected[0], rtol=1e-11, atol=1e-12)
