import numpy as np

from spectral_continuation.audit_waveguide_boundary_closure import harmonic_tail


def test_quadratic_cosine_tail_is_exact():
    depth = 100.0
    z, weights = np.polynomial.legendre.leggauss(384)
    z, weights = (z + 1) * depth / 2, weights * depth / 2
    q = 0.002 * z**2 + 0.01 * z + 0.5
    slopes = np.array([[0.01, 0.004 * depth + 0.01]])
    expected = harmonic_tail(slopes, depth, 32, 64)[0]
    modes = np.arange(32, 64)
    actual = (
        np.sqrt(2)
        / depth
        * np.sum((weights * q)[:, None] * np.cos(np.pi * z[:, None] * modes / depth), axis=0)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-11)
