import numpy as np

from spectral_continuation.audit_waveguide_continuous_generator import resample_generator


def test_continuous_basis_is_identity_on_its_original_grid_and_linear_in_latent():
    rng = np.random.default_rng(4)
    mean = rng.normal(size=(8, 4))
    factor = rng.normal(size=(8, 4, 3))
    alpha = rng.normal(size=3)
    beta = mean + np.einsum("rdk,k->rd", factor, alpha)
    np.testing.assert_allclose(resample_generator(beta, 8), beta, rtol=1e-12, atol=1e-12)
    expected = resample_generator(mean, 32) + np.einsum(
        "rdk,k->rd", resample_generator(factor, 32), alpha
    )
    np.testing.assert_allclose(resample_generator(beta, 32), expected, rtol=1e-12, atol=1e-12)


def test_cubic_continuous_basis_reproduces_smooth_polynomial_including_boundaries():
    x = (np.arange(8) + 0.5) / 8
    y = (np.arange(32) + 0.5) / 32
    values = np.stack([x, x**2, x**3], axis=1)
    np.testing.assert_allclose(
        resample_generator(values, 32), np.stack([y, y**2, y**3], axis=1), atol=1e-14
    )
