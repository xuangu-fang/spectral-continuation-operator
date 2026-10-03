import numpy as np

from spectral_continuation.audit_waveguide_generator_identification import (
    envelope_derivatives,
    generator_geometry,
    identify,
    latent_from_coefficients,
)
from spectral_continuation.coherent_waveguide_data import environment


def test_analytic_carrier_derivatives_preserve_oscillations():
    r = np.linspace(0, 2000, 17)
    q = np.array([123.4, 1135.2])
    kr = np.array([0.41, 0.12])
    amplitude = np.array([1 + 0.2j, -0.1 + 0.8j])
    values = np.exp(1j * r[:, None] * kr) * amplitude
    a, da, dda = envelope_derivatives(values, r, q, kr, np.eye(2))
    true = np.exp(1j * q[:, None] * kr) * amplitude
    np.testing.assert_allclose(a, true, atol=1e-12)
    np.testing.assert_allclose(da, 1j * kr * true, atol=1e-12)
    np.testing.assert_allclose(dda, -(kr**2) * true, atol=1e-12)


def test_background_geometry_never_reads_hidden_contrasts():
    spec = environment(42)
    first = generator_geometry(spec, 8, 4)
    second = generator_geometry({**spec, "contrasts_m_s": [100, -100, 999]}, 8, 4)
    for key in first:
        np.testing.assert_allclose(first[key], second[key], atol=1e-13)


def test_local_real_identification_and_positive_decoder():
    gram = np.tile(np.eye(4), (8, 1, 1))
    rhs = np.ones((8, 4)) * 0.01
    estimate = identify(gram, rhs, 0)
    np.testing.assert_allclose(estimate, rhs)
    geometry = generator_geometry(environment(42), 8, 4)
    latent, clip = latent_from_coefficients(estimate, np.ones(32), geometry)
    assert clip == 0
    assert np.linalg.eigvalsh(latent["mass"]).min() > 0
