import numpy as np

from spectral_continuation.audit_waveguide_generator_identification import (
    generator_geometry,
    true_projection,
)
from spectral_continuation.audit_waveguide_spectral_exact_basis import analytic_prior
from spectral_continuation.audit_waveguide_spectral_exact_nonlinear import (
    exact_family_projection,
)
from spectral_continuation.audit_waveguide_spectral_taper_shift import shifted_environment
from spectral_continuation.coherent_waveguide_data import spectral_environment


def test_exact_speed_family_projection_matches_oracle_and_directional_derivative():
    spec = spectral_environment(20264600, terms=16, perturbation_rms_m_s=15)
    geometry = generator_geometry(spec, cells=8, depth_terms=8, modes=8, quadrature_nodes=80)
    index = np.arange(16)
    taper = np.exp(-0.5 * (index / 5) ** 2)
    cosine_norm = np.where(index == 0, 1.0, 0.5)
    variance = np.outer(taper**2 * cosine_norm, taper**2 * cosine_norm)
    variance[0, 0] = 0
    speed_scale = spec["spectral_perturbation_rms_m_s"] / np.sqrt(variance.sum())
    physical = np.asarray(spec["spectral_coefficients_m_s"])
    coordinates = (physical / (speed_scale * np.outer(taper, taper))).ravel()[1:]
    beta, factor = exact_family_projection(spec, geometry, coordinates, speed_scale)
    np.testing.assert_allclose(beta, true_projection(spec, geometry), rtol=1e-12, atol=1e-12)
    changed = {key: value for key, value in spec.items()
               if key != "spectral_coefficients_m_s"}
    changed_beta, changed_factor = exact_family_projection(
        changed, geometry, coordinates, speed_scale
    )
    np.testing.assert_array_equal(beta, changed_beta)
    np.testing.assert_array_equal(factor, changed_factor)
    direction = np.random.default_rng(19).normal(size=len(coordinates))
    direction /= np.linalg.norm(direction)
    h = 1e-4
    forward, _ = exact_family_projection(spec, geometry, coordinates + h * direction,
                                         speed_scale)
    backward, _ = exact_family_projection(spec, geometry, coordinates - h * direction,
                                          speed_scale)
    finite_difference = (forward - backward) / (2 * h)
    analytic = np.einsum("rdk,k->rd", factor, direction)
    np.testing.assert_allclose(finite_difference, analytic, rtol=1e-6, atol=1e-10)


def test_exact_speed_family_projection_matches_shifted_taper_and_legacy_default():
    spec = shifted_environment(20267000, terms=16, rms=15.0, taper_scale=10.0)
    geometry = generator_geometry(spec, cells=64, depth_terms=32, modes=32)
    config = {"range_cosine_terms": 16, "depth_cosine_terms": 16,
              "depth_terms": 32, "taper_scale": 10.0}
    _, _, speed_scale = analytic_prior(spec, config)
    taper = np.exp(-0.5 * (np.arange(16) / 10.0) ** 2)
    physical = np.asarray(spec["spectral_coefficients_m_s"])
    coordinates = (physical / (speed_scale * np.outer(taper, taper))).ravel()[1:]
    beta, factor = exact_family_projection(
        spec, geometry, coordinates, speed_scale, taper_scale=10.0
    )
    np.testing.assert_allclose(beta, true_projection(spec, geometry), rtol=1e-12,
                               atol=1e-12)
    assert factor.shape == (64, 32, 255)
    legacy, legacy_factor = exact_family_projection(
        spec, geometry, coordinates, speed_scale
    )
    explicit, explicit_factor = exact_family_projection(
        spec, geometry, coordinates, speed_scale, taper_scale=5.0
    )
    np.testing.assert_array_equal(legacy, explicit)
    np.testing.assert_array_equal(legacy_factor, explicit_factor)
