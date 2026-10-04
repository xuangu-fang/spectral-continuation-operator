import copy

import numpy as np

from spectral_continuation.audit_waveguide_generator_identification import (
    generator_geometry,
    true_projection,
)
from spectral_continuation.audit_waveguide_spectral_taper_shift import shifted_environment
from spectral_continuation.learned_speed_manifold import speed_delta, speed_projection


def test_speed_projection_matches_true_medium_and_has_no_hidden_coefficient_read():
    spec = shifted_environment(20267000, 16, 15.0, 10.0)
    geometry = generator_geometry(spec, cells=8, depth_terms=8, modes=8,
                                  quadrature_nodes=80)
    mean_speed = speed_delta(spec, geometry)
    factor_speed = np.random.default_rng(19).normal(
        size=(*mean_speed.shape, 2)
    ) * 0.1
    coordinates = np.zeros(2)
    beta, factor = speed_projection(
        spec, geometry, mean_speed, factor_speed, coordinates
    )
    np.testing.assert_allclose(beta, true_projection(spec, geometry),
                               rtol=1e-12, atol=1e-12)
    changed = copy.deepcopy(spec)
    changed.pop("spectral_coefficients_m_s")
    changed_beta, changed_factor = speed_projection(
        changed, geometry, mean_speed, factor_speed, coordinates
    )
    np.testing.assert_array_equal(beta, changed_beta)
    np.testing.assert_array_equal(factor, changed_factor)
    direction = np.array([0.3, -0.7])
    h = 1e-4
    forward, _ = speed_projection(spec, geometry, mean_speed, factor_speed,
                                  coordinates + h * direction, derivative=False)
    backward, _ = speed_projection(spec, geometry, mean_speed, factor_speed,
                                   coordinates - h * direction, derivative=False)
    finite_difference = (forward - backward) / (2 * h)
    analytic = np.einsum("rdk,k->rd", factor, direction)
    np.testing.assert_allclose(finite_difference, analytic, rtol=1e-6, atol=1e-10)
