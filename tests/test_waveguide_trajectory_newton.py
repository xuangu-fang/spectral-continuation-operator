import numpy as np
import pytest

from spectral_continuation.audit_waveguide_generator_identification import (
    generator_geometry,
    latent_from_coefficients,
)
from spectral_continuation.audit_waveguide_trajectory_newton import (
    newton_step,
    observation_objective,
    trajectory_jacobian,
)
from spectral_continuation.coarse_waveguide_operator import query_coefficients
from spectral_continuation.coherent_waveguide_data import environment


@pytest.mark.parametrize("frequency", [15, 70])
def test_trajectory_derivative_and_value_match_independent_decoder(frequency):
    geometry = generator_geometry(environment(20261332), 8, 4, modes=8)
    rng = np.random.default_rng(13)
    beta = rng.normal(0, 0.001, (8, 4))
    factor = rng.normal(0, 0.005, (8, 4, 3))
    initial = rng.normal(size=8) + 1j * rng.normal(size=8)
    value, jacobian = trajectory_jacobian(beta, factor, initial, frequency, geometry)
    latent, _ = latent_from_coefficients(beta, initial, geometry)
    expected = query_coefficients(latent, frequency, geometry["edges"])
    np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-10)
    direction = rng.normal(size=3)
    delta = np.einsum("rdk,k->rd", factor, direction)
    h = 1e-5
    plus, _ = latent_from_coefficients(beta + h * delta, initial, geometry)
    minus, _ = latent_from_coefficients(beta - h * delta, initial, geometry)
    finite = (
        query_coefficients(plus, frequency, geometry["edges"])
        - query_coefficients(minus, frequency, geometry["edges"])
    ) / (2 * h)
    np.testing.assert_allclose(
        np.einsum("rmk,k->rm", jacobian, direction), finite, rtol=2e-6, atol=2e-7
    )


def test_sparse_observation_objective_matches_only_selected_states():
    geometry = generator_geometry(environment(20261332), 8, 4, modes=8)
    beta = np.zeros((8, 4))
    initial = np.ones(8, complex)
    latent, _ = latent_from_coefficients(beta, initial, geometry)
    indices = np.array([0, 4, 8])
    truth = query_coefficients(latent, 70, geometry["edges"])[indices]
    roi = np.array([False, True, True])
    value = observation_objective(beta, initial, geometry, [70], [truth], roi, indices)
    assert value < 1e-20
    truth[0] += 100  # Excluded source coordinate must not affect fitting.
    assert observation_objective(beta, initial, geometry, [70], [truth], roi, indices) < 1e-20


def test_map_step_minimizes_same_prior_objective_and_respects_frequency_average():
    geometry = generator_geometry(environment(20261332), 8, 4, modes=8)
    rng = np.random.default_rng(91)
    factor = rng.normal(0, 0.0002, (8, 4, 3))
    coordinates = np.array([2.0, -3.0, 1.0])
    beta = np.einsum("rdk,k->rd", factor, coordinates)
    initial = np.ones(8, complex)
    latent, _ = latent_from_coefficients(beta, initial, geometry)
    indices = np.array([0, 4, 8])
    truth = query_coefficients(latent, 70, geometry["edges"])[indices]
    roi = np.array([False, True, True])
    options = {
        "observation_indices": indices,
        "prior_strength": 1.0,
        "ridge_relative": 1e-8,
        "max_coefficient_step": 0.01,
        "line_search": [1, 0.5, 0.25],
    }
    one, diagnostic = newton_step(
        beta, factor, initial, geometry, [70], [truth], roi, options, coordinates
    )
    assert diagnostic["objective_after"] < diagnostic["objective_before"]
    assert diagnostic["prior_coordinate_mean_square_after"] < np.mean(coordinates**2)
    assert diagnostic["context_after"] > diagnostic["context_before"]
    # Duplicating the same frequency should not halve the effective prior.
    twice, _ = newton_step(
        beta, factor, initial, geometry, [70, 70], [truth, truth], roi, options, coordinates
    )
    np.testing.assert_allclose(one, twice, rtol=1e-10, atol=1e-12)
    updated = np.linalg.lstsq(factor.reshape(-1, 3), one.ravel(), rcond=None)[0]
    actual = observation_objective(one, initial, geometry, [70], [truth], roi, indices)
    actual += options["prior_strength"] * np.mean(updated**2)
    np.testing.assert_allclose(actual, diagnostic["objective_after"], rtol=1e-10)
