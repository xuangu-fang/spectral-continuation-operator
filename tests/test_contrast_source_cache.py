import numpy as np

from spectral_continuation.coherent_helmholtz_data import environment
from spectral_continuation.contrast_source_cache import (
    base_transform,
    projected_coefficients,
    quadratic_error,
    quadratic_forms,
    source_windows,
)


def test_quadratic_projection_and_loss_match_grid_calculation():
    rng = np.random.default_rng(47)
    av = rng.normal(size=(100, 12)) + 1j * rng.normal(size=(100, 12))
    response = rng.normal(size=(60, 12)) + 1j * rng.normal(size=(60, 12))
    target = rng.normal(size=60) + 1j * rng.normal(size=60)
    rhs = rng.normal(size=100)
    transform = base_transform(3, 4)
    forms = quadratic_forms(av, rhs, response, target, np.vdot(target, target).real)
    coefficients = projected_coefficients(forms, transform, ridge=0)
    direct = transform @ np.linalg.lstsq(av @ transform, rhs, rcond=None)[0]
    np.testing.assert_allclose(coefficients, direct, atol=1e-12)
    error = np.linalg.norm(response @ direct - target) / np.linalg.norm(target)
    np.testing.assert_allclose(quadratic_error(forms, coefficients), error, atol=1e-12)
    forms["t"] *= 5
    np.testing.assert_allclose(
        projected_coefficients(forms, transform, ridge=0), direct, atol=1e-12
    )


def test_windows_are_continuous_coordinate_functions():
    spec = environment(20261003)
    coarse, features = source_windows(spec, np.linspace(0, 1, 33), 10)
    fine, _ = source_windows(spec, np.linspace(0, 1, 65), 10)
    np.testing.assert_allclose(coarse.reshape(31, 31, 10), fine.reshape(63, 63, 10)[1::2, 1::2])
    assert features.shape == (9, 9) and np.isfinite(features).all()
