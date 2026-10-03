import numpy as np
import pytest

from spectral_continuation.audit_waveguide_first_order_inverse import generator_action
from spectral_continuation.audit_waveguide_generator_identification import generator_geometry
from spectral_continuation.coherent_waveguide_data import environment


@pytest.mark.parametrize("frequency", [15, 70])
def test_square_root_action_jacobian_matches_finite_difference(frequency):
    geometry = generator_geometry(environment(20261332), 4, 3, modes=6)
    rng = np.random.default_rng(1)
    beta = rng.normal(0, 0.001, (4, 3))
    a = rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))
    _, jac = generator_action(beta, a, frequency, geometry)
    direction = rng.normal(size=beta.shape)
    step = 1e-6
    plus, _ = generator_action(beta + step * direction, a, frequency, geometry, False)
    minus, _ = generator_action(beta - step * direction, a, frequency, geometry, False)
    np.testing.assert_allclose(
        (plus - minus) / (2 * step), np.einsum("rmd,rd->rm", jac, direction), rtol=1e-7, atol=1e-8
    )
