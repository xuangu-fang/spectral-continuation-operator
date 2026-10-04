import numpy as np

from spectral_continuation.audit_helmholtz_generator_compression import (
    spline_readout,
    tensor_readout,
)


def test_continuous_tensor_readout_reproduces_separable_polynomials_and_boundaries():
    axis = np.linspace(0.02, 0.98, 17)
    fine = np.linspace(0, 1, 41)
    values = 1 - 0.1 * axis[:, None] ** 2 * axis[None, :] ** 3
    expected = 1 - 0.1 * fine[:, None] ** 2 * fine[None, :] ** 3
    np.testing.assert_allclose(tensor_readout(values, axis, fine, 1), expected, atol=1e-12)
    np.testing.assert_allclose(spline_readout(values, axis, fine), expected, atol=1e-12)
