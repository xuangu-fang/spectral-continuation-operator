import numpy as np

from spectral_continuation.coherent_helmholtz_data import environment
from spectral_continuation.evaluate_continuous_phase_rom import (
    interpolate_envelopes,
    reference_carriers,
)


def test_envelope_interpolation_exact_identity_and_boundary_compatible_polynomial():
    coarse, fine = np.linspace(0, 1, 17), np.linspace(0, 1, 33)

    def value(axis):
        x, y = np.meshgrid(axis[1:-1], axis[1:-1])
        return ((1 + 2j) * x * (1 - x) * y * (1 - y)).reshape(-1, 1)

    np.testing.assert_array_equal(
        interpolate_envelopes(value(coarse), coarse, coarse), value(coarse)
    )
    np.testing.assert_allclose(
        interpolate_envelopes(value(coarse), coarse, fine), value(fine), atol=1e-14
    )


def test_source_regularization_keeps_carrier_laplacian_bounded():
    spec = environment(20261132)
    spec["source"] = [0.5, 0.5]
    spec["contrasts"] = [0.0] * 4
    raw, smooth = [], []
    for grid in [129, 257]:
        axis = np.linspace(0, 1, grid)
        for regularize, result in [(False, raw), (True, smooth)]:
            c, _ = reference_carriers(spec, axis, [8.0], False, regularize)
            c = c[:, 0].reshape(grid - 2, grid - 2)
            j = (grid - 2) // 2
            lap = (c[j - 1, j] + c[j + 1, j] + c[j, j - 1] + c[j, j + 1] - 4 * c[j, j]) * (
                grid - 1
            ) ** 2
            result.append(lap)
    expected = 4j * 2 * np.pi * 8 / spec["source_width"]
    assert abs(smooth[-1] - expected) / abs(expected) < 0.03
    assert abs(raw[-1]) / abs(raw[0]) > 1.8
