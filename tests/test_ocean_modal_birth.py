import numpy as np

from spectral_continuation.audit_ocean_modal_birth import scalar_modes, scalar_query


def test_scalar_shapes_and_linear_birth():
    f = np.array([90.0, 100.0, 110.0, 120.0])
    speed, kz2 = 1500.0, np.array([0.01, 0.03])
    depths = np.linspace(1, 78, 48)
    states = []
    for frequency in f:
        present = 1 if frequency < 105 else 2
        strength = np.array([1j, 0.02j * max(frequency - 105, 0)])[:present]
        kr = np.sqrt((2 * np.pi * frequency / speed) ** 2 - kz2[:present])
        a = np.sin(depths[:, None] * np.sqrt(kz2[None, :present])) * strength
        states.append((kr, a))
    scalars, fit = scalar_modes(f, states, depths, speed)
    assert max(r["shape_error"] for r in fit) < 1e-12
    query_f, x = np.array([102.0, 107.0]), np.linspace(50, 750, 100)
    prediction, info = scalar_query(f, scalars, query_f, x, depths, speed, "cutoff_linear")
    assert abs(info[1]["estimated_cutoff"] - 105) < 1e-10
    expected = []
    for frequency in query_f:
        kr = np.sqrt((2 * np.pi * frequency / speed) ** 2 - kz2)
        a = np.sin(depths[:, None] * np.sqrt(kz2[None])) * np.array(
            [1j, 0.02j * max(frequency - 105, 0)]
        )
        expected.append((a @ np.exp(1j * kr[:, None] * x[None])) / np.sqrt(x)[None])
    np.testing.assert_allclose(prediction, expected, atol=1e-12)
