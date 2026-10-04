import numpy as np

from spectral_continuation.audit_waveguide_generator_identification import background_spec
from spectral_continuation.audit_waveguide_spectral_transport import background_modal_carrier
from spectral_continuation.coherent_waveguide_data import sound_speed, spectral_environment


def test_correlated_spectral_medium_is_reproducible_and_nonparametric():
    spec = spectral_environment(20264600)
    same = spectral_environment(20264600)
    assert np.array_equal(spec["spectral_coefficients_m_s"], same["spectral_coefficients_m_s"])
    assert np.asarray(spec["spectral_coefficients_m_s"]).shape == (16, 16)
    ranges = np.linspace(0, spec["range_m"], 17)[:, None]
    depths = np.linspace(0, spec["depth_m"], 13)[None]
    speeds = sound_speed(spec, ranges, depths)
    assert speeds.shape == (17, 13)
    assert np.min(speeds) > 1300
    assert np.max(np.abs(speeds - sound_speed(background_spec(spec), ranges, depths))) > 2


def test_spectral_medium_fast_grid_matches_pointwise_evaluation():
    spec = spectral_environment(20264601)
    ranges = np.array([50.0, 700.0, 1970.0])
    depths = np.array([3.0, 51.0, 99.0])
    fast = sound_speed(spec, ranges[:, None], depths[None])
    for i, radius in enumerate(ranges):
        for j, depth in enumerate(depths):
            np.testing.assert_allclose(fast[i, j], sound_speed(spec, radius, depth))


def test_context_carrier_never_reads_hidden_spectral_medium():
    first = spectral_environment(20264602)
    second = {**first, "spectral_coefficients_m_s": np.zeros((16, 16)).tolist()}
    frequency = 100
    carrier_first = background_modal_carrier(first, frequency, 8, "boundary_derivative")
    carrier_second = background_modal_carrier(second, frequency, 8, "boundary_derivative")
    for a, b in zip(carrier_first, carrier_second, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-14)
