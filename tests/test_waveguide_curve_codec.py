import numpy as np

from spectral_continuation.audit_waveguide_curve_codec import curve_codec


def test_carrier_codec_recovers_fast_phase_with_constant_envelope():
    ranges = np.linspace(0, 2000, 2049)
    rate = np.array([0.1, 0.2, 0.3])
    phase = ranges[:, None] * rate
    coefficients = np.exp(1j * phase) * np.array([1, 0.5j, 0.2])
    take = np.linspace(0, 2048, 17, dtype=int)
    prediction = curve_codec(coefficients, ranges, take, phase, np.eye(3))
    np.testing.assert_allclose(prediction, coefficients, atol=1e-12)


def test_curve_codec_reproduces_observed_profiles_without_truth_at_queries():
    rng = np.random.default_rng(3)
    ranges = np.linspace(0, 20, 129)
    take = np.arange(0, 129, 16)
    coefficients = rng.normal(size=(129, 3)) + 1j * rng.normal(size=(129, 3))
    phase = ranges[:, None] * np.array([1, 2, 3])
    first = curve_codec(coefficients, ranges, take, phase, np.eye(3))
    changed = coefficients.copy()
    mask = np.ones(len(ranges), dtype=bool)
    mask[take] = False
    changed[mask] = 1e6
    second = curve_codec(changed, ranges, take, phase, np.eye(3))
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[take], coefficients[take], atol=1e-12)
