import numpy as np

from spectral_continuation.wave_operator_data import _build_medium, _ricker


def test_medium_and_wavelet_are_finite():
    speed, damping = _build_medium(32, 0.1, 0.4, 7, 40.0)
    assert speed.shape == damping.shape == (32, 32)
    assert np.isfinite(speed).all() and np.isfinite(damping).all()
    assert speed.min() < 1.0 and speed.max() == 1.0
    wavelet = _ricker(np.linspace(0.0, 0.5, 100), 12.0)
    assert wavelet.shape == (100,)
    assert np.isfinite(wavelet).all()
