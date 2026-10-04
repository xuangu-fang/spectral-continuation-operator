import numpy as np

from spectral_continuation.audit_waveguide_spectral_taper_shift import shifted_environment
from spectral_continuation.coherent_waveguide_data import sound_speed, spectral_environment


def test_shifted_medium_changes_correlation_but_preserves_rms_and_background():
    seed = 20267000
    original = spectral_environment(seed, 16, 15.0)
    shifted = shifted_environment(seed, 16, 15.0, 10.0)
    assert not np.array_equal(
        original["spectral_coefficients_m_s"], shifted["spectral_coefficients_m_s"]
    )
    for key in ("thermocline_amplitude_m_s", "thermocline_depth_m",
                "thermocline_width_m", "source_depth_m", "source_width_m"):
        assert shifted[key] == original[key]
    rr = np.linspace(0, shifted["range_m"], 128)[:, None]
    zz = np.linspace(0, shifted["depth_m"], 128)[None]
    background = dict(shifted)
    background["spectral_coefficients_m_s"] = np.zeros((16, 16)).tolist()
    deviation = sound_speed(shifted, rr, zz) - sound_speed(background, rr, zz)
    assert np.isclose(np.sqrt(np.mean(deviation**2)), 15.0, rtol=1e-12)
