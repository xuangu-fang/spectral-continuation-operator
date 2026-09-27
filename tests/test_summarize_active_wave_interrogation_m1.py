from spectral_continuation.summarize_active_wave_interrogation_m1 import _relative_gap


def test_relative_gap_uses_reference_denominator():
    assert _relative_gap(1.02, 1.0) == 0.020000000000000018
