from spectral_continuation.audit_wavelength_scaled_phase_space import (
    wavelength_window_pixels,
)


def test_wavelength_window_is_odd_and_shrinks_with_frequency() -> None:
    low = wavelength_window_pixels(
        10.0,
        1.0 / 127.0,
        reference_speed=1.0,
        wavelengths=1.5,
        minimum=7,
        maximum=17,
    )
    high = wavelength_window_pixels(
        24.0,
        1.0 / 127.0,
        reference_speed=1.0,
        wavelengths=1.5,
        minimum=7,
        maximum=17,
    )
    assert low == 17
    assert high == 9
    assert low % 2 == 1 and high % 2 == 1
