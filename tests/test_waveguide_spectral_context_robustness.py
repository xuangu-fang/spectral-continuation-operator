import numpy as np

from spectral_continuation.audit_waveguide_spectral_context_robustness import make_context


def test_context_subset_and_relative_complex_noise_are_exact():
    ranges = np.arange(65, dtype=float)
    observed = np.ones((3, 65, 2), dtype=complex)
    rng = np.random.default_rng(18)
    unit = rng.normal(size=observed.shape) + 1j * rng.normal(size=observed.shape)
    chosen_ranges, clean = make_context(
        ranges, observed,
        {"context_profiles": 17, "relative_complex_noise": 0}, unit,
    )
    np.testing.assert_array_equal(chosen_ranges, ranges[::4])
    np.testing.assert_array_equal(clean, observed[:, ::4])
    _, noisy = make_context(
        ranges, observed,
        {"context_profiles": 65, "relative_complex_noise": 0.001}, unit,
    )
    relative = np.linalg.norm(noisy - observed, axis=(1, 2)) / np.linalg.norm(
        observed, axis=(1, 2)
    )
    np.testing.assert_allclose(relative, 0.001, rtol=1e-12)
