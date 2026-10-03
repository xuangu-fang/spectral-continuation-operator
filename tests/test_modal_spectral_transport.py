import numpy as np

from spectral_continuation.audit_modal_spectral_transport import modal_query, spatial_modes


def test_spatial_mode_identification():
    rng = np.random.default_rng(9)
    ranges = np.linspace(50, 750, 384)
    k = np.array([0.51, 0.43, 0.31])
    a = rng.normal(size=(12, 3)) + 1j * rng.normal(size=(12, 3))
    u = (a @ np.exp(1j * k[:, None] * ranges[None])) / np.sqrt(ranges)[None]
    found, coeff, info = spatial_modes(u, ranges, 0.6)
    np.testing.assert_allclose(found, k, atol=1e-10)
    np.testing.assert_allclose(coeff, a, atol=1e-9)
    assert info["fit_complex_rel_l2"] < 1e-10


def test_dispersion_query_without_target_observation():
    f, fq, c = np.array([90.0, 100.0, 110.0, 120.0]), np.array([95.0, 105.0, 115.0]), 1500.0
    kz = np.array([0.05, 0.12])
    a = np.array([[1.0, 0.5j], [0.5, 1.0]])
    modes = [(np.sqrt((2 * np.pi * freq / c) ** 2 - kz**2), a) for freq in f]
    ranges = np.linspace(50, 750, 100)
    prediction = modal_query(f, modes, fq, ranges, c, True)
    kq = np.sqrt((2 * np.pi * fq[:, None] / c) ** 2 - kz[None] ** 2)
    truth = (
        np.einsum("zm,fmr->fzr", a, np.exp(1j * kq[:, :, None] * ranges[None, None]))
        / np.sqrt(ranges)[None, None]
    )
    np.testing.assert_allclose(prediction, truth, atol=1e-12)
