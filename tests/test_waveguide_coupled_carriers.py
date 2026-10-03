import numpy as np

from spectral_continuation.audit_waveguide_coupled_carriers import (
    carrier_basis,
    decode_coefficients,
    fit_basis,
    fit_coefficients,
    local_carrier_basis,
    splines,
)


def test_envelope_splines_partition_unity_and_count():
    ranges = np.linspace(0, 2000, 257)
    for count in [4, 8, 16, 64]:
        b = splines(ranges, count)
        assert b.shape == (257, count)
        np.testing.assert_allclose(b.sum(-1), 1, atol=1e-12)


def test_coupled_carriers_recover_interfering_mode_curve():
    rng = np.random.default_rng(3)
    ranges = np.linspace(0, 2000, 513)
    kr = np.array([0.1, 0.3, 0.4], dtype=complex)
    basis = carrier_basis(ranges, kr, 3, 4)
    coefficient = rng.normal(size=(12, 5)) + 1j * rng.normal(size=(12, 5))
    value = basis @ coefficient
    recovered = fit_basis(basis, value, 1e-10)
    np.testing.assert_allclose(basis @ recovered, value, atol=1e-8)


def test_local_carriers_bounded_even_with_evanescent_modes_and_include_own_phase():
    ranges = np.linspace(0, 2000, 257)
    kr = np.array([0.3j, 0.2j, 0.1, 0.2, 0.3])
    b = local_carrier_basis(ranges, kr, 0, 4, 8)
    assert np.max(abs(b)) <= 1 + 1e-12 and np.isfinite(b).all()
    np.testing.assert_allclose(b[:, :8].sum(-1), 1, atol=1e-12)
    modes = np.exp(1j * ranges[:, None] * kr.real[None]) * np.array([1, 0.5j, 0.2, 0.8, 0.7j])[None]
    coef = fit_coefficients(modes, ranges, kr, 4, 8, 1e-10, "local_unitary")
    reconstructed = decode_coefficients(coef, ranges, kr, 4, 8, "local_unitary")
    np.testing.assert_allclose(reconstructed, modes, atol=1e-6)
