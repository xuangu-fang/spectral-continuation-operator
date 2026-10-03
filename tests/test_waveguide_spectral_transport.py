import numpy as np

from spectral_continuation.audit_waveguide_spectral_transport import (
    background_modal_carrier,
    diagonal_medium_phase,
    encode_field,
)
from spectral_continuation.coherent_waveguide_data import depth_basis, environment


def test_complete_field_projection_recovers_modal_coefficients():
    rng = np.random.default_rng(3)
    depths = np.linspace(0, 100, 129)
    coefficients = rng.normal(size=(100, 32)) + 1j * rng.normal(size=(100, 32))
    field = depth_basis(depths, 100, 32) @ coefficients.T
    np.testing.assert_allclose(encode_field(field, depths, 100, 32), coefficients, atol=1e-12)


def test_background_modal_carrier_is_unitary_even_with_evanescent_modes():
    kr, vectors = background_modal_carrier(environment(9), 60, 32)
    carrier = vectors @ np.diag(np.exp(1j * kr * 2000)) @ vectors.T
    np.testing.assert_allclose(carrier.conj().T @ carrier, np.eye(32), atol=1e-12)


def test_surface_derivative_gauge_is_positive_without_field_observations():
    spec = environment(9)
    kz = (np.arange(32) + 0.5) * np.pi / spec["depth_m"]
    for frequency in [60, 100, 140]:
        _, vectors = background_modal_carrier(spec, frequency, 32, "boundary_derivative")
        assert np.all(kz @ vectors > 0)


def test_diagonal_phase_reduces_to_background_when_range_independent():
    spec = environment(13)
    spec["contrasts_m_s"] = [0, 0, 0]
    ranges = np.linspace(0, 2000, 65)
    kr, vectors = background_modal_carrier(spec, 100, 32, "boundary_derivative")
    phase = diagonal_medium_phase(spec, 100, ranges, vectors)
    np.testing.assert_allclose(phase, ranges[:, None] * kr[None], atol=1e-9)
