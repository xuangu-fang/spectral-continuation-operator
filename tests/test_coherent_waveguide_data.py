import numpy as np

from spectral_continuation.coherent_waveguide_data import (
    depth_basis,
    environment,
    quadrature,
    solve,
)


def test_depth_basis_orthogonality_and_boundary_conditions():
    spec = environment(2)
    _z, weights, basis = quadrature(spec, 24)
    np.testing.assert_allclose(basis.T @ (weights[:, None] * basis), np.eye(24), atol=1e-12)
    np.testing.assert_allclose(depth_basis([0], spec["depth_m"], 24), 0, atol=1e-15)
    kz = (np.arange(24) + 0.5) * np.pi / spec["depth_m"]
    np.testing.assert_allclose(kz * np.cos(kz * spec["depth_m"]), 0, atol=1e-14)


def test_homogeneous_marching_matches_analytic_modes_and_is_contracting():
    spec = environment(3)
    spec["thermocline_amplitude_m_s"] = 0
    spec["contrasts_m_s"] = [0, 0, 0]
    fields, coefficients, ranges, depths, _ = solve(spec, [60], 9, 33, modes=16)
    kz = (np.arange(16) + 0.5) * np.pi / spec["depth_m"]
    kr = np.sqrt(((2 * np.pi * 60 / spec["reference_speed_m_s"]) ** 2 - kz**2).astype(complex))
    expected = coefficients[0, 0][None] * np.exp(
        (1j * kr[None] - spec["attenuation_per_m"]) * ranges[:, None]
    )
    np.testing.assert_allclose(coefficients[0], expected, atol=1e-10)
    assert np.max(np.diff(np.linalg.norm(coefficients[0], axis=-1))) <= 1e-10
    np.testing.assert_allclose(
        fields[0], depth_basis(depths, spec["depth_m"], 16) @ coefficients[0].T
    )
