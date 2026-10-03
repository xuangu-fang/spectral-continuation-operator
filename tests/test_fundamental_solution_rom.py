import numpy as np
from scipy.special import hankel1

from spectral_continuation.audit_fundamental_solution_rom import gaussian_reference


def test_radial_gaussian_reference_far_field_and_pde():
    f, width, loss = 8.0, 0.025, 0.002
    r = np.linspace(0.001, 0.7, 10001)
    k = 2 * np.pi * f * (1 + 1j * loss)
    u = gaussian_reference(r, f, width, loss=loss)
    far = 0.5j * np.pi * width**2 * np.exp(-0.5 * k**2 * width**2) * hankel1(0, k * r)
    np.testing.assert_allclose(u[r > 0.3], far[r > 0.3], rtol=1e-8, atol=1e-12)
    derivative = np.gradient(u, r, edge_order=2)
    residual = (
        np.gradient(derivative, r, edge_order=2)
        + derivative / r
        + k * k * u
        + np.exp(-0.5 * (r / width) ** 2)
    )
    interior = (r > 0.01) & (r < 0.6)
    assert (
        np.linalg.norm(residual[interior])
        / np.linalg.norm(np.exp(-0.5 * (r[interior] / width) ** 2))
        < 0.0002
    )
