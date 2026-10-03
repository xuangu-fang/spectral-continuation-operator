import numpy as np

from spectral_continuation.audit_background_krylov import right_preconditioned_iterations


def test_right_preconditioning_reaches_dense_reference_without_target():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)) + 10 * np.eye(8)
    rhs = rng.normal(size=8).astype(complex)
    diagonal = a.diagonal()
    output = list(
        right_preconditioned_iterations(
            a, lambda x: x / diagonal, rhs, np.zeros(8, dtype=complex), 8
        )
    )
    count, result, residual = output[-1]
    assert count <= 8 and residual < 1e-10
    np.testing.assert_allclose(result, np.linalg.solve(a, rhs), atol=1e-10)
