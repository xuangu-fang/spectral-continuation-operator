import numpy as np

from spectral_continuation.audit_learned_speed_basis_storage import (
    factorize_depth,
    reconstruct,
)


def test_depth_factorization_recovers_a_shared_low_rank_prior():
    rng = np.random.default_rng(32)
    depth_basis = rng.normal(size=(12, 3))
    mean = rng.normal(size=(4, 3)) @ depth_basis.T
    factor = np.einsum(
        "rjk,zj->rzk", rng.normal(size=(4, 3, 5)), depth_basis,
    )
    mean_core, factor_core, learned_basis, singular = factorize_depth(mean, factor, 3)
    recovered_mean, recovered_factor = reconstruct(
        mean_core, factor_core, learned_basis
    )
    np.testing.assert_allclose(recovered_mean, mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(recovered_factor, factor, rtol=1e-12, atol=1e-12)
    assert singular[3] < 1e-12 * singular[0]
