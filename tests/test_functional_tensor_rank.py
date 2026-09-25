import numpy as np

from spectral_continuation.audit_functional_tensor_rank import (
    factorize_subspace,
    matrix_rank_approximation,
    reciprocal_projection,
    representation_storage,
    shared_tucker_subspace,
    tucker_storage,
)
from spectral_continuation.cross_domain_query_audit import FrequencySubspace


def test_matrix_rank_approximation_preserves_complex_rank_one():
    left = np.asarray([1 + 2j, 3 - 1j, -2 + 0.5j])
    right = np.asarray([2 - 1j, -1 + 4j])
    table = left[:, None] * right[None]
    prediction = matrix_rank_approximation(table, 1)
    assert np.allclose(prediction, table, atol=1e-5)


def test_reciprocal_projection_is_complex_symmetric_without_conjugation():
    table = np.asarray(
        [
            [1 + 2j, 4 + 7j, 9 - 1j],
            [2 - 5j, 3 + 4j, 8 + 6j],
        ],
        dtype=np.complex64,
    )
    projected = reciprocal_projection(table, np.asarray([0, 1]))
    assert projected[0, 1] == projected[1, 0]
    assert projected[0, 1] == 3 + 1j
    assert projected[0, 2] == table[0, 2]


def test_factorized_subspace_enforces_reciprocity():
    mean = np.arange(12, dtype=np.float32).reshape(3, 4).astype(np.complex64)
    basis = np.stack((mean.ravel(), (1j * mean).ravel()), axis=-1)
    result = factorize_subspace(
        FrequencySubspace(mean=mean.ravel(), basis=basis),
        source_count=3,
        receiver_count=4,
        latent_rank=2,
        functional_rank=2,
        receiver_index=np.asarray([0, 1, 2]),
        enforce_reciprocity=True,
    )
    for vector in [result.mean, result.basis[:, 0], result.basis[:, 1]]:
        block = vector.reshape(3, 4)[:, :3]
        assert np.allclose(block, block.T)


def test_storage_ratio_counts_factor_matrices():
    dense, factorized, ratio = representation_storage(5, 70, 4, 2)
    assert dense == 5 * 5 * 70
    assert factorized == 5 * 2 * 75
    assert np.isclose(ratio, factorized / dense)


def test_shared_tucker_is_exact_at_full_axis_ranks():
    generator = np.random.default_rng(4)
    mean = generator.normal(size=(3, 5)) + 1j * generator.normal(size=(3, 5))
    raw_basis = generator.normal(size=(15, 2)) + 1j * generator.normal(size=(15, 2))
    subspace = FrequencySubspace(mean=mean.ravel(), basis=raw_basis)
    result = shared_tucker_subspace(
        subspace,
        source_count=3,
        receiver_count=5,
        latent_rank=2,
        source_rank=3,
        receiver_rank=5,
        receiver_index=np.full(3, -1),
        enforce_reciprocity=False,
    )
    assert np.allclose(result.mean, subspace.mean, atol=1e-5)
    assert np.allclose(result.basis, subspace.basis, atol=1e-5)


def test_tucker_storage_includes_shared_factors_and_cores():
    dense, factorized, ratio = tucker_storage(5, 70, 4, 5, 8)
    assert dense == 5 * 5 * 70
    assert factorized == 5 * 5 + 70 * 8 + 5 * 5 * 8
    assert np.isclose(ratio, factorized / dense)
