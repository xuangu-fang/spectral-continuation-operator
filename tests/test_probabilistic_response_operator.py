import numpy as np

from spectral_continuation.cross_domain_response_data import CrossDomainResponseArchive
from spectral_continuation.probabilistic_response_operator import (
    BatchedQueryEvaluator,
    LatentConfig,
    ProbabilisticSubspace,
    evaluate_config,
    latent_penalty,
    paired_summary,
    predict_query,
)


def test_spectrum_prior_penalizes_low_variance_direction_more():
    subspace = ProbabilisticSubspace(
        mean=np.zeros(4, dtype=np.complex64),
        basis=np.eye(4, 2, dtype=np.complex64),
        variance=np.asarray([10.0, 0.1]),
    )
    penalty = latent_penalty(subspace, LatentConfig(2, 0.2, "spectrum"))
    assert penalty[1] > penalty[0]
    assert np.isclose(np.median(subspace.variance) * 0.2 / subspace.variance[0], penalty[0])


def test_isotropic_config_matches_scalar_ridge_solution():
    basis = np.asarray([[1], [2], [3], [4]], dtype=np.complex64)
    basis /= np.linalg.norm(basis)
    subspace = ProbabilisticSubspace(
        mean=np.zeros(4, dtype=np.complex64),
        basis=basis,
        variance=np.ones(1),
    )
    table = np.asarray([[[2 + 1j, 4 + 2j], [6 + 3j, 8 + 4j]]], dtype=np.complex64)
    prediction = predict_query(
        subspace,
        table,
        np.asarray([0]),
        target=1,
        receiver_count=2,
        config=LatentConfig(1, 0.0, "isotropic"),
    )
    assert np.allclose(prediction, table[:, 1], atol=1e-5)


def test_paired_summary_uses_ratio_of_macro_means():
    baseline = np.asarray([1.0, 3.0])
    candidate = np.asarray([0.5, 1.5])
    result = paired_summary(baseline, candidate, seed=3, draws=1000)
    assert np.isclose(result["ratio_of_means_gain"], 0.5)
    assert result["system_win_rate"] == 1.0


def test_batched_evaluator_matches_reference_query_loop():
    generator = np.random.default_rng(4)
    response = (
        generator.normal(size=(3, 3, 2, 4))
        + 1j * generator.normal(size=(3, 3, 2, 4))
    ).astype(np.complex64)
    archive = CrossDomainResponseArchive(
        responses=response,
        sources=np.zeros((3, 2), dtype=np.float32),
        receivers=np.zeros((4, 2), dtype=np.float32),
        frequencies=np.asarray([1.0, 2.0], dtype=np.float32),
        source_receiver_index=np.asarray([0, 1, 2]),
        split=np.asarray(["train", "test", "test"]),
        system_ids=np.asarray(["train", "test_a", "test_b"]),
        valid_response_mask=np.ones((3, 4), dtype=bool),
        provenance={},
    )
    subspaces = []
    for frequency in range(2):
        basis = generator.normal(size=(12, 3)) + 1j * generator.normal(size=(12, 3))
        basis, _ = np.linalg.qr(basis)
        subspaces.append(
            ProbabilisticSubspace(
                mean=np.zeros(12, dtype=np.complex64),
                basis=basis.astype(np.complex64),
                variance=np.asarray([3.0, 1.0, 0.2]),
            )
        )
    config = LatentConfig(3, 0.1, "spectrum")
    reference = evaluate_config(archive, subspaces, "test", 1, config, 2, 9)
    batched = BatchedQueryEvaluator(
        archive, subspaces, "test", 1, 2, 9, "cpu"
    ).evaluate(config)
    assert np.allclose(batched, reference, rtol=1e-5, atol=1e-6)
