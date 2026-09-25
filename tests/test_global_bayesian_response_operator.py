import numpy as np

from spectral_continuation.cross_domain_response_data import CrossDomainResponseArchive
from spectral_continuation.global_bayesian_response_operator import (
    GlobalConfig,
    GlobalQ3Evaluator,
    _q3_events,
    fit_global_subspace,
)


def _archive() -> CrossDomainResponseArchive:
    rng = np.random.default_rng(7)
    systems, sources, frequencies, receivers, rank = 24, 4, 3, 5, 3
    basis = rng.normal(size=(sources * frequencies * receivers, rank)) + 1j * rng.normal(
        size=(sources * frequencies * receivers, rank)
    )
    latent = rng.normal(size=(systems, rank)) + 1j * rng.normal(size=(systems, rank))
    responses = (latent @ basis.T).reshape(systems, sources, frequencies, receivers)
    split = np.asarray(["train"] * 16 + ["validation"] * 4 + ["test"] * 4)
    return CrossDomainResponseArchive(
        responses=responses.astype(np.complex64),
        sources=np.arange(sources, dtype=np.float32)[:, None],
        receivers=np.arange(receivers, dtype=np.float32)[:, None],
        frequencies=np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
        source_receiver_index=np.full(sources, -1, dtype=np.int64),
        split=split,
        system_ids=np.asarray([f"s{index}" for index in range(systems)]),
        valid_response_mask=np.ones((sources, receivers), dtype=bool),
        provenance={},
    )


def test_q3_events_exclude_target_source_and_frequency():
    context, target = _q3_events(5, 4, budget=6, repeats=2, seed=3)
    for prompts, query in zip(context, target, strict=True):
        assert np.all(prompts[:, 0] != query[0])
        assert np.all(prompts[:, 1] != query[1])


def test_global_operator_recovers_shared_low_rank_system():
    archive = _archive()
    subspace = fit_global_subspace(archive, 6, "raw")
    evaluator = GlobalQ3Evaluator(
        archive, subspace, "test", budget=4, repeats=1, seed=11, device="cpu"
    )
    config = GlobalConfig(6, 1e-7, "isotropic", "raw")
    assert evaluator.evaluate(config).mean() < 1e-3
    assert np.isfinite(evaluator.evaluate_system_retrieval()).all()
    adaptive, selection_rate = evaluator.evaluate_adaptive_mean(config)
    assert adaptive.mean() < 1e-3
    assert 0.0 <= selection_rate <= 1.0
