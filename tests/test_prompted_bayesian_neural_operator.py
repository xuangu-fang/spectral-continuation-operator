import numpy as np
import torch

from spectral_continuation.cross_domain_response_data import CrossDomainResponseArchive
from spectral_continuation.global_bayesian_response_operator import (
    GlobalConfig,
    GlobalQ3Evaluator,
    fit_global_subspace,
    solve_complex_posterior,
)
from spectral_continuation.posterior_memory_response_operator import (
    MemoryConfig,
    evaluate_memory_config_batched,
    evaluate_memory_configs,
)
from spectral_continuation.train_prompted_bayesian_neural_operator import (
    PromptedBayesianNeuralOperator,
    _q2_events,
)


def test_zero_initialized_functional_residual_preserves_base_basis():
    rng = np.random.default_rng(4)
    coordinates = rng.normal(size=(20, 5)).astype(np.float32)
    basis = (
        rng.normal(size=(20, 4)) + 1j * rng.normal(size=(20, 4))
    ).astype(np.complex64)
    model = PromptedBayesianNeuralOperator(
        coordinates, basis, np.ones(4), 0.1, "spectrum", hidden=16
    )
    index = torch.tensor([1, 5, 9])
    torch.testing.assert_close(model.basis(index), torch.from_numpy(basis)[index])
    coefficient = torch.randn(3, 4, dtype=torch.complex64)
    posterior_variance = torch.ones(4)
    torch.testing.assert_close(
        model.transport_latent(coefficient, posterior_variance), coefficient
    )


def test_q2_events_use_target_frequency_but_exclude_target_source():
    contexts, targets = _q2_events(5, 4, 3, 2, 9)
    for context, target in zip(contexts, targets, strict=True):
        assert np.all(context[:, 0] != target[0])
        assert np.all(context[:, 1] == target[1])


def test_prompt_set_updater_is_permutation_invariant():
    rng = np.random.default_rng(8)
    coordinates = rng.normal(size=(12, 5)).astype(np.float32)
    basis = (
        rng.normal(size=(12, 3)) + 1j * rng.normal(size=(12, 3))
    ).astype(np.complex64)
    model = PromptedBayesianNeuralOperator(
        coordinates,
        basis,
        np.ones(3),
        0.1,
        "spectrum",
        hidden=8,
        receiver_count=2,
        use_prompt_set=True,
    )
    torch.nn.init.normal_(model.latent_residual[-1].weight, std=0.05)
    torch.nn.init.normal_(model.latent_residual[-1].bias, std=0.05)
    coefficient = torch.randn(4, 3, dtype=torch.complex64)
    variance = torch.ones(3)
    context = torch.randn(4, 4, dtype=torch.complex64)
    observed = torch.tensor([0, 1, 4, 5])
    first = model.transport_latent(coefficient, variance, context, observed)
    second = model.transport_latent(
        coefficient,
        variance,
        context.reshape(4, 2, 2)[:, [1, 0]].reshape(4, 4),
        observed.reshape(2, 2)[[1, 0]].reshape(4),
    )
    torch.testing.assert_close(first, second)


def test_zero_receiver_residual_preserves_baseline_across_scales():
    rng = np.random.default_rng(12)
    coordinates = rng.normal(size=(12, 5)).astype(np.float32)
    basis = (
        rng.normal(size=(12, 3)) + 1j * rng.normal(size=(12, 3))
    ).astype(np.complex64)
    residual_scale = np.geomspace(0.01, 10.0, 12).astype(np.float32)
    model = PromptedBayesianNeuralOperator(
        coordinates,
        basis,
        np.ones(3),
        0.1,
        "spectrum",
        hidden=8,
        receiver_count=2,
        use_receiver_residual=True,
        residual_scale=residual_scale,
    )
    context = torch.randn(4, 4, dtype=torch.complex64)
    observed = torch.tensor([0, 1, 4, 5])
    target = torch.tensor([8, 9])
    baseline = torch.randn(4, 2, dtype=torch.complex64)
    base_scale = torch.from_numpy(np.geomspace(2.0, 0.2, 12).astype(np.float32))
    corrected = model.correct_response(
        context, observed, target, baseline, base_scale
    )
    torch.testing.assert_close(corrected, baseline)


def test_stable_posterior_is_invariant_to_event_batching():
    generator = torch.Generator().manual_seed(18)
    base = torch.randn(24, 8, generator=generator, dtype=torch.complex64)
    # Nearly duplicate columns make complex64 batched solves shape-sensitive.
    base[:, 7] = base[:, 6] + 1e-5 * base[:, 7]
    observed_basis = torch.stack((base[:12], base[12:]))
    values = torch.randn(2, 5, 12, generator=generator, dtype=torch.complex64)
    penalty = torch.full((2, 8), 1e-8)
    batched, _ = solve_complex_posterior(observed_basis, values, penalty)
    separate = torch.cat(
        [
            solve_complex_posterior(
                observed_basis[index : index + 1],
                values[index : index + 1],
                penalty[index : index + 1],
            )[0]
            for index in range(2)
        ]
    )
    torch.testing.assert_close(batched, separate, rtol=1e-5, atol=1e-5)


def test_vectorized_posterior_memory_matches_reference_event_loop():
    rng = np.random.default_rng(23)
    responses = (
        rng.normal(size=(11, 3, 3, 4))
        + 1j * rng.normal(size=(11, 3, 3, 4))
    ).astype(np.complex64)
    archive = CrossDomainResponseArchive(
        responses=responses,
        sources=np.linspace(0, 1, 3, dtype=np.float32)[:, None],
        receivers=np.linspace(0, 1, 4, dtype=np.float32)[:, None],
        frequencies=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        source_receiver_index=np.asarray([0, 1, 2]),
        split=np.asarray(["train"] * 6 + ["validation"] * 2 + ["test"] * 3),
        system_ids=np.asarray([f"system_{index}" for index in range(11)]),
        valid_response_mask=np.ones((3, 4), dtype=bool),
        provenance={},
    )
    operator = GlobalConfig(4, 0.1, "spectrum", "raw")
    subspace = fit_global_subspace(archive, 4, "raw", "cpu")
    evaluator = GlobalQ3Evaluator(archive, subspace, "test", 1, 2, 9, "cpu")
    memory = MemoryConfig(3, 3, 0.3, 0.5)
    reference = evaluate_memory_configs(evaluator, operator, (memory,))[memory.name]
    vectorized = evaluate_memory_config_batched(evaluator, operator, memory)
    np.testing.assert_allclose(vectorized, reference, rtol=1e-5, atol=1e-6)
