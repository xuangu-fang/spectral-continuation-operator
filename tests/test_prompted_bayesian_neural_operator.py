import numpy as np
import torch

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
