from __future__ import annotations

import torch

from spectral_continuation.train_posterior_operator_composer import (
    PosteriorOperatorComposer,
    _kernel_coefficient,
)


def test_zero_update_exactly_recovers_scalar_prior_composition() -> None:
    torch.manual_seed(7)
    batch, rank, value_rank = 3, 8, 4
    coefficient = torch.randn(batch, rank, dtype=torch.complex64)
    bank = torch.randn(6, rank, dtype=torch.complex64)
    memory = torch.randn(6, value_rank, dtype=torch.complex64)
    variance = torch.linspace(0.5, 1.5, rank)
    neural = torch.randn(batch, rank, dtype=torch.complex64)
    prior_gate = torch.tensor([0.1, 0.4, 0.8])
    model = PosteriorOperatorComposer(
        rank=rank,
        value_rank=value_rank,
        retrieval_dimensions=4,
        neighbors=3,
        width=16,
        heads=4,
        layers=1,
        expansion=2,
        residual_bound=0.5,
    )
    actual = model(
        coefficient,
        torch.ones(rank),
        bank,
        memory,
        variance,
        torch.zeros(34),
        4,
        3,
        0.3,
        0.75,
        neural,
        prior_gate,
        torch.zeros(batch, 17),
    )
    kernel = _kernel_coefficient(
        coefficient, bank, memory, variance, 4, 3, 0.3, 0.75
    )
    expected = kernel + prior_gate[:, None] * (neural - kernel)
    torch.testing.assert_close(actual, expected)


def test_composer_is_small_and_backpropagates() -> None:
    model = PosteriorOperatorComposer(
        rank=8,
        value_rank=4,
        retrieval_dimensions=4,
        neighbors=2,
        width=16,
        heads=4,
        layers=1,
        expansion=2,
        residual_bound=0.5,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000
    coefficient = torch.randn(2, 8, dtype=torch.complex64)
    result = model(
        coefficient,
        torch.ones(8),
        torch.randn(4, 8, dtype=torch.complex64),
        torch.randn(4, 4, dtype=torch.complex64),
        torch.ones(8),
        torch.zeros(34),
        4,
        2,
        0.3,
        1.0,
        torch.randn(2, 8, dtype=torch.complex64),
        torch.full((2,), 0.5),
        torch.zeros(2, 17),
    )
    result.abs().mean().backward()
    assert model.gate.weight.grad is not None
    assert torch.isfinite(model.gate.weight.grad).all()
