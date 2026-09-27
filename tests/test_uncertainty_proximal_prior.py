from __future__ import annotations

import torch

from spectral_continuation.train_uncertainty_proximal_prior import (
    UncertaintyConditionedOperatorPrior,
    apply_proximal_correction,
    assimilate_prior,
)


def test_zero_prior_recovers_closed_form_analytic_posterior() -> None:
    torch.manual_seed(17)
    observations, rank, batch = 11, 5, 3
    basis = torch.randn(observations, rank, dtype=torch.complex128)
    values = torch.randn(batch, observations, dtype=torch.complex128)
    penalty = torch.linspace(0.2, 0.8, rank, dtype=torch.float64)
    precision = basis.mH @ basis + torch.diag(penalty).to(torch.complex128)
    data_projector = torch.linalg.solve(precision, basis.mH)
    prior_projector = torch.linalg.solve(
        precision, torch.diag(penalty).to(torch.complex128)
    )
    analytic = values @ data_projector.T
    actual = assimilate_prior(
        torch.zeros(batch, rank, dtype=torch.complex128),
        analytic,
        prior_projector,
    )
    torch.testing.assert_close(actual, analytic)


def test_operator_prior_is_small_zero_initialized_and_phase_equivariant() -> None:
    torch.manual_seed(19)
    rank = 8
    model = UncertaintyConditionedOperatorPrior(rank, 12, 2, 3.0)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000
    posterior = torch.randn(4, rank, dtype=torch.complex64)
    posterior_variance = torch.linspace(0.1, 0.9, rank)
    variance = torch.linspace(0.5, 1.5, rank)
    torch.testing.assert_close(
        model(posterior, posterior_variance, variance),
        torch.zeros_like(posterior),
    )
    torch.nn.init.normal_(model.output.real, std=0.02)
    torch.nn.init.normal_(model.output.imag, std=0.02)
    phase = torch.exp(torch.tensor(0.71j))
    expected = phase * model(posterior, posterior_variance, variance)
    actual = model(phase * posterior, posterior_variance, variance)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_zero_proximal_correction_exactly_preserves_strong_baseline() -> None:
    baseline = torch.randn(3, 5, dtype=torch.complex64)
    basis = torch.randn(7, 5, dtype=torch.complex64)
    observed = baseline @ basis.T
    actual = apply_proximal_correction(
        baseline,
        torch.zeros_like(baseline),
        torch.eye(5, dtype=torch.complex64),
        basis,
        observed,
        True,
    )
    torch.testing.assert_close(actual, baseline)
