import numpy as np
import torch

from spectral_continuation.learned_coherent_kernel import CoherentKernelEncoder
from spectral_continuation.mixture_coherent_kernel import MixtureCoherentKernelEncoder


def test_one_component_matches_existing_quadratic_kernel():
    torch.manual_seed(2)
    x = torch.randn(3, 17, 2, dtype=torch.float64)
    raw = torch.randn(3, 8, dtype=torch.float64) * 0.1
    original = CoherentKernelEncoder(curvature=True)
    mixture = MixtureCoherentKernelEncoder(components=1)
    expanded = torch.cat((raw, torch.zeros(3, 1)), -1)
    expected = original.covariance(x, x, *original.geometry_from_raw(raw))
    result = mixture.covariance(x, x, *mixture.geometry_from_raw(expanded))
    torch.testing.assert_close(result, expected)


def test_mixture_kernel_positive_and_global_phase_equivariant():
    torch.manual_seed(3)
    model = MixtureCoherentKernelEncoder(components=3, ridge=1e-6)
    x = torch.rand(2, 16, 2) * 2 - 1
    y = torch.randn(2, 16, dtype=torch.complex64)
    q = torch.rand(2, 9, 2)
    raw = model.raw_geometry(x, y, torch.ones_like(y.real))
    covariance = model.covariance(x, x, *model.geometry_from_raw(raw))
    torch.testing.assert_close(covariance, covariance.mH)
    assert torch.linalg.eigvalsh(covariance).min() > -1e-10
    phase = np.exp(0.75j)
    torch.testing.assert_close(model(x, phase * y, q), phase * model(x, y, q), atol=1e-5, rtol=1e-5)


def test_two_carriers_reconstruct_opposing_waves_without_phase_alignment():
    torch.manual_seed(7)
    model = MixtureCoherentKernelEncoder(components=2, ridge=1e-9)
    x = torch.rand(1, 24, 2, dtype=torch.float64) * 2 - 1
    q = torch.rand(1, 13, 2, dtype=torch.float64) * 2 - 1
    raw = torch.zeros(1, 2, 9, dtype=torch.float64)
    raw[0, :, 0] = torch.tensor([np.arctanh(0.5), -np.arctanh(0.5)])
    raw[..., 2:4] = -5
    geometry = model.geometry_from_raw(raw.flatten(1))
    y = torch.exp(5j * x[..., 0]) + (0.7 + 0.2j) * torch.exp(-5j * x[..., 0])
    target = torch.exp(5j * q[..., 0]) + (0.7 + 0.2j) * torch.exp(-5j * q[..., 0])
    k = model.covariance(x, x, *geometry) + model.ridge * torch.eye(x.shape[1])
    pred = (model.covariance(q, x, *geometry) @ torch.linalg.solve(k, y[..., None])).squeeze(-1)
    assert torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target) < 1e-3


def test_context_only_fit_runs_in_double_precision():
    from spectral_continuation.audit_mixture_kernel import fit

    torch.manual_seed(7)
    model = MixtureCoherentKernelEncoder(components=2)
    x = torch.rand(2, 12, 2, dtype=torch.float64)
    y = torch.exp(3j * x[..., 0])
    q = torch.rand(2, 5, 2, dtype=torch.float64)
    config = {"ridge": 1e-8, "initial_epsilon": 1.25, "learning_rate": 0.03, "steps": 2}
    pred, raw = fit(model, x, y, q, config)
    assert pred.shape == (2, 5) and raw.shape == (2, 18)
    assert torch.isfinite(pred).all()
