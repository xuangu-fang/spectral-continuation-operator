import numpy as np
import torch
from scipy.interpolate import RBFInterpolator

from spectral_continuation.learned_coherent_kernel import CoherentKernelEncoder


def test_zero_initialization_matches_gaussian_and_phase_equivariance():
    torch.manual_seed(4)
    x, q = torch.rand(2, 30, 2).double(), torch.rand(2, 20, 2).double()
    y = torch.randn(2, 30, dtype=torch.complex128)
    model = CoherentKernelEncoder()
    p = model(x, y, q)
    for i in range(2):
        expected = RBFInterpolator(
            x[i].numpy(), y[i].numpy(), kernel="gaussian", epsilon=1, degree=-1, smoothing=1e-8
        )(q[i].numpy())
        np.testing.assert_allclose(p[i].detach().numpy(), expected, rtol=1e-5, atol=1e-5)
    phase = torch.exp(torch.tensor(0.7j, dtype=torch.complex128))
    torch.testing.assert_close(model(x, y * phase, q), p * phase, rtol=1e-5, atol=1e-5)


def test_kernel_hermitian_and_gradients():
    torch.manual_seed(5)
    model = CoherentKernelEncoder()
    x, q = torch.rand(2, 30, 2), torch.rand(2, 20, 2)
    y = torch.randn(2, 30, dtype=torch.complex64)
    k, lower = model.infer_geometry(x, y, torch.ones_like(y.real))
    kernel = model.covariance(x, x, k, lower)
    torch.testing.assert_close(kernel, kernel.mH)
    assert torch.linalg.eigvalsh(kernel).min() > -1e-12
    model(x, y, q).abs().square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_curved_phase_preserves_positive_kernel():
    torch.manual_seed(6)
    model = CoherentKernelEncoder(curvature=True, mode="global")
    with torch.no_grad():
        model.encoder["delta"].copy_(torch.randn(8) * 0.2)
    x = torch.rand(2, 30, 2)
    y = torch.randn(2, 30, dtype=torch.complex64)
    k, lower = model.infer_geometry(x, y, torch.ones_like(y.real))
    kernel = model.covariance(x, x, k, lower)
    torch.testing.assert_close(kernel, kernel.mH)
    assert torch.linalg.eigvalsh(kernel).min() > -1e-12


def test_padding_is_not_observation():
    torch.manual_seed(7)
    model = CoherentKernelEncoder()
    x, q = torch.rand(1, 12, 2), torch.rand(1, 10, 2)
    y = torch.randn(1, 12, dtype=torch.complex64)
    expected = model(x, y, q)
    padded_x = torch.cat((x, torch.zeros(1, 8, 2)), 1)
    padded_y = torch.cat((y, torch.zeros(1, 8, dtype=torch.complex64)), 1)
    weights = torch.cat((torch.ones(1, 12), torch.zeros(1, 8)), 1)
    torch.testing.assert_close(
        model(padded_x, padded_y, q, weights), expected, rtol=1e-7, atol=1e-7
    )
