import torch

from spectral_continuation.unrolled_coherent_kernel import UnrolledCoherentKernel


def test_unrolled_second_order_gradient_and_eval():
    torch.manual_seed(23)
    model = UnrolledCoherentKernel(curvature=True, inner_steps=2, ridge=1e-5)
    x, q = torch.rand(2, 12, 2), torch.rand(2, 9, 2)
    y = torch.exp(3j * x[..., 0])
    prediction = model(x, y, q)
    prediction.abs().square().mean().backward()
    assert model.step_logits.grad is not None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        actual = model(x, y, q)
        phase = torch.exp(torch.tensor(0.43j))
        rotated = model(x, y * phase, q)
    torch.testing.assert_close(actual, prediction.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rotated, actual * phase, rtol=1e-4, atol=1e-4)
