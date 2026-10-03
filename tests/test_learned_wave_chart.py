import numpy as np
import torch

from spectral_continuation.learned_wave_chart import WaveChartEncoder


def test_shared_chart_phase_equivariance_and_sample_permutation():
    torch.manual_seed(17)
    model = WaveChartEncoder(branches=4, width=16)
    torch.nn.init.normal_(model.encoder[-1].weight, std=0.001)
    x = torch.rand(2, 40, 2) * 2 - 1
    q = torch.rand(2, 13, 2) * 2 - 1
    y = torch.exp(1j * (7 * x[..., 0] + 3 * x[..., 1]))
    out = model(x, y, q)
    phase = np.exp(0.8j)
    rotated = model(x, y * phase, q)
    assert torch.allclose(rotated, out * phase, atol=2e-5, rtol=2e-5)
    perm = torch.randperm(40)
    permuted = model(x[:, perm], y[:, perm], q)
    assert torch.allclose(permuted, out, atol=2e-5, rtol=2e-5)
    loss = out.abs().square().mean()
    loss.backward()
    assert torch.isfinite(model.encoder[-1].weight.grad).all()


def test_zero_encoder_exactly_matches_fixed_chart():
    torch.manual_seed(2)
    learned = WaveChartEncoder(branches=4)
    fixed = WaveChartEncoder(branches=4, learned=False)
    x, q = torch.rand(2, 40, 2), torch.rand(2, 10, 2)
    y = torch.complex(torch.rand(2, 40), torch.rand(2, 40))
    assert torch.equal(learned(x, y, q), fixed(x, y, q))
