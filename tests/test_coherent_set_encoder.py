import torch

from spectral_continuation.coherent_set_encoder import CoherentSetEncoder


def test_permutation_phase_and_padding_invariance():
    torch.manual_seed(12)
    model = CoherentSetEncoder()
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.02)
    x, y = torch.rand(3, 32, 2), torch.randn(3, 32, dtype=torch.complex64)
    weights = torch.rand(3, 32)
    expected = model(x, y, weights)
    order = torch.randperm(32)
    phase = torch.exp(torch.tensor(0.6j))
    actual = model(x[:, order], y[:, order] * phase, weights[:, order])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    padded = model(
        torch.cat((x, torch.zeros(3, 4, 2)), 1),
        torch.cat((y, torch.zeros(3, 4, dtype=torch.complex64)), 1),
        torch.cat((weights, torch.zeros(3, 4)), 1),
    )
    torch.testing.assert_close(padded, expected, rtol=1e-5, atol=1e-6)
