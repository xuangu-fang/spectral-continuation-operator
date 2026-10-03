import torch

from spectral_continuation.learned_waveguide_spectral_codec import EnvelopeOperator, SpectralConv1d


def test_spectral_layer_shape_and_gradient():
    layer = SpectralConv1d(8, 12)
    x = torch.randn(2, 8, 33, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()


def test_envelope_model_starts_exactly_at_classical_interpolation():
    model = EnvelopeOperator(226, width=16, modes=8, layers=2)
    features = torch.randn(3, 226, 65)
    baseline = torch.randn(3, 65, 32, dtype=torch.complex64)
    prediction = model(features, baseline)
    torch.testing.assert_close(prediction, baseline)
    prediction.abs().square().mean().backward()
    assert model.readout[-1].weight.grad.abs().max() > 0
