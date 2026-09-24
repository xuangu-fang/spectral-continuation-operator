import torch

from spectral_continuation.fourier_response_operator import FourierResponseOperator


def test_fourier_response_operator_preserves_grid_shape():
    model = FourierResponseOperator(input_channels=8, width=16, modes=6, layers=2)
    inputs = torch.randn(3, 8, 32, 32)
    output = model(inputs)
    assert output.shape == (3, 2, 32, 32)
