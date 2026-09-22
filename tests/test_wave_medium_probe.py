import numpy as np
import torch

from spectral_continuation.train_wave_medium_probe import MediumUNet, _input_tensor


def test_medium_unet_shape_and_range():
    model = MediumUNet(12, width=8)
    value = model(torch.randn(2, 12, 32, 32))
    assert value.shape == (2, 1, 32, 32)
    assert 0.0 <= float(value.min().detach()) <= float(value.max().detach()) <= 0.7


def test_input_tensor_obeys_source_budget():
    class Bank:
        values = np.zeros((2, 9, 8, 5, 4, 4), dtype=np.float32)

    assert _input_tensor(Bank(), source_count=2).shape == (2, 40, 4, 4)
