import numpy as np
import torch

from spectral_continuation.audit_waveguide_generator_identification import (
    generator_geometry,
    latent_from_coefficients,
)
from spectral_continuation.coarse_waveguide_operator import query_coefficients
from spectral_continuation.coherent_waveguide_data import environment
from spectral_continuation.train_waveguide_generator_field_loss import DifferentiableGenerator


def test_differentiable_decoder_matches_independent_numpy_and_gradients():
    geometry = generator_geometry(environment(42), 8, 4, modes=8)
    beta = torch.zeros(1, 8, 4, dtype=torch.float64, requires_grad=True)
    background = torch.tensor(geometry["background_q"])[None]
    initial = torch.ones(1, 8, dtype=torch.complex128) * (1 + 0.3j)
    decoder = DifferentiableGenerator(geometry)
    prediction = decoder(beta, background, initial, torch.tensor([110.0]))
    latent, _ = latent_from_coefficients(beta.detach().numpy()[0], initial.numpy()[0], geometry)
    expected = query_coefficients(latent, 110, geometry["edges"])
    np.testing.assert_allclose(prediction.detach().numpy()[0], expected, rtol=1e-10, atol=1e-10)
    prediction.real.square().sum().backward()
    assert torch.isfinite(beta.grad).all() and beta.grad.abs().max() > 0
    h = 1e-7
    plus = beta.detach().clone()
    plus[0, 3, 1] += h
    minus = beta.detach().clone()
    minus[0, 3, 1] -= h
    finite = (
        decoder(plus, background, initial, torch.tensor([110.0])).real.square().sum()
        - decoder(minus, background, initial, torch.tensor([110.0])).real.square().sum()
    ) / (2 * h)
    torch.testing.assert_close(beta.grad[0, 3, 1], finite, rtol=1e-4, atol=1e-5)
