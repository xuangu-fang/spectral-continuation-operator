import numpy as np
import torch

from spectral_continuation.audit_waveguide_generator_identification import (
    generator_geometry,
    normal_statistics,
)
from spectral_continuation.coherent_waveguide_data import environment
from spectral_continuation.learned_waveguide_generator import GeneratorEncoder


def test_generator_encoder_starts_at_real_classical_estimate():
    model = GeneratorEncoder(12, width=8, modes=4, layers=2, terms=4)
    x = torch.randn(2, 12, 16)
    baseline = torch.randn(2, 16, 4)
    torch.testing.assert_close(model(x, baseline), baseline)
    model(x, baseline).square().mean().backward()
    assert model.readout[-1].weight.grad.abs().max() > 0


def test_generator_statistics_are_common_phase_and_scale_invariant():
    spec = environment(42)
    geometry = generator_geometry(spec, 8, 4, modes=4)
    rng = np.random.default_rng(7)
    a = rng.normal(size=(3, 65, 4)) + 1j * rng.normal(size=(3, 65, 4))
    r = np.linspace(0, 2000, 65)
    first = normal_statistics(spec, a, [60, 100, 140], r, geometry)
    second = normal_statistics(spec, a * (1.3 + 0.7j), [60, 100, 140], r, geometry)
    for x, y in zip(first, second, strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-10)
