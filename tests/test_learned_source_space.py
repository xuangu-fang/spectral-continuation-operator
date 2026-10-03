import numpy as np
import torch

from spectral_continuation.contrast_source_cache import base_transform, projected_coefficients
from spectral_continuation.learned_source_space import SourceSpaceEncoder, physics_coefficients


def test_source_encoder_initialization_and_complex_gradients():
    torch.manual_seed(9)
    features = torch.randn(2, 9, 9, dtype=torch.float64)
    scales = torch.rand(2, 60, dtype=torch.float64) + 0.1
    for kind in ["global", "conditional"]:
        model = SourceSpaceEncoder(kind).double()
        transform = model(features, scales)
        expected = torch.tensor(base_transform(6, 10))[None].expand_as(transform)
        torch.testing.assert_close(transform, expected)
        (transform.imag.sum() + transform.real.sum()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_compressed_physics_solve_matches_numpy():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(120, 60)) + 1j * rng.normal(size=(120, 60))
    forms = {"H": a.conj().T @ a, "g": a.conj().T @ rng.normal(size=120)}
    transform = base_transform(6, 10)
    expected = projected_coefficients(forms, transform)
    result = physics_coefficients(
        torch.tensor(forms["H"])[None],
        torch.tensor(forms["g"])[None],
        torch.tensor(transform)[None],
    )
    np.testing.assert_allclose(result[0].numpy(), expected, atol=1e-12)
