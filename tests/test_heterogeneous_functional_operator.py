import torch

from spectral_continuation.train_functional_evidence_operator import (
    FunctionalEvidenceOperator,
)
from spectral_continuation.train_functional_prompt_reranker import (
    FunctionalPromptReranker,
)
from spectral_continuation.train_wave_neural_field_autodecoder import ComplexWaveField


def _complex(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.complex(torch.randn(shape), torch.randn(shape))


def test_functional_evidence_operator_has_exact_fallback_and_set_invariance() -> None:
    torch.manual_seed(7)
    model = FunctionalEvidenceOperator(5, 4, 16, 0.5)
    values = _complex((3, 2, 5))
    coordinates = torch.randn(2, 5, 5)
    target_coordinates = torch.randn(7, 5)
    baseline = _complex((3, 7))
    prediction, uncertainty = model(
        values, coordinates, target_coordinates, baseline
    )
    assert torch.equal(prediction, baseline)
    assert torch.isfinite(uncertainty).all()

    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted, _ = model(
        values[:, :, permutation],
        coordinates[:, permutation],
        target_coordinates,
        baseline,
    )
    torch.testing.assert_close(permuted, prediction)


def test_functional_reranker_starts_at_posterior_memory() -> None:
    torch.manual_seed(11)
    model = FunctionalPromptReranker(5, 8, 16, 4, 1, 1.0)
    prompt = _complex((3, 2, 4))
    bank = _complex((6, 2, 4))
    coordinates = torch.randn(2, 4, 5)
    order = torch.tensor([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
    logits = torch.randn(3, 3)
    targets = _complex((6, 7))
    baseline = _complex((3, 7))
    prediction, entropy, mix = model(
        prompt, bank, coordinates, order, logits, targets, baseline
    )
    assert torch.equal(prediction, baseline)
    assert torch.equal(mix, torch.zeros_like(mix))
    assert torch.isfinite(entropy).all()


def test_complex_wave_field_is_coordinate_queryable_and_differentiable() -> None:
    torch.manual_seed(13)
    model = ComplexWaveField(5, 6, 16, 2)
    latent = torch.randn(4, 6, requires_grad=True)
    coordinates = torch.randn(9, 5)
    prediction = model(latent, coordinates)
    assert prediction.shape == (4, 9)
    assert prediction.is_complex()
    prediction.abs().square().mean().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
