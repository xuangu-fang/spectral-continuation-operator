import numpy as np
import torch

from spectral_continuation.train_wave_operator_contrastive import (
    ViewEncoder,
    _multiscale_phase_gradient_channels,
    _phase_gradient_channels,
    _phase_space_channels,
    anti_collapse_loss,
    retrieval_metrics,
)


def test_view_encoder_and_phase_space_shapes():
    model = ViewEncoder(7, dimension=16, width=8)
    result = model(torch.randn(3, 4, 7, 32, 32), torch.randn(3, 4, 3))
    assert result.shape == (3, 16)
    np.testing.assert_allclose(np.linalg.norm(result.detach().numpy(), axis=1), 1.0, rtol=1e-5)
    assert _phase_space_channels(np.ones((16, 16), dtype=np.complex64)).shape == (7, 16, 16)
    value = np.ones((16, 16), dtype=np.complex64)
    assert _phase_gradient_channels(value, include_amplitude=False).shape == (4, 16, 16)
    assert _phase_gradient_channels(value, include_amplitude=True).shape == (5, 16, 16)
    assert _multiscale_phase_gradient_channels(value).shape == (13, 16, 16)


def test_retrieval_metrics_perfect_alignment():
    embedding = np.eye(4)
    metrics = retrieval_metrics(embedding, embedding)
    assert metrics["top1"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 1.0


def test_anti_collapse_loss_penalizes_constant_embeddings():
    collapsed = torch.ones(8, 16)
    collapsed = torch.nn.functional.normalize(collapsed, dim=1)
    diverse = torch.eye(8, 16)
    diverse = torch.nn.functional.normalize(diverse, dim=1)
    assert anti_collapse_loss(collapsed, collapsed) > anti_collapse_loss(diverse, diverse)
