import numpy as np
import torch

from spectral_continuation.audit_k1_prior_blend import mixture_error
from spectral_continuation.train_unified_prompt_prior_adapter import (
    UnifiedPriorAdapter,
    adapter_features,
)


def _archive(tmp_path):
    path = tmp_path / "router_features.npz"
    np.savez(
        path,
        _prompt_features=np.arange(48, dtype=np.float32).reshape(2, 3, 8),
        _self_error=np.ones((2, 3, 3), dtype=np.float32),
        _branch_event_error=np.ones((2, 3, 3), dtype=np.float32),
        _kernel_neural_cross=np.full((2, 3), 0.25, dtype=np.float32),
    )
    return np.load(path)


def test_unified_adapter_is_small_and_bounded() -> None:
    model = UnifiedPriorAdapter()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1665
    output = model(torch.randn(5, 17))
    assert output.shape == (5,)
    assert torch.all((output >= 0.0) & (output <= 1.0))


def test_adapter_features_mark_loo_availability(tmp_path) -> None:
    archive = _archive(tmp_path)
    without_loo = adapter_features(archive, has_loo=False)
    with_loo = adapter_features(archive, has_loo=True)
    assert without_loo.shape == (2, 3, 17)
    assert np.all(without_loo[..., 8:16] == 0.0)
    assert np.all(without_loo[..., -1] == 0.0)
    assert np.all(with_loo[..., -1] == 1.0)


def test_complex_residual_mixture_matches_endpoints(tmp_path) -> None:
    archive = _archive(tmp_path)
    error = archive["_branch_event_error"]
    np.testing.assert_allclose(mixture_error(archive, 0.0), error[..., 1].mean(axis=0))
    np.testing.assert_allclose(mixture_error(archive, 1.0), error[..., 2].mean(axis=0))
