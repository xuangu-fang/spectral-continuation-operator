import torch

from spectral_continuation.audit_posterior_memory_compression import (
    farthest_point_indices,
    kmeans_operator_bank,
)
from spectral_continuation.global_bayesian_response_operator import GlobalConfig


def test_farthest_point_indices_are_unique_and_deterministic() -> None:
    features = torch.tensor([[0.0], [1.0], [3.0], [8.0]], dtype=torch.complex128)
    first = farthest_point_indices(features, 3)
    second = farthest_point_indices(features, 3)
    assert torch.equal(first, second)
    assert len(torch.unique(first)) == 3


def test_farthest_point_indices_return_full_bank_when_requested() -> None:
    features = torch.randn(5, 2, dtype=torch.complex128)
    assert torch.equal(farthest_point_indices(features, 5), torch.arange(5))


def test_kmeans_operator_bank_preserves_weighted_mean() -> None:
    class Evaluator:
        basis = torch.eye(3, dtype=torch.complex128)
        variance = torch.ones(3, dtype=torch.float64)
        train_encoded = torch.tensor(
            [[0, 0, 0], [1, 1, 1], [10, 10, 10], [11, 11, 11]],
            dtype=torch.complex128,
        )
        train_raw = train_encoded.clone()

    operator = GlobalConfig(3, 0.1, "spectrum", "raw")
    encoded, raw, mass = kmeans_operator_bank(Evaluator(), operator, 2, 2)
    expected = Evaluator.train_raw.mean(dim=0)
    assert torch.allclose((encoded * mass[:, None]).sum(dim=0) / mass.sum(), expected)
    assert torch.allclose((raw * mass[:, None]).sum(dim=0) / mass.sum(), expected)
