import torch

from spectral_continuation.train_query_conditioned_posterior_metric import (
    QueryConditionedPosteriorMetric,
    posterior_kernel,
)


def test_query_metric_initializes_to_prefix_metric():
    variance = torch.linspace(1.0, 2.0, 8)
    model = QueryConditionedPosteriorMetric(8, 3, 3, variance, 16, 4, -10.0, 4.0)
    weight, temperature_scale, projection = model(torch.zeros(5))
    assert torch.allclose(weight, torch.ones(3))
    assert torch.allclose(temperature_scale, torch.ones(()))
    assert torch.allclose(projection[:3], torch.eye(3, dtype=torch.complex64))


def test_posterior_kernel_excludes_training_identity():
    query = torch.tensor([[0 + 0j], [10 + 0j]], dtype=torch.complex64)
    memory = query.clone()
    values = torch.tensor([[1 + 0j], [3 + 0j]], dtype=torch.complex64)
    result = posterior_kernel(
        query,
        memory,
        values,
        torch.ones(1),
        torch.ones(1),
        neighbors=1,
        temperature=0.1,
        temperature_scale=torch.ones(()),
        blend=1.0,
        value_rank=1,
        query_indices=torch.arange(2),
    )
    assert torch.allclose(result, values.flip(0))
