import torch

from spectral_continuation.train_wave_operator_identifier import (
    ContextQueryUNet,
    MediumQueryDataset,
)


def test_context_query_unet_shapes():
    model = ContextQueryUNet(120, width=8)
    prediction = model(torch.randn(2, 120, 32, 32), torch.randn(2, 7, 32, 32))
    assert prediction.shape == (2, 2, 32, 32)


def test_medium_query_dataset_shapes():
    import numpy as np

    dataset = MediumQueryDataset(
        np.ones((2, 9, 8, 16, 16), dtype=np.complex64),
        np.ones((2, 16, 16), dtype=np.float32),
        np.asarray([0, 1]),
        np.full((9, 2), 0.5, dtype=np.float32),
        np.linspace(6.0, 24.0, 8),
        np.ones(8, dtype=np.float32),
        "raw",
        all_queries=False,
    )
    context, query, target, *_ = dataset[0]
    assert context.shape == (1, 16, 16)
    assert query.shape == (7, 16, 16)
    assert target.shape == (2, 16, 16)
