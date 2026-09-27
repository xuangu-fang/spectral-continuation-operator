import numpy as np
import torch

from spectral_continuation.evaluate_sequential_active_wave_query import (
    greedy_variance_schedule,
)
from spectral_continuation.train_posterior_operator_composer import DomainState


def test_greedy_schedule_keeps_first_probe_and_budget():
    # Two coordinates per source-frequency query and a two-dimensional latent.
    basis = torch.tensor(
        [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]] * 4, dtype=torch.complex64
    )
    state = object.__new__(DomainState)
    state.__dict__.update(
        basis=basis,
        scale=torch.ones(8),
        penalty=torch.ones(2),
        source_count=2,
        frequency_count=2,
        receiver_count=2,
    )
    ordered = np.asarray([[1, 1], [1, 0], [0, 1]], dtype=np.int64)
    schedule = greedy_variance_schedule(state, np.asarray([0, 0]), ordered, 2)
    assert schedule.shape == (2, 2)
    assert np.array_equal(schedule[0], ordered[0])
