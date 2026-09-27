import numpy as np
import torch

from spectral_continuation.evaluate_acquisition_baselines import (
    action_coordinates,
    d_optimal_schedule,
    farthest_schedule,
    nearest_target_schedule,
    schedule_cost,
)
from spectral_continuation.train_posterior_operator_composer import DomainState


def _state() -> DomainState:
    state = object.__new__(DomainState)
    archive = type(
        "Archive",
        (),
        {
            "sources": np.asarray([[0.0], [0.5], [1.0]], dtype=np.float32),
            "frequencies": np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
        },
    )()
    state.__dict__.update(
        archive=archive,
        source_count=3,
        frequency_count=3,
        receiver_count=1,
        basis=torch.eye(9, 3, dtype=torch.complex64),
        penalty=torch.ones(3),
    )
    return state


def test_geometric_and_d_optimal_schedules_are_nested_and_keep_first():
    state = _state()
    target = np.asarray([0, 0])
    ordered = np.asarray([[1, 1], [2, 2], [1, 2], [2, 1]], dtype=np.int64)
    assert action_coordinates(state).shape == (9, 2)
    schedules = (
        nearest_target_schedule(state, target, ordered, 3),
        farthest_schedule(state, ordered, 3),
        d_optimal_schedule(state, ordered, 3),
    )
    for schedule in schedules:
        assert schedule.shape == (3, 2)
        np.testing.assert_array_equal(schedule[0], ordered[0])
        assert len(np.unique(schedule, axis=0)) == 3


def test_schedule_cost_counts_physical_switches():
    schedule = np.asarray([[0, 0], [0, 1], [2, 1], [1, 2]])
    cost = schedule_cost(schedule, receivers=8)
    assert cost["source_activations"] == 4
    assert cost["unique_sources"] == 3
    assert cost["source_switches"] == 2
    assert cost["frequency_switches"] == 2
    assert cost["receiver_samples"] == 32
