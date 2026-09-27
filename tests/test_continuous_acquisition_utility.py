import numpy as np
import torch

from spectral_continuation.evaluate_acquisition_baselines import action_coordinates
from spectral_continuation.train_continuous_acquisition_utility import (
    ContinuousAcquisitionUtility,
    _heldout_actions,
    build_events,
    hybrid_shortlist_schedule,
    utility_loss,
)
from spectral_continuation.train_posterior_operator_composer import DomainState


def test_continuous_utility_is_permutation_invariant_and_trainable():
    torch.manual_seed(3)
    model = ContinuousAcquisitionUtility(3, width=16, frequencies=2)
    context = torch.rand(2, 4, 3)
    context_mask = torch.tensor([[True, True, False, False], [True] * 4])
    target = torch.rand(2, 3)
    candidates = torch.rand(2, 5, 3)
    candidate_mask = torch.tensor([[True] * 4 + [False], [True] * 5])
    score = model(context, context_mask, target, candidates, candidate_mask)
    permutation = torch.tensor([1, 0, 3, 2])
    permuted = model(
        context[:, permutation],
        context_mask[:, permutation],
        target,
        candidates,
        candidate_mask,
    )
    torch.testing.assert_close(score, permuted)
    batch = {
        "candidate_mask": candidate_mask,
        "heldout": torch.zeros_like(candidate_mask),
        "label": torch.zeros_like(score),
    }
    loss = utility_loss(score, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_heldout_actions_never_enter_training_context_or_target():
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
        basis=torch.complex(
            torch.arange(27, dtype=torch.float32).reshape(9, 3) / 17.0,
            torch.arange(27, 0, -1, dtype=torch.float32).reshape(9, 3) / 23.0,
        ),
        scale=torch.ones(9),
        penalty=torch.ones(3),
    )
    events = build_events(state, [7], 3, 4, 0, include_analytic_path=False)
    coordinate = action_coordinates(state)
    heldout = _heldout_actions(state, 4, 0)
    assert events
    for event in events:
        target_index = int(np.argmin(np.square(coordinate - event.target).sum(axis=1)))
        assert not heldout[target_index]
        for value in event.context:
            context_index = int(np.argmin(np.square(coordinate - value).sum(axis=1)))
            assert not heldout[context_index]
        assert event.heldout.any()


def test_coordinate_holdout_removes_full_sources_and_frequencies():
    state = object.__new__(DomainState)
    state.__dict__.update(source_count=6, frequency_count=5)
    heldout = _heldout_actions(state, 5, 0, "source_frequency").reshape(6, 5)
    assert np.any(heldout.all(axis=1))
    assert np.any(heldout.all(axis=0))
    assert np.any(~heldout)


def test_full_shortlist_recovers_exact_greedy_schedule():
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
        basis=torch.complex(
            torch.arange(27, dtype=torch.float32).reshape(9, 3) / 17.0,
            torch.arange(27, 0, -1, dtype=torch.float32).reshape(9, 3) / 23.0,
        ),
        scale=torch.ones(9),
        penalty=torch.ones(3),
    )
    model = ContinuousAcquisitionUtility(2, width=8, frequencies=1)
    target = np.asarray([0, 0])
    ordered = np.asarray([[1, 1], [2, 2], [1, 2], [2, 1]], dtype=np.int64)
    hybrid = hybrid_shortlist_schedule(model, state, target, ordered, 3, shortlist=4)
    from spectral_continuation.evaluate_sequential_active_wave_query import (
        greedy_variance_schedule,
    )

    exact = greedy_variance_schedule(state, target, ordered, 3)
    np.testing.assert_array_equal(hybrid, exact)
