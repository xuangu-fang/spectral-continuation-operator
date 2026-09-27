import numpy as np
import torch

from spectral_continuation.cross_domain_response_data import CrossDomainResponseArchive
from spectral_continuation.evaluate_sequential_active_wave_query import (
    attach_evaluation_archive,
    candidate_target_risks,
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
    risks = candidate_target_risks(
        state,
        np.asarray([0, 0]),
        ordered[:1],
        ordered[1:],
    )
    assert risks.shape == (2,)
    assert np.isfinite(risks).all()
    # Compare the batched implementation against the literal candidate loop.
    receiver = torch.arange(state.receiver_count)
    target_basis = state.basis[receiver].to(torch.complex128)
    target_gram = target_basis.mH @ target_basis
    literal = []
    for candidate in ordered[1:]:
        extended = np.stack((ordered[0], candidate))
        query = torch.as_tensor(
            extended[:, 0] * state.frequency_count + extended[:, 1]
        )
        observed = (query[:, None] * state.receiver_count + receiver[None]).reshape(-1)
        basis_candidate = state.basis[observed].to(torch.complex128)
        precision = basis_candidate.mH @ basis_candidate + torch.diag(
            state.penalty.double()
        ).to(torch.complex128)
        literal.append(
            float(torch.trace(torch.linalg.solve(precision, target_gram)).real)
        )
    np.testing.assert_allclose(risks, literal, rtol=1e-12, atol=1e-12)


def test_attach_evaluation_archive_uses_frozen_normalization():
    response = np.ones((2, 2, 2, 2), dtype=np.complex64) * (3 + 4j)
    coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    archive = CrossDomainResponseArchive(
        responses=response,
        sources=coordinates,
        receivers=coordinates,
        frequencies=np.asarray([1.0, 2.0], dtype=np.float32),
        source_receiver_index=np.asarray([0, 1]),
        split=np.asarray(["test", "test"]),
        system_ids=np.asarray(["new_0", "new_1"]),
        valid_response_mask=np.ones((2, 2), dtype=bool),
        provenance={},
    )
    state = object.__new__(DomainState)
    state.__dict__.update(
        archive=archive,
        basis=torch.ones((8, 1), dtype=torch.complex64),
        center=torch.ones(8, dtype=torch.complex64),
        scale=torch.full((8,), 2.0),
        neural_center=torch.zeros(8, dtype=torch.complex64),
        neural_scale=torch.full((8,), 4.0),
        split_raw={},
        split_encoded={},
        neural_encoded={},
    )
    attach_evaluation_archive(state, archive, "test")
    assert state.split_raw["test"].shape == (2, 8)
    torch.testing.assert_close(
        state.split_encoded["test"],
        (state.split_raw["test"] - 1) / 2,
    )
    torch.testing.assert_close(
        state.neural_encoded["test"], state.split_raw["test"] / 4
    )
