import numpy as np
import torch

from spectral_continuation.cross_domain_query_audit import FrequencySubspace
from spectral_continuation.train_episodic_response_basis import EpisodicResponseBasis


def test_episodic_basis_forward_recovers_rank_one_table():
    table = np.asarray(
        [[1 + 1j, 2 - 1j, 3 + 0.5j], [2 + 2j, 4 - 2j, 6 + 1j]],
        dtype=np.complex64,
    )
    mean = np.zeros(table.size, dtype=np.complex64)
    basis = table.reshape(-1, 1)
    basis /= np.linalg.norm(basis)
    model = EpisodicResponseBasis(
        [FrequencySubspace(mean=mean, basis=basis)], source_count=2, receiver_count=3
    )
    context = torch.from_numpy(table[0][None, None])
    prediction = model(
        context,
        torch.tensor([[0]]),
        torch.tensor([1]),
        torch.tensor([0]),
        ridge=1e-8,
    )
    assert torch.allclose(prediction[0], torch.from_numpy(table[1]), atol=1e-4)


def test_retraction_enforces_reciprocity_and_orthogonality():
    generator = np.random.default_rng(7)
    mean = generator.normal(size=12) + 1j * generator.normal(size=12)
    basis = generator.normal(size=(12, 2)) + 1j * generator.normal(size=(12, 2))
    model = EpisodicResponseBasis(
        [FrequencySubspace(mean=mean.astype(np.complex64), basis=basis.astype(np.complex64))],
        source_count=3,
        receiver_count=4,
    )
    model.retract(np.asarray([0, 1, 2]), orthogonalize=True)
    learned_mean, learned_basis = model.complex_parameters()
    mean_block = learned_mean[0].reshape(3, 4)[:, :3]
    assert torch.allclose(mean_block, mean_block.T)
    for column in range(2):
        block = learned_basis[0, :, column].reshape(3, 4)[:, :3]
        assert torch.allclose(block, block.T, atol=1e-5)
    gram = learned_basis[0].conj().T @ learned_basis[0]
    assert torch.allclose(gram, torch.eye(2, dtype=gram.dtype), atol=1e-5)
