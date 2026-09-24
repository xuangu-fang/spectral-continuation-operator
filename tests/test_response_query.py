import numpy as np
import pytest
import torch

from spectral_continuation.response_query import (
    MaskedResponseTransformer,
    QueryEpisode,
    ResponseQueryCNP,
    SpatialResponseQueryOperator,
    cartesian_receiver_array,
    load_response_archive,
    sample_episode,
    validate_episode,
)


def test_receiver_arrays_are_nested_and_interior():
    rc32, _ = cartesian_receiver_array(64, 32)
    rc64, xy64 = cartesian_receiver_array(64, 64)
    assert rc32.shape == (32, 2)
    assert rc64.shape == (64, 2)
    assert {tuple(value) for value in rc32}.issubset({tuple(value) for value in rc64})
    assert np.all((xy64 > 0.0) & (xy64 < 1.0))


def test_source_array_requires_matching_count(tmp_path):
    field = np.zeros((2, 3, 2, 8, 8), dtype=np.complex64)
    sources = np.asarray([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], dtype=np.float32)
    path = tmp_path / "archive.npz"
    np.savez(path, field=field, sources=sources, frequencies=np.ones(2), split=["train", "test"])
    archive = load_response_archive(path, 3, "source_array")
    assert archive.responses.shape == (2, 3, 2, 3)
    with pytest.raises(ValueError):
        load_response_archive(path, 4, "source_array")


@pytest.mark.parametrize("regime", ["q1", "q2", "q3"])
def test_episode_protocol_has_no_query_leakage(regime):
    episode = sample_episode(
        np.random.default_rng(7),
        system=3,
        source_count=9,
        frequency_count=8,
        budget=16,
        regime=regime,
        target=(4, 5),
    )
    validate_episode(episode)
    assert len(episode.context) == 16


def test_invalid_q3_is_rejected():
    episode = QueryEpisode(0, 2, 3, ((2, 1),), "q3")
    with pytest.raises(ValueError):
        validate_episode(episode)


def test_cnp_is_permutation_invariant_and_supports_masks():
    torch.manual_seed(4)
    model = ResponseQueryCNP(receiver_count=64, hidden=32)
    response = torch.randn(2, 5, 2, 8, 8)
    query = torch.randn(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    target = torch.randn(2, 3)
    output = model(response, query, mask, target)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(response[:, permutation], query[:, permutation], mask[:, permutation], target)
    assert output.shape == (2, 2, 8, 8)
    torch.testing.assert_close(output, permuted, atol=1e-5, rtol=1e-5)


def test_spatial_query_operator_is_permutation_invariant():
    torch.manual_seed(5)
    model = SpatialResponseQueryOperator(width=16)
    response = torch.randn(2, 4, 2, 8, 8)
    query = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    target = torch.randn(2, 3)
    first = model(response, query, mask, target)
    permutation = torch.tensor([2, 0, 3, 1])
    second = model(response[:, permutation], query[:, permutation], mask, target)
    assert first.shape == (2, 2, 8, 8)
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_masked_response_transformer_is_set_equivariant():
    torch.manual_seed(6)
    model = MaskedResponseTransformer(receiver_count=64, hidden=32, layers=2, heads=4)
    response = torch.randn(2, 4, 2, 8, 8)
    query = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    target = torch.randn(2, 3)
    first = model(response, query, mask, target)
    permutation = torch.tensor([1, 3, 0, 2])
    second = model(response[:, permutation], query[:, permutation], mask, target)
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)
