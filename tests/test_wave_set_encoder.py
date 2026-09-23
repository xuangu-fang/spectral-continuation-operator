import numpy as np
import torch

from spectral_continuation.train_wave_set_encoder import dense_system_contrastive_loss
from spectral_continuation.wave_set_encoder import (
    FrozenMediumDecoder,
    QueryViewDecoder,
    ScalarFieldDecoder,
    WaveSetEncoder,
    apply_independent_phase_rotation,
    sample_rectangular_views,
)


def test_set_encoder_is_permutation_invariant_and_decodes_spatial_map():
    torch.manual_seed(3)
    encoder = WaveSetEncoder(5, width=8, dimension=16).eval()
    decoder = FrozenMediumDecoder(width=8).eval()
    views = torch.randn(2, 4, 5, 32, 32)
    metadata = torch.randn(2, 4, 3)
    order = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        first = encoder(views, metadata)
        second = encoder(views[:, order], metadata[:, order])
        prediction = decoder(encoder.spatial(views, metadata))
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
    assert prediction.shape == (2, 1, 32, 32)
    assert 0.0 <= float(prediction.min()) <= float(prediction.max()) <= 0.7


def test_unrestricted_scalar_decoder_matches_spatial_shape():
    torch.manual_seed(31)
    encoder = WaveSetEncoder(5, width=8, dimension=16, aggregation="slots", slots=2)
    decoder = ScalarFieldDecoder(width=8)
    views = torch.randn(2, 3, 5, 32, 32)
    metadata = torch.randn(2, 3, 3)
    prediction = decoder(encoder.spatial(views, metadata))
    assert prediction.shape == (2, 1, 32, 32)


def test_set_encoder_mask_matches_removing_a_view():
    torch.manual_seed(4)
    encoder = WaveSetEncoder(2, width=8, dimension=16).eval()
    views = torch.randn(2, 3, 2, 32, 32)
    metadata = torch.randn(2, 3, 3)
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)
    with torch.no_grad():
        masked = encoder(views, metadata, mask)
        removed = encoder(views[:, :2], metadata[:, :2])
    torch.testing.assert_close(masked, removed, atol=1e-6, rtol=1e-6)


def test_slot_set_encoder_is_permutation_invariant():
    torch.manual_seed(6)
    encoder = WaveSetEncoder(5, width=8, dimension=16, aggregation="slots", slots=3).eval()
    views = torch.randn(2, 4, 5, 16, 16)
    metadata = torch.randn(2, 4, 3)
    order = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        first = encoder(views, metadata)
        second = encoder(views[:, order], metadata[:, order])
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_slot_stack_preserves_slots_and_is_permutation_invariant():
    torch.manual_seed(8)
    encoder = WaveSetEncoder(
        5, width=8, dimension=16, aggregation="slot_stack", slots=3
    ).eval()
    views = torch.randn(2, 4, 5, 16, 16)
    metadata = torch.randn(2, 4, 3)
    order = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        first = encoder.spatial(views, metadata)
        second = encoder.spatial(views[:, order], metadata[:, order])
    assert [value.shape[1] for value in first] == [40, 80, 160]
    for first_scale, second_scale in zip(first, second, strict=True):
        torch.testing.assert_close(first_scale, second_scale, atol=2e-6, rtol=1e-6)


def test_raw_phase_augmentation_preserves_magnitude():
    views = torch.randn(3, 4, 2, 8, 8)
    rotated = apply_independent_phase_rotation(views)
    torch.testing.assert_close(views.square().sum(2), rotated.square().sum(2))


def test_rectangular_view_sampling_has_requested_budget():
    views = sample_rectangular_views(
        np.random.default_rng(2),
        sources=(0, 1, 2, 3),
        frequencies=(0, 1, 2),
        source_count=2,
        frequency_count=3,
    )
    assert len(views) == 6
    assert len({source for source, _ in views}) == 2
    assert len({frequency for _, frequency in views}) == 3


def test_dense_system_contrastive_prefers_aligned_systems():
    torch.manual_seed(5)
    first = torch.randn(4, 8, 4, 4)
    aligned = dense_system_contrastive_loss(first, first.clone(), 0.1)
    shuffled = dense_system_contrastive_loss(first, first.roll(1, dims=0), 0.1)
    assert aligned < shuffled


def test_query_view_decoder_uses_spatial_pyramid_and_metadata():
    torch.manual_seed(7)
    encoder = WaveSetEncoder(5, width=8, dimension=16, aggregation="slots", slots=3)
    decoder = QueryViewDecoder(5, width=8)
    views = torch.randn(2, 4, 5, 32, 32)
    metadata = torch.randn(2, 4, 3)
    query = torch.randn(2, 3)
    prediction = decoder(encoder.spatial(views, metadata), query)
    assert prediction.shape == (2, 5, 32, 32)


def test_multiscale_metadata_conditioning_is_permutation_invariant():
    torch.manual_seed(9)
    encoder = WaveSetEncoder(
        5,
        width=8,
        dimension=16,
        aggregation="slots",
        slots=3,
        metadata_conditioning="multiscale_film",
    ).eval()
    views = torch.randn(2, 4, 5, 16, 16)
    metadata = torch.randn(2, 4, 3)
    order = torch.tensor([1, 3, 0, 2])
    with torch.no_grad():
        first = encoder(views, metadata)
        second = encoder(views[:, order], metadata[:, order])
    torch.testing.assert_close(first, second, atol=2e-6, rtol=1e-6)


def test_coordinate_basis_preserves_query_structure_and_permutation_invariance():
    torch.manual_seed(10)
    encoder = WaveSetEncoder(
        5,
        width=8,
        dimension=16,
        aggregation="coordinate_basis",
        basis_dimension=4,
    ).eval()
    views = torch.randn(2, 4, 5, 16, 16)
    metadata = torch.randn(2, 4, 3)
    order = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        first = encoder.spatial(views, metadata)
        second = encoder.spatial(views[:, order], metadata[:, order])
    assert [value.shape[1] for value in first] == [32, 64, 128]
    for first_scale, second_scale in zip(first, second, strict=True):
        torch.testing.assert_close(first_scale, second_scale, atol=2e-6, rtol=1e-6)
