import numpy as np
import torch

from spectral_continuation.audit_multibranch_phase_space import (
    DictionaryConfig,
    controlled_mixtures,
    sparse_plane_wave_fit,
)


def test_multibranch_oracle_improves_two_wave_mixture() -> None:
    patches, _ = controlled_mixtures(
        samples=32,
        patch_size=17,
        branch_count=2,
        snr_db=40.0,
        seed=7,
    )
    config = DictionaryConfig(
        patch_size=17,
        frequency_grid_size=31,
        max_cycles_per_pixel=0.45,
        train_fraction=0.70,
        branch_counts=(1, 2),
        nms_bins=2,
        ridge=1e-4,
    )
    result = sparse_plane_wave_fit(
        torch.from_numpy(patches), config, device=torch.device("cpu"), mask_seed=11
    )
    assert np.mean(result.errors[2]) < 0.7 * np.mean(result.errors[1])


def test_wavevector_selection_is_global_phase_invariant() -> None:
    patches, _ = controlled_mixtures(
        samples=16,
        patch_size=17,
        branch_count=2,
        snr_db=60.0,
        seed=9,
    )
    config = DictionaryConfig(
        patch_size=17,
        frequency_grid_size=31,
        max_cycles_per_pixel=0.45,
        train_fraction=0.70,
        branch_counts=(1, 2),
        nms_bins=2,
        ridge=1e-4,
    )
    first = sparse_plane_wave_fit(
        torch.from_numpy(patches), config, device=torch.device("cpu"), mask_seed=13
    )
    rotation = np.complex64(np.exp(0.7j))
    second = sparse_plane_wave_fit(
        torch.from_numpy(patches * rotation), config, device=torch.device("cpu"), mask_seed=13
    )
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
