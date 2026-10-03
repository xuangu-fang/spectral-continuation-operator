import numpy as np
import pytest
import torch

from spectral_continuation.audit_waveguide_generator_identification import generator_geometry
from spectral_continuation.coherent_waveguide_data import environment
from spectral_continuation.generate_waveguide_query_transfer import source_variant
from spectral_continuation.train_waveguide_generator_field_loss import sampled_gaussian_initial


def test_new_source_is_public_query_without_mutating_environment():
    original = environment(20261332)
    changed = source_variant(original, "narrow", {"narrow_source_width_m": 2})
    assert changed["source_width_m"] == 2
    assert original["source_width_m"] != 2
    assert {k: v for k, v in original.items() if k != "source_width_m"} == {
        k: v for k, v in changed.items() if k != "source_width_m"
    }
    with pytest.raises(ValueError):
        source_variant(original, "unknown", {})


def test_random_source_projection_equals_known_source_quadrature():
    geometry = generator_geometry(environment(20261332), 4, 3, 8)
    options = {"source_depth_bounds_m": [30, 30], "source_width_bounds_m": [2, 2]}
    initial = sampled_gaussian_initial(
        geometry, 2, options, torch.Generator().manual_seed(8), "cpu"
    )
    expected = (np.exp(-0.5 * ((geometry["z"] - 30) / 2) ** 2) * geometry["weights"]) @ geometry[
        "basis"
    ]
    np.testing.assert_allclose(initial.numpy(), np.tile(expected, (2, 1)), atol=1e-14)
