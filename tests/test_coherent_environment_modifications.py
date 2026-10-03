import numpy as np

from spectral_continuation.coherent_helmholtz_data import (
    apply_environment_modifications,
    environment,
    sample_environment,
)


def test_modifications_preserve_geometry_and_enforce_declared_speed():
    original = environment(20261115)
    assert apply_environment_modifications(original, {}) is original
    modified = apply_environment_modifications(
        original, {"contrast_scale": 3, "source_width": 0.0125, "minimum_speed": 0.45}
    )
    assert original["source_width"] == 0.025
    assert modified["centers"] == original["centers"]
    assert modified["source"] == original["source"]
    speed, _, _ = sample_environment(modified, np.linspace(0, 1, 513))
    assert speed.min() >= 0.45 - 1e-12
    assert modified["generation_modifications"]["effective_contrast_scale"] <= 3
