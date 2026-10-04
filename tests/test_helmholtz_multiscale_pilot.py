import json
from pathlib import Path

import numpy as np

from spectral_continuation.coherent_helmholtz_data import sample_environment
from spectral_continuation.coherent_helmholtz_multiscale_pilot import environment


def test_multiscale_environment_is_deterministic_and_safe_on_nested_grids():
    config = json.loads(Path("configs/coherent_helmholtz_multiscale_pilot_20261004.json").read_text())
    first = environment(20269700, config)
    second = environment(20269700, config)
    assert first == second
    axis65 = np.linspace(0, 1, 65)
    axis129 = np.linspace(0, 1, 129)
    speed65, _, _ = sample_environment(first, axis65)
    speed129, _, _ = sample_environment(first, axis129)
    np.testing.assert_allclose(speed65, speed129[::2, ::2], rtol=0, atol=2e-15)
    assert np.min(speed129) > 0.55


def test_old_gaussian_environment_has_no_texture_change():
    from spectral_continuation.coherent_helmholtz_data import environment as old_environment

    spec = old_environment(123)
    speed, _, _ = sample_environment(spec, np.linspace(0, 1, 33))
    assert speed.shape == (33, 33)
    assert "spectral_texture" not in spec
