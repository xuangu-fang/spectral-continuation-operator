import json

import numpy as np

from spectral_continuation.audit_waveguide_local_stability import (
    LOWER,
    UPPER,
    pack,
    paired_step,
    unpack,
)
from spectral_continuation.coherent_waveguide_data import environment


def test_medium_tangent_uses_actual_parameter_family_and_preserves_background():
    original = environment(20261332)
    parameters = pack(original)
    restored = unpack(parameters, original)
    for key in ["contrasts_m_s", "centers_m", "widths_m"]:
        np.testing.assert_allclose(restored[key], original[key])
    changed = parameters.copy()
    changed[0] += 0.01
    perturbed = unpack(changed, original)
    assert perturbed["source_depth_m"] == original["source_depth_m"]
    assert perturbed["thermocline_depth_m"] == original["thermocline_depth_m"]
    assert perturbed["contrasts_m_s"] != original["contrasts_m_s"]


def test_stability_pair_stays_inside_physical_family_bounds():
    parameters = pack(environment(20261332))
    direction = np.ones(15) / np.sqrt(15)
    step, requested = paired_step(parameters, direction, 10, np.eye(15), 0.02)
    json.dumps({"step": step, "requested": requested, "limited": step < requested})
    for sign in [-1, 1]:
        endpoint = parameters + sign * 0.5 * step * direction
        assert np.all(endpoint >= LOWER) and np.all(endpoint <= UPPER)
        assert np.max(np.abs(endpoint - parameters)) <= 0.020000000001
