"""A deployed spectral code must not read the hidden evaluation medium."""

import numpy as np

from spectral_continuation.audit_waveguide_frequency_scaling import propagate_code
from spectral_continuation.audit_waveguide_generator_identification import generator_geometry
from spectral_continuation.audit_waveguide_spectral_transport import background_modal_carrier
from spectral_continuation.coherent_waveguide_data import spectral_environment


def test_known_background_geometry_and_physical_readout_ignore_hidden_spectrum():
    spec = spectral_environment(20265000, terms=16, perturbation_rms_m_s=15)
    changed = {**spec, "spectral_coefficients_m_s": (np.ones((16, 16)) * 50).tolist()}
    geometry = generator_geometry(spec, cells=16, depth_terms=8, modes=16)
    changed_geometry = generator_geometry(changed, cells=16, depth_terms=8, modes=16)
    for key in ["matrices", "background_q", "mass0", "middle", "edges"]:
        np.testing.assert_array_equal(geometry[key], changed_geometry[key])
    for first_carrier, second_carrier in zip(
        background_modal_carrier(spec, 140, 16, "boundary_derivative"),
        background_modal_carrier(changed, 140, 16, "boundary_derivative"),
        strict=True,
    ):
        np.testing.assert_array_equal(first_carrier, second_carrier)
    beta = np.zeros((16, 8))
    ranges = np.linspace(50, spec["range_m"], 19)
    first = propagate_code(spec, 400, ranges, 16, 16, 80, "cpu", 8, beta)
    second = propagate_code(changed, 400, ranges, 16, 16, 80, "cpu", 8, beta)
    np.testing.assert_array_equal(first, second)
