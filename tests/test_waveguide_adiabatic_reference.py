import numpy as np

from spectral_continuation.audit_waveguide_adiabatic_reference import adiabatic_reference
from spectral_continuation.audit_waveguide_spectral_transport import modal_projection
from spectral_continuation.coherent_waveguide_data import depth_basis, environment, solve


def test_stationary_medium_adiabatic_reference_matches_teacher():
    spec = environment(13)
    spec["contrasts_m_s"] = [0, 0, 0]
    fields, _, ranges, depths, _ = solve(spec, [60], 65, 65, 24, 160)
    envelope, phase, vectors = adiabatic_reference(spec, 60, ranges, 24)
    modes = modal_projection(envelope * np.exp(1j * phase), vectors, True)
    pred = depth_basis(depths, spec["depth_m"], 24) @ modes.T
    np.testing.assert_allclose(pred, fields[0], atol=1e-11)
