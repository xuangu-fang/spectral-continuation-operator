import numpy as np

from spectral_continuation.audit_waveguide_unwrapped_envelopes import relative_unwrapped_phase


def test_spatial_unwrap_uses_source_reference_not_target_phase():
    ranges = np.linspace(0, 2000, 2049)
    slopes = np.array([0.003, 0.006, 0.009])
    source = np.array([[1, -0.5j], [-1, 0.5j], [1j, -0.5]], dtype=complex)
    phase = slopes[:, None, None] * ranges[None, :, None]
    envelope = source[:, None] * np.exp(1j * phase)
    result = relative_unwrapped_phase(envelope, source, 1)
    np.testing.assert_allclose(result, np.broadcast_to(phase, result.shape), atol=1e-12)
    rotated = relative_unwrapped_phase(envelope * np.exp(0.7j), source * np.exp(0.7j), 1)
    np.testing.assert_allclose(rotated, result, atol=1e-12)
