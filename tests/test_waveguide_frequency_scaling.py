import numpy as np
import pytest

from spectral_continuation.audit_waveguide_frequency_scaling import (
    backend_audit,
    modal_raw,
    propagate_code,
)
from spectral_continuation.audit_waveguide_generator_identification import generator_geometry
from spectral_continuation.coherent_waveguide_data import environment


def test_frequency_backend_matches_independent_scipy():
    result = backend_audit({"audit_systems": [20263300], "eigh_device": "cpu", "eigh_batch": 4})
    assert max(result["scipy_comparison_raw"]) < 1e-9


def test_high_mode_quadrature_is_orthonormal():
    geometry = generator_geometry(environment(20263300), 4, 32, 128, 384)
    gram = geometry["basis"].T @ (geometry["weights"][:, None] * geometry["basis"])
    np.testing.assert_allclose(gram, np.eye(128), atol=1e-12)


def test_modal_raw_includes_omitted_modes():
    truth = np.ones((3, 4, 2), dtype=complex)
    assert np.allclose(modal_raw(truth[:, :2], truth), np.sqrt(0.5))
    with pytest.raises(ValueError):
        modal_raw(truth, truth[:, :2])


def test_zero_contrast_coefficients_match_background_teacher():
    spec = environment(20263300)
    spec["contrasts_m_s"] = [0.0, 0.0, 0.0]
    ranges = np.linspace(0, 2000, 13)
    truth = propagate_code(spec, 140, ranges, 8, 16, 96, "cpu", 4)
    predicted = propagate_code(spec, 140, ranges, 8, 16, 96, "cpu", 4, np.zeros((8, 8)))
    assert max(modal_raw(predicted, truth)) < 1e-9
