import json

import numpy as np

from spectral_continuation.audit_waveguide_spectral_exact_basis import analytic_prior
from spectral_continuation.coherent_waveguide_data import spectral_environment


def test_exact_family_basis_uses_known_background_not_hidden_coefficients():
    spec = spectral_environment(20264600)
    changed = json.loads(json.dumps(spec))
    changed["spectral_coefficients_m_s"] = np.random.default_rng(1).normal(
        size=(16, 16)
    ).tolist()
    config = {"range_cosine_terms": 16, "depth_cosine_terms": 16, "depth_terms": 32}
    mean, factor, scale = analytic_prior(spec, config)
    other_mean, other_factor, other_scale = analytic_prior(changed, config)
    assert mean.shape == (64, 32)
    assert factor.shape == (64, 32, 255)
    assert np.isfinite(factor).all() and np.linalg.norm(factor) > 0
    np.testing.assert_array_equal(mean, other_mean)
    np.testing.assert_array_equal(factor, other_factor)
    assert scale == other_scale
