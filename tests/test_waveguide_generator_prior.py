import numpy as np
import pytest

from spectral_continuation.audit_waveguide_generator_prior import (
    conditional_mean,
    fit_prior,
    unpack_statistics,
)
from spectral_continuation.learned_waveguide_generator import context_features


def test_feature_statistics_roundtrip():
    rng = np.random.default_rng(6)
    design = rng.normal(size=(5, 3, 3))
    gram = np.einsum("rki,rkj->rij", design, design)
    rhs, beta = rng.normal(size=(2, 5, 3))
    geometry = {
        "background_q": np.ones(7),
        "weights": np.ones(7) / 7,
        "psi": np.ones((7, 3)),
        "depth": 1,
        "middle": np.linspace(0, 1, 5),
        "edges": np.linspace(0, 1, 6),
    }
    features = context_features(gram, rhs, beta, geometry)
    recovered_g, recovered_h = unpack_statistics(features, 3)
    np.testing.assert_allclose(recovered_g, gram)
    np.testing.assert_allclose(recovered_h, rhs)


def test_gaussian_prior_matches_dense_posterior():
    rng = np.random.default_rng(17)
    targets = rng.normal(size=(20, 3, 2))
    mean, factor = fit_prior(targets, 6)
    matrix = factor.reshape(6, 6)
    np.testing.assert_allclose(matrix @ matrix.T, np.cov(targets.reshape(20, 6).T))
    gram = np.tile(np.eye(2), (3, 1, 1))
    rhs = rng.normal(size=(3, 2))
    ridge = 0.3
    prediction = conditional_mean(gram, rhs, mean, factor, ridge)
    covariance = matrix @ matrix.T
    expected = mean.ravel() + covariance @ np.linalg.solve(
        covariance + ridge * np.eye(6), (rhs - mean).ravel()
    )
    np.testing.assert_allclose(prediction.ravel(), expected)
    with pytest.raises(ValueError):
        conditional_mean(gram, rhs, mean, factor, 0)
