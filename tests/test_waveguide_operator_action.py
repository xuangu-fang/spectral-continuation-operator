import numpy as np

from spectral_continuation.audit_waveguide_generator_identification import (
    generator_geometry,
    latent_from_coefficients,
)
from spectral_continuation.audit_waveguide_operator_action import action_metrics, propagate_actions
from spectral_continuation.coarse_waveguide_operator import query_coefficients
from spectral_continuation.coherent_waveguide_data import environment


def test_multi_source_propagation_matches_each_independent_scalar_query():
    geometry = generator_geometry(environment(20261332), 8, 4, modes=8)
    beta = np.zeros((8, 4))
    sources = np.eye(8, 3).astype(complex)
    sources[:, 1] *= 1j
    latent, _ = latent_from_coefficients(beta, sources, geometry)
    matrices = (2 * np.pi * 70) ** 2 * latent["mass"] - np.diag(latent["kz2"])
    eigen, vectors = np.linalg.eigh(matrices)
    roots = np.sqrt(eigen.astype(complex))
    ranges = np.linspace(0, 2000, 21)
    result = propagate_actions(
        latent["edges"],
        sources,
        ranges,
        lambda cell: (roots[cell], vectors[cell]),
        latent["attenuation"],
    )
    for index in range(3):
        expected = query_coefficients({**latent, "initial": sources[:, index]}, 70, ranges)
        np.testing.assert_allclose(result[:, :, index], expected, rtol=1e-10, atol=1e-10)


def test_worst_relative_error_covers_complex_source_superpositions_without_ridge():
    rng = np.random.default_rng(11)
    truth = rng.normal(size=(5, 8, 3)) + 1j * rng.normal(size=(5, 8, 3))
    prediction = truth.copy()
    prediction[:, :, 1] *= 1.4
    metrics = action_metrics(prediction, truth)
    assert metrics["worst_case_valid"]
    for _ in range(20):
        source = rng.normal(size=3) + 1j * rng.normal(size=3)
        error = np.linalg.norm((prediction - truth) @ source) / np.linalg.norm(truth @ source)
        assert error <= metrics["worst_relative_in_source_subspace"] + 1e-12
    singular = np.ones((2, 3, 2), complex)
    assert action_metrics(singular, singular)["worst_relative_in_source_subspace"] is None
