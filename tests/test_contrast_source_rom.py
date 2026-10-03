import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from spectral_continuation.audit_contrast_source_rom import (
    affine_projection,
    background_spec,
    source_trial,
)
from spectral_continuation.audit_phase_reduced_operator import discrete_operator
from spectral_continuation.coherent_helmholtz_data import environment, solve


def test_contrast_source_identity_and_affine_projection():
    spec = environment(20261003)
    f = 4.0
    fields, _, _, _ = solve(spec, 33, [f])
    truth = fields[0, 1:-1, 1:-1].ravel()
    lap, mass, rhs = discrete_operator(spec, 33)
    _, bg_mass, _ = discrete_operator(background_spec(spec), 33)
    lu = splu(lap + sparse.diags((2 * np.pi * f) ** 2 * bg_mass, format="csc"))
    u0 = lu.solve(rhs)
    basis = source_trial(lu, mass - bg_mass, truth[:, None], f)
    np.testing.assert_allclose(u0 + basis[:, 0], truth, rtol=1e-10, atol=1e-12)
    pred, rank, residual = affine_projection(lap, mass, rhs, u0, basis, f)
    assert rank == 1 and residual < 1e-10
    np.testing.assert_allclose(pred, truth, rtol=1e-9, atol=1e-12)


def test_background_copy_preserves_original_and_boundary():
    spec = environment(20261003)
    bg = background_spec(spec)
    assert sum(spec["contrasts"]) > 0 and sum(bg["contrasts"]) == 0
    assert bg["sponge_strength"] == spec["sponge_strength"]
