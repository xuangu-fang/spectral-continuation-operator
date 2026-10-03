import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

from spectral_continuation.coherent_helmholtz_data import environment
from spectral_continuation.separable_background import SeparableInverse, one_dimensional_background


def test_separable_inverse_matches_sparse_kronecker_and_has_gradient():
    matrix, mass = one_dimensional_background(environment(8), 33, 4.0)
    assert mass.shape == (31 * 31,)
    rng = np.random.default_rng(9)
    rhs = rng.normal(size=(31 * 31, 3)) + 1j * rng.normal(size=(31 * 31, 3))
    inverse = SeparableInverse(matrix)
    expected = splu(
        sparse.kronsum(sparse.csc_matrix(matrix), sparse.csc_matrix(matrix), format="csc")
    ).solve(rhs)
    np.testing.assert_allclose(inverse.solve(rhs), expected, atol=1e-11, rtol=1e-9)
    tensor = torch.tensor(rhs, requires_grad=True)
    inverse.torch_solve(tensor).abs().square().sum().backward()
    assert torch.isfinite(tensor.grad).all()
