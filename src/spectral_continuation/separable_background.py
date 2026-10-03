"""Classical separable absorbing background inverse on CPU/GPU.

The sum of two one-dimensional sponge potentials differs from the teacher only
in the way corners combine absorption. This difference MUST remain in the
contrast source; it is not a change to the target PDE or truth.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.linalg import eig


def one_dimensional_background(spec, grid, frequency):
    n = grid - 2
    h = 1 / (grid - 1)
    axis = np.linspace(0, 1, grid)[1:-1]
    d2 = (
        np.diag(-2 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    ) / h**2
    d4 = d2 - h * h / 12 * (d2 @ d2)
    edge = np.minimum(axis, 1 - axis)
    sponge = spec["sponge_strength"] * np.maximum(1 - edge / spec["sponge_width"], 0) ** 3
    base = 1 + 1j * spec["background_loss"]
    mass1 = 0.5 * base**2 + 2j * base * sponge - sponge**2
    matrix = d4.astype(complex) + np.diag((2 * np.pi * frequency) ** 2 * mass1)
    mass2 = (mass1[:, None] + mass1[None]).ravel()
    return matrix, mass2


class SeparableInverse:
    def __init__(self, matrix, device="cpu"):
        started = time.perf_counter()
        values, vectors = eig(matrix, check_finite=False)
        inverse = np.linalg.inv(vectors)
        self.eigenvector_condition = float(np.linalg.cond(vectors))
        self.device = torch.device(device)
        self.values = torch.tensor(values, device=self.device)
        self.vectors = torch.tensor(vectors, device=self.device)
        self.inverse = torch.tensor(inverse, device=self.device)
        self.denominator = self.values[:, None] + self.values[None]
        self.n = len(values)
        if self.eigenvector_condition > 1e8:
            raise ValueError("ill-conditioned eigenspace; use Schur or sparse fallback")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - started

    def torch_solve(self, rhs):
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        block = rhs.T.reshape(-1, self.n, self.n)
        modal = self.inverse @ block @ self.inverse.T
        solution = self.vectors @ (modal / self.denominator) @ self.vectors.T
        solution = solution.reshape(-1, self.n * self.n).T
        return solution[:, 0] if vector else solution

    def solve(self, rhs):
        value = torch.as_tensor(np.asarray(rhs), dtype=torch.complex128, device=self.device)
        return self.torch_solve(value).detach().cpu().numpy()
