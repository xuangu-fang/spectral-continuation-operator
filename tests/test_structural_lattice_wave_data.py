import numpy as np
import torch

from spectral_continuation.generate_structural_lattice_wave_data import (
    assemble_matrices,
    perimeter_nodes,
)


def test_perimeter_nodes_are_unique() -> None:
    nodes = perimeter_nodes(6)
    assert len(nodes) == 20
    assert len(np.unique(nodes)) == len(nodes)


def test_structural_matrices_are_symmetric_and_positive() -> None:
    density = torch.ones((2, 4, 4), dtype=torch.float64)
    stiffness = torch.ones_like(density)
    anisotropy = torch.ones((2, 2), dtype=torch.float64)
    mass, spring = assemble_matrices(density, stiffness, anisotropy, 0.2)
    assert torch.allclose(mass, mass.mT)
    assert torch.allclose(spring, spring.mT)
    assert torch.linalg.eigvalsh(mass).min() > 0
    assert torch.linalg.eigvalsh(spring).min() > 0
