import numpy as np
import torch

from spectral_continuation.audit_phase_reduced_operator import discrete_operator, projected_query
from spectral_continuation.coherent_helmholtz_data import environment, solve
from spectral_continuation.learned_phase_rom import (
    EnvelopeCoordinate,
    PhaseCoordinate,
    torch_projection,
)


def test_torch_projection_matches_classical_and_has_finite_phase_gradient():
    spec = environment(20261003)
    fields, _, _, _ = solve(spec, 33, [3.0, 4.0])
    lap, mass, rhs = discrete_operator(spec, 33)
    basis = fields[:, 1:-1, 1:-1].reshape(2, -1).T
    coo = lap.tocoo()
    t_lap = torch.sparse_coo_tensor(
        np.stack((coo.row, coo.col)), coo.data.astype(complex), coo.shape
    ).coalesce()
    phase = torch.zeros(len(rhs), dtype=torch.float64, requires_grad=True)
    t_basis = torch.tensor(basis) * torch.exp(1j * phase[:, None])
    prediction = torch_projection(t_lap, torch.tensor(mass), torch.tensor(rhs), t_basis, 3.5, 0)
    classical, _, _ = projected_query(lap, mass, rhs, basis, 3.5, "least_residual")
    np.testing.assert_allclose(prediction.detach().numpy(), classical, rtol=1e-9, atol=1e-12)
    prediction.abs().square().sum().backward()
    assert phase.grad is not None and torch.isfinite(phase.grad).all()


def test_initial_coordinate_and_homogeneous_constraint():
    f = torch.randn(17, 9)
    reference = torch.rand(17, dtype=torch.float64)
    straight = reference + 0.1
    for kind in ["scalar", "neural"]:
        model = PhaseCoordinate(kind)
        torch.testing.assert_close(model(f, reference, straight), straight)
        torch.testing.assert_close(model(f, reference, reference), reference)


def test_learned_envelope_starts_at_fixed_radial_space():
    item = {
        "features": torch.randn(17, 9, dtype=torch.float64),
        "coordinates": {"reference_phase": torch.rand(17, dtype=torch.float64)},
        "source_width": 0.025,
    }
    for kind in ["envelope_scalar", "envelope_neural"]:
        model = EnvelopeCoordinate(kind).double()
        factors = model.factors(item)
        torch.testing.assert_close(factors, model.fixed_factors(item))
        factors.square().sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
