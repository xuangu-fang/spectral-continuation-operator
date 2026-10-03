import numpy as np
import torch

from spectral_continuation.audit_phase_reduced_operator import discrete_operator, projected_query
from spectral_continuation.coherent_helmholtz_data import environment, solve
from spectral_continuation.learned_phase_rom import (
    EnvelopeCoordinate,
    PhaseCoordinate,
    predict,
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


def test_complex_envelope_initialization_and_frequency_phase_law():
    item = {
        "features": torch.randn(17, 9, dtype=torch.float64),
        "coordinates": {"reference_phase": torch.rand(17, dtype=torch.float64)},
        "source_width": 0.025,
        "source_regularized": True,
        "observed_f": torch.tensor([6.0, 10.0, 14.0]),
    }
    for kind in ["envelope_complex_global", "envelope_complex_neural"]:
        model = EnvelopeCoordinate(kind).double()
        factors = model.factors(item, 10.0)
        torch.testing.assert_close(factors, model.fixed_factors(item).to(torch.complex128))
        with torch.no_grad():
            if kind.endswith("global"):
                model.raw[2:] = 0.03
            else:
                model.network[-1].bias[2:] = 0.03
        low, high = model.factors(item, 5.0), model.factors(item, 10.0)
        torch.testing.assert_close(torch.angle(high[:, 1:]), 2 * torch.angle(low[:, 1:]))
        high.real.square().sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_source_aware_readout_matches_explicit_complex_basis():
    spec = environment(20261003)
    fields, _, _, _ = solve(spec, 33, [3.0, 4.0])
    lap, mass, rhs = discrete_operator(spec, 33)
    coo = lap.tocoo()
    snapshots = fields[:, 1:-1, 1:-1].reshape(2, -1).T
    rng = np.random.default_rng(47)
    observed = np.exp(rng.normal(size=snapshots.shape) * 0.1j)
    query = np.exp(rng.normal(size=len(rhs)) * 0.1j)
    item = {
        "snapshots": torch.tensor(snapshots),
        "reference_carriers": {3.5: torch.tensor(query)},
        "observed_reference_carriers": torch.tensor(observed),
        "lap": torch.sparse_coo_tensor(
            np.stack((coo.row, coo.col)), coo.data.astype(complex), coo.shape
        ).coalesce(),
        "mass": torch.tensor(mass),
        "rhs": torch.tensor(rhs),
    }
    factors = torch.ones(len(rhs), 1, dtype=torch.float64, requires_grad=True)
    pred = predict(item, None, 3.5, {"ridge": 0}, factors, "source_aware")
    basis = snapshots * query[:, None] / observed
    basis = np.column_stack((basis, basis.conj()))
    expected, _, _ = projected_query(lap, mass, rhs, basis, 3.5, "least_residual")
    np.testing.assert_allclose(pred.detach().numpy(), expected, rtol=1e-8, atol=1e-12)
    pred.abs().square().sum().backward()
    assert torch.isfinite(factors.grad).all()
