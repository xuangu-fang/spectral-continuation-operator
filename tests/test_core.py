import numpy as np
import torch

from spectral_continuation.data import (
    FieldBundle,
    MultiQueryDataset,
    SpectralQueryDataset,
    fit_normalization,
    load_ocean_npz,
    spectral_carrier,
)
from spectral_continuation.evaluate_horizon import verify_identity
from spectral_continuation.metrics import metric_vectors, paired_bootstrap
from spectral_continuation.model import QueryUNet, reconstruct
from spectral_continuation.representation import (
    ComplexDelayOperator,
    StableSpectralStateOperator,
    phase_equivariance_error,
)


def make_bundle():
    height, width = 16, 32
    tau = torch.linspace(0.0, 0.02, width).view(1, 1, width).expand(4, height, width)
    fields = {}
    for frequency in (10.0, 15.0, 20.0, 40.0):
        value = torch.polar(torch.ones_like(tau), 2 * torch.pi * frequency * tau)
        fields[frequency] = torch.stack((value.real, value.imag), dim=1)
    return FieldBundle(
        fields=fields,
        environment=torch.ones(4, 1, height, width),
        ids=np.arange(4),
        split=np.asarray(["train", "train", "val", "val"]),
        frequencies=(10.0, 15.0, 20.0, 40.0),
        lineage={},
    )


def test_spectral_secant_is_exact_for_unaliased_single_path():
    bundle = make_bundle()
    carrier = spectral_carrier(bundle.fields, (10.0, 15.0, 20.0), 40.0, np.arange(4))
    expected = bundle.fields[40.0]
    assert torch.allclose(carrier, expected, atol=2e-5)


def test_dataset_and_matched_reconstruction_contract():
    bundle = make_bundle()
    train = np.asarray([0, 1])
    norm = fit_normalization(bundle, train, (10.0, 15.0, 20.0), 40.0)
    dataset = SpectralQueryDataset(bundle, train, (10.0, 15.0, 20.0), 40.0, norm)
    inputs, carrier, target = dataset[0]
    model = QueryUNet(dataset.input_channels, width=8)
    raw = model(inputs.unsqueeze(0))
    direct = reconstruct(raw, carrier.unsqueeze(0), norm.target_scale, "direct")
    relative = reconstruct(raw, carrier.unsqueeze(0), norm.target_scale, "carrier_relative")
    assert direct.shape == relative.shape == target.unsqueeze(0).shape
    assert sum(p.numel() for p in model.parameters()) > 0

    multi = MultiQueryDataset(bundle, train, (10.0, 15.0, 20.0), (20.0, 40.0), norm)
    assert len(multi) == 4
    assert multi[0][-1] == 0
    assert multi[1][-1] == 1


def test_metrics_and_bootstrap():
    target = torch.randn(6, 2, 8, 8)
    exact = metric_vectors(target, target)
    assert np.allclose(exact["complex_rel_l2"], 0.0)
    stats = paired_bootstrap(np.ones(6), np.full(6, 0.8), replicates=100)
    np.testing.assert_allclose(stats["relative_improvement_percent"], 20.0)


def test_ocean_loader_excludes_solver_inactive_channel(tmp_path):
    path = tmp_path / "ocean.npz"
    names = np.asarray(
        [
            "water_speed_mps",
            "water_depth_m",
            "seabed_speed_mps",
            "seabed_density_kgm3",
            "seabed_attenuation_db_per_wavelength",
        ]
    )
    environment = np.stack(
        [np.full((5, 4, 8), index, dtype=np.float32) for index in range(2)], axis=0
    )
    np.savez(
        path,
        complex_field=np.zeros((2, 2, 4, 8, 2), dtype=np.float32),
        frequencies_hz=np.asarray([79.0, 80.0]),
        environment_channels=environment,
        environment_channel_names=names,
        environment_ids=np.asarray(["a", "b"]),
        split=np.asarray(["train", "val"]),
    )
    bundle = load_ocean_npz(path)
    assert bundle.environment.shape == (2, 4, 4, 8)
    assert bundle.lineage["environment_channels"] == names[:4].tolist()
    assert bundle.lineage["excluded_environment_channels"] == [names[-1]]


def test_dense_horizon_identity_check_rejects_changed_ids():
    source = make_bundle()
    dense = make_bundle()
    verify_identity(source, dense)
    dense.ids = np.asarray([0, 1, 2, 99])
    with np.testing.assert_raises(RuntimeError):
        verify_identity(source, dense)


def test_complex_delay_operator_is_phase_equivariant_and_resolution_flexible():
    model = ComplexDelayOperator((8.0, 9.0, 10.0), 0, modes=8, blocks=1)
    for height, width in ((16, 32), (24, 48)):
        fields = torch.randn(2, 3, height, width, dtype=torch.complex64)
        context = torch.randn(2, 3, height, width)
        query = torch.tensor([11.0, 12.0])
        output = model(fields, context, query)
        assert output.shape == (2, height, width)
        assert phase_equivariance_error(model, fields, context, query) < 2e-6


def test_stable_spectral_state_is_phase_equivariant_and_has_finite_history_loss():
    model = StableSpectralStateOperator(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        0,
        order=3,
        width=8,
        blocks=1,
    )
    fields = torch.randn(2, 7, 12, 24, dtype=torch.complex64)
    context = torch.randn(2, 3, 12, 24)
    query = torch.tensor([8.0, 9.0])
    assert phase_equivariance_error(model, fields, context, query) < 2e-6
    assert torch.isfinite(model.visible_reconstruction_loss(fields, context))
