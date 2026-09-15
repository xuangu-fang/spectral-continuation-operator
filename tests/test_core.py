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
    CarrierGaugedPoleOperator,
    ComplexDelayOperator,
    StableSpectralStateOperator,
    phase_equivariance_error,
)
from spectral_continuation.representation_data import (
    FrequencyBandDataset,
    fit_representation_normalization,
)
from spectral_continuation.train_representation_poc import complex_relative_loss


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
    grouped = model.forward_queries(fields, context, (8.0, 9.0))
    separate = torch.stack(
        [
            model(fields, context, torch.full((2,), frequency))
            for frequency in (8.0, 9.0)
        ],
        dim=1,
    )
    assert torch.allclose(grouped, separate)


def test_carrier_gauged_poles_are_continuous_anchored_and_phase_equivariant():
    torch.manual_seed(7)
    visible = (1.0, 2.0, 3.0, 4.0, 5.0)
    model = CarrierGaugedPoleOperator(
        visible, environment_channels=2, poles=3, width=8, blocks=2
    )
    fields = torch.complex(torch.randn(2, 5, 8, 12), torch.randn(2, 5, 8, 12))
    context = torch.randn(2, 5, 8, 12)
    anchor = torch.full((2,), 5.0)
    fractional = torch.tensor([6.25, 7.75])
    anchored = model(fields, context, anchor)
    queried = model(fields, context, fractional)
    grouped_query = model.forward_queries(fields, context, (6.25, 7.75))
    loss = model.visible_reconstruction_loss(fields, context)
    equivariance = phase_equivariance_error(model, fields, context, fractional)
    assert torch.allclose(anchored, fields[:, -1], rtol=1e-5, atol=1e-5)
    assert queried.shape == fields[:, -1].shape
    assert torch.allclose(grouped_query[:, 0], model(fields, context, torch.full((2,), 6.25)))
    assert torch.isfinite(queried.real).all()
    assert torch.isfinite(queried.imag).all()
    assert torch.isfinite(loss)
    assert equivariance < 2e-5

    identified = CarrierGaugedPoleOperator(
        visible,
        environment_channels=2,
        poles=2,
        width=8,
        blocks=1,
        residue_mode="least_squares",
    )
    identified_output = identified(fields, context, fractional)
    identified_loss = identified.visible_reconstruction_loss(fields, context)
    assert torch.isfinite(identified_output.real).all()
    assert torch.isfinite(identified_output.imag).all()
    assert torch.isfinite(identified_loss)
    assert phase_equivariance_error(identified, fields, context, fractional) < 2e-5

    stable_sum = CarrierGaugedPoleOperator(
        visible,
        environment_channels=2,
        poles=3,
        width=8,
        blocks=1,
        pole_growth="stable",
        pole_anchor="residue_sum",
        learn_carrier_correction=True,
    )
    stable_state = stable_sum.spectral_state(fields, context)
    assert torch.all(stable_state[2].real <= 0)
    assert torch.allclose(
        stable_state[1].sum(dim=1),
        torch.complex(fields[:, -1].abs(), torch.zeros_like(fields[:, -1].abs())),
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.allclose(
        stable_sum(fields, context, anchor), fields[:, -1], rtol=1e-5, atol=1e-5
    )

    endpoint = CarrierGaugedPoleOperator(
        visible,
        environment_channels=2,
        poles=2,
        width=8,
        blocks=1,
        carrier_estimator="endpoint_increment",
    )
    expected_increment = torch.angle(
        fields[:, -1] * torch.conj(fields[:, -2])
    )
    assert torch.allclose(endpoint.carrier_slope(fields), expected_increment)


def test_frequency_band_dataset_returns_one_environment_with_all_queries():
    bundle = make_bundle()
    positions = np.asarray([0, 1])
    visible = (10.0, 15.0)
    queries = (20.0, 40.0)
    normalization = fit_representation_normalization(
        bundle, positions, visible, queries
    )
    dataset = FrequencyBandDataset(
        bundle, positions, visible, queries, normalization
    )
    fields, environment, targets = dataset[0]
    assert len(dataset) == 2
    assert fields.shape == (2, 16, 32)
    assert environment.shape == (1, 16, 32)
    assert targets.shape == (2, 16, 32)
    assert torch.is_complex(fields)
    assert torch.is_complex(targets)


def test_relative_training_loss_can_match_reported_unsquared_metric():
    target = torch.ones(2, 4, 8, dtype=torch.complex64)
    prediction = torch.stack((0.5 * target[0], torch.zeros_like(target[1])))
    assert torch.allclose(
        complex_relative_loss(prediction, target, power=1), torch.tensor(0.75)
    )
    assert torch.allclose(
        complex_relative_loss(prediction, target, power=2), torch.tensor(0.625)
    )
