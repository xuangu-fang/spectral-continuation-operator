import numpy as np
import torch

from spectral_continuation.data import (
    FieldBundle,
    MultiQueryDataset,
    SpectralQueryDataset,
    fit_normalization,
    spectral_carrier,
)
from spectral_continuation.metrics import metric_vectors, paired_bootstrap
from spectral_continuation.model import QueryUNet, reconstruct


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
