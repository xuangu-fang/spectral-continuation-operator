import torch

from spectral_continuation.learned_waveguide_spectral_codec import EnvelopeOperator, SpectralConv1d


def test_spectral_layer_shape_and_gradient():
    layer = SpectralConv1d(8, 12)
    x = torch.randn(2, 8, 33, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()


def test_envelope_model_starts_exactly_at_classical_interpolation():
    model = EnvelopeOperator(226, width=16, modes=8, layers=2)
    features = torch.randn(3, 226, 65)
    baseline = torch.randn(3, 65, 32, dtype=torch.complex64)
    prediction = model(features, baseline)
    torch.testing.assert_close(prediction, baseline)
    prediction.abs().square().mean().backward()
    assert model.readout[-1].weight.grad.abs().max() > 0


def test_query_modulated_operator_starts_at_same_classical_baseline():
    model = EnvelopeOperator(226, width=16, modes=8, layers=2, conditioning="film")
    features = torch.randn(2, 226, 65)
    baseline = torch.randn(2, 65, 32, dtype=torch.complex64)
    torch.testing.assert_close(model(features, baseline), baseline)
    # Query-conditioned feature modulation is initially the identity.
    torch.testing.assert_close(model.query_modulator(features[:, -33:, 0]), torch.zeros(2, 64))


def test_declared_background_carrier_does_not_use_hidden_range_contrasts():
    import numpy as np

    from spectral_continuation.audit_waveguide_spectral_transport import background_modal_carrier
    from spectral_continuation.coherent_waveguide_data import environment

    spec = environment(4)
    first = background_modal_carrier(spec, 100, 32, "boundary_derivative")
    changed = {
        **spec,
        "contrasts_m_s": [33, -27, 13],
        "centers_m": [[400, 20], [500, 60], [1000, 50]],
    }
    second = background_modal_carrier(changed, 100, 32, "boundary_derivative")
    for a, b in zip(first, second, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-13)


def test_compressed_training_loader_does_not_encode_query_labels(tmp_path):
    import json

    import numpy as np

    from spectral_continuation.coherent_waveguide_data import environment
    from spectral_continuation.learned_waveguide_spectral_codec import prepare

    spec = environment(5)
    ranges = np.linspace(0, 2000, 33)
    frequencies = np.array([60, 80, 100, 140])
    coefficients = np.zeros((4, 33, 32), dtype=np.complex64)
    coefficients[..., -1] = np.exp(0.3j * ranges)[None]
    path = tmp_path / "system5_modal.npz"

    def save():
        np.savez(
            path,
            modal_coefficients=coefficients,
            ranges_m=ranges,
            frequencies_hz=frequencies,
            spec=json.dumps(spec),
        )

    save()
    config = {
        "data": str(tmp_path),
        "training_modal_cache": str(tmp_path),
        "train_systems": [5],
        "observed_frequencies": [60, 100, 140],
        "query_frequencies": [80],
        "range_samples": 17,
        "roi_min_range_m": 50,
    }
    first, _, _ = prepare(config, [5])
    coefficients[1] *= -7j
    save()
    second, _, _ = prepare(config, [5])
    np.testing.assert_array_equal(first["features"], second["features"])
    np.testing.assert_array_equal(first["baseline"], second["baseline"])
    assert not np.allclose(first["target"], second["target"])
