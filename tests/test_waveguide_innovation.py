import numpy as np
import torch

from spectral_continuation.learned_waveguide_generator import GeneratorEncoder
from spectral_continuation.train_waveguide_innovation import expanded_model, stable_innovation


def test_residual_invariant_to_global_phase_and_source_scale():
    rng = np.random.default_rng(13)
    observed = rng.normal(size=(2, 5, 3)) + 1j * rng.normal(size=(2, 5, 3))
    predicted = observed * 0.8
    initial = np.array([1.0, 0.5, 0.2], dtype=complex)
    mass = np.eye(3)
    kz2 = np.array([0.0, 30.0, 10000.0])  # include strongly evanescent components
    args = (mass, kz2, [1.0, 2.0], np.arange(5.0))
    a = stable_innovation(observed, predicted, initial, *args)
    phase_scale = 3.7 * np.exp(0.78j)
    b = stable_innovation(
        observed * phase_scale, predicted * phase_scale, initial * phase_scale, *args
    )
    np.testing.assert_allclose(a, b, atol=1e-13)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(stable_innovation(observed, observed, initial, *args), 0)


def test_zero_added_weights_preserve_pretrained_prediction():
    torch.manual_seed(13)
    options = {"width": 8, "modes": 4, "layers": 2, "terms": 3}
    original = GeneratorEncoder(5, **options)
    torch.nn.init.normal_(original.readout[-1].weight)
    expanded = expanded_model(original.state_dict(), 5, 7, options)
    x, extra, baseline = torch.randn(2, 5, 16), torch.randn(2, 7, 16), torch.randn(2, 16, 3)
    torch.testing.assert_close(
        original(x, baseline), expanded(torch.cat([x, extra], dim=1), baseline)
    )
