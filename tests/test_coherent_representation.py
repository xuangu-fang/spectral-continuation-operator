import numpy as np
import torch

from spectral_continuation.coherent_representation import (
    LocalWaveCharts,
    build_patches,
    complex_metrics,
    regular_fit_mask,
)


def test_metrics_keep_absolute_phase_and_penalize_zero():
    t = np.ones(10, dtype=complex)
    assert complex_metrics(t, t)["complex_rel_l2"] == 0
    assert np.isclose(complex_metrics(1j * t, t)["energy_weighted_phase_rmse_rad"], np.pi / 2)
    assert np.isclose(complex_metrics(0 * t, t)["energy_weighted_phase_rmse_rad"], np.pi)


def test_partition_and_plane_wave_prediction():
    x = np.linspace(0, 1, 25)
    xx, yy = np.meshgrid(x, x)
    xy = np.stack((xx, yy), -1)
    target = np.exp(1j * 4 * np.pi * xx)
    fit = regular_fit_mask(target.shape, 2)
    patches = build_patches(xy, target, fit, 0.3, "cpu")
    model = LocalWaveCharts(patches, 4 * np.pi, 4, ridge=1e-10)
    model.set_coefficients(patches)
    pred = model.predict(xy)
    assert complex_metrics(pred[~fit], target[~fit])["complex_rel_l2"] < 1e-6


def test_variable_projection_gradient():
    x = np.linspace(0, 1, 20)
    xx, yy = np.meshgrid(x, x)
    xy = np.stack((xx, yy), -1)
    target = np.exp(1j * 5 * (xx + 0.3 * yy))
    fit = regular_fit_mask(target.shape, 2)
    patches = build_patches(xy, target, fit, 0.5, "cpu")
    model = LocalWaveCharts(patches, 5, 4, learnable=True)
    loss, _ = model.project(patches)
    loss.backward()
    assert torch.isfinite(model.k.grad).all()
    assert model.k.grad.abs().max() > 0
