"""Multiple coherent phase frames, not one effective carrier for interference.

This is a spectral-mixture / nonstationary GP construction, NOT a novelty claim.
Positive mixing weights retain a Hermitian positive-definite kernel. Observed
complex values are jointly decoded, preserving relative and absolute phase.
"""

from __future__ import annotations

import torch
from torch import nn

from .learned_coherent_kernel import CoherentKernelEncoder
from .learned_wave_chart import WaveChartEncoder


class MixtureCoherentKernelEncoder(WaveChartEncoder):
    def __init__(self, components=2, width=128, extent=0.75, ridge=1e-8, learned=True):
        super().__init__(
            branches=1, degree=0, width=width, extent=extent, ridge=ridge, learned=learned
        )
        self.components = components
        self.encoder[-1] = nn.Linear(width, components * 9)
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        base = torch.zeros(components, 9)
        # Break identical-component symmetry without looking at query labels.
        if components > 1:
            angle = torch.arange(components) * 2 * torch.pi / components
            base[:, :2] = 0.025 * torch.stack((angle.cos(), angle.sin()), -1)
        self.register_buffer("initial_raw", base.flatten())

    def raw_geometry(self, x, y, weights):
        base = self.initial_raw[None].expand(len(x), -1)
        if not self.learned:
            return base
        return base + self.encoder(self.encode_features(x, y, weights))

    def geometry_from_raw(self, raw):
        raw = raw.double().reshape(len(raw), self.components, 9)
        k = torch.cat((10 * raw[..., :2].tanh(), 6 * raw[..., 5:8].tanh()), -1)
        lower = torch.zeros((*raw.shape[:2], 2, 2), device=raw.device, dtype=torch.float64)
        lower[..., 0, 0] = torch.exp(2 * raw[..., 2].tanh())
        lower[..., 1, 1] = torch.exp(2 * raw[..., 3].tanh())
        lower[..., 1, 0] = 2 * raw[..., 4].tanh()
        return k, lower, raw[..., 8].softmax(-1)

    @staticmethod
    def covariance(a, b, k, lower, mixture):
        batch, count = k.shape[:2]
        a = a[:, None].expand(-1, count, -1, -1).reshape(batch * count, a.shape[1], 2)
        b = b[:, None].expand(-1, count, -1, -1).reshape(batch * count, b.shape[1], 2)
        covariance = CoherentKernelEncoder.covariance(
            a, b, k.reshape(batch * count, 5), lower.reshape(batch * count, 2, 2)
        ).reshape(batch, count, a.shape[1], b.shape[1])
        return (mixture[..., None, None] * covariance).sum(1)

    def forward(self, x, y, query, weights=None):
        if weights is None:
            weights = torch.ones_like(y.real)
        geometry = self.geometry_from_raw(self.raw_geometry(x, y, weights))
        kernel = self.covariance(x, x, *geometry)
        valid = (weights > 0).double()
        kernel = kernel * valid[:, :, None] * valid[:, None, :]
        diagonal = self.ridge * valid + (1 - valid)
        kernel = kernel + torch.diag_embed(diagonal)
        alpha = torch.linalg.solve(kernel, (y.to(torch.complex128) * valid)[..., None])
        return ((self.covariance(query, x, *geometry) * valid[:, None]) @ alpha).squeeze(-1)
