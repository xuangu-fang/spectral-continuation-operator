"""Conditional carrier and anisotropic Gaussian envelope kernel.

Minimal alternative to a finite plane-wave dictionary: learn the phase frame and
smoothness metric, preserve complex observed values through a linear kernel solve.
The kernel is Hermitian positive definite by construction. This is a local
representation experiment; spectral-mixture kernels/GPs themselves are prior art.
"""

from __future__ import annotations

import torch
from torch import nn

from .learned_wave_chart import WaveChartEncoder


class CoherentKernelEncoder(WaveChartEncoder):
    def __init__(
        self,
        width=128,
        extent=0.75,
        ridge=1e-8,
        learned=True,
        mode="conditional",
        carrier=True,
        metric=True,
        curvature=False,
        encoder_kind="spectral_moments",
    ):
        super().__init__(
            branches=1, degree=0, width=width, extent=extent, ridge=ridge, learned=learned
        )
        self.mode, self.carrier, self.metric = mode, carrier, metric
        self.curvature = curvature
        self.encoder_kind = encoder_kind
        output_count = 8 if curvature else 5
        self.encoder[-1] = nn.Linear(width, output_count)
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        if encoder_kind == "set_attention":
            from .coherent_set_encoder import CoherentSetEncoder

            self.encoder = CoherentSetEncoder(output_count=output_count)
        elif encoder_kind != "spectral_moments":
            raise ValueError(encoder_kind)
        if mode == "global":
            self.encoder = nn.ParameterDict({"delta": nn.Parameter(torch.zeros(output_count))})
        elif mode != "conditional":
            raise ValueError(mode)

    def infer_geometry(self, x, y, weights):
        return self.geometry_from_raw(self.raw_geometry(x, y, weights))

    def raw_geometry(self, x, y, weights):
        if not self.learned:
            raw = torch.zeros((len(x), 8 if self.curvature else 5), dtype=x.dtype, device=x.device)
        elif self.mode == "global":
            raw = self.encoder["delta"][None].expand(len(x), -1)
        elif self.encoder_kind == "set_attention":
            raw = self.encoder(x, y, weights)
        else:
            raw = self.encoder(self.encode_features(x, y, weights))
        return raw

    def geometry_from_raw(self, raw):
        raw = raw.double()
        k = 10 * torch.tanh(raw[:, :2]) if self.carrier else torch.zeros_like(raw[:, :2])
        if self.curvature:
            k = torch.cat((k, 6 * torch.tanh(raw[:, 5:])), dim=-1)
        metric = raw[:, 2:5] if self.metric else torch.zeros_like(raw[:, 2:5])
        lower = torch.zeros((len(raw), 2, 2), dtype=torch.float64, device=raw.device)
        lower[:, 0, 0] = torch.exp(2 * torch.tanh(metric[:, 0]))
        lower[:, 1, 1] = torch.exp(2 * torch.tanh(metric[:, 1]))
        lower[:, 1, 0] = 2 * torch.tanh(metric[:, 2])
        return k, lower

    @staticmethod
    def covariance(a, b, k, lower):
        a, b = a.double(), b.double()
        aa, bb = a @ lower, b @ lower
        distance = (
            aa.square().sum(-1)[:, :, None]
            + bb.square().sum(-1)[:, None, :]
            - 2 * (aa @ bb.transpose(-2, -1))
        ).clamp_min(0)
        if k.shape[-1] == 5:
            a = torch.cat(
                (a, a[..., :1].square(), a.prod(-1, keepdim=True), a[..., 1:].square()), -1
            )
            b = torch.cat(
                (b, b[..., :1].square(), b.prod(-1, keepdim=True), b[..., 1:].square()), -1
            )
        phase_a, phase_b = (a * k[:, None]).sum(-1), (b * k[:, None]).sum(-1)
        return torch.exp(-distance + 1j * (phase_a[:, :, None] - phase_b[:, None, :]))

    def forward(self, x, y, query, weights=None):
        if weights is None:
            weights = torch.ones_like(y.real)
        k, lower = self.infer_geometry(x, y, weights)
        kernel = self.covariance(x, x, k, lower)
        eye = torch.eye(x.shape[1], device=x.device, dtype=torch.complex128)
        # Weights define encoder attention, not a change in observation noise.
        # Zero weights mark padding and must never be treated as zero-valued data.
        valid = (weights > 0).double()
        kernel = kernel * valid[:, :, None] * valid[:, None, :]
        diagonal = self.ridge * valid + (1 - valid)
        coefficients = torch.linalg.solve(
            kernel + diagonal[:, :, None] * eye, (y.to(torch.complex128) * valid)[..., None]
        )
        return ((self.covariance(query, x, k, lower) * valid[:, None]) @ coefficients).squeeze(-1)
