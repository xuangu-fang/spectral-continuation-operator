"""A compact Fourier neural operator used as a response-decoder existence test."""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / width
        self.weight_positive = nn.Parameter(
            scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat)
        )

    @staticmethod
    def _multiply(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", value, weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        spectrum = torch.fft.rfft2(value, norm="ortho")
        output = torch.zeros(
            batch, channels, height, width // 2 + 1, dtype=torch.cfloat, device=value.device
        )
        modes_x = min(self.modes, height // 2)
        modes_y = min(self.modes, width // 2 + 1)
        output[:, :, :modes_x, :modes_y] = self._multiply(
            spectrum[:, :, :modes_x, :modes_y],
            self.weight_positive[:, :, :modes_x, :modes_y],
        )
        output[:, :, -modes_x:, :modes_y] = self._multiply(
            spectrum[:, :, -modes_x:, :modes_y],
            self.weight_negative[:, :, :modes_x, :modes_y],
        )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class FourierResponseOperator(nn.Module):
    """Map medium, query and a mean background response to a complex residual."""

    def __init__(self, input_channels: int = 8, width: int = 48, modes: int = 16, layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(input_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.normalization = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, 2, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.lift(inputs)
        for spectral, local, normalization in zip(
            self.spectral, self.local, self.normalization, strict=True
        ):
            value = torch.nn.functional.gelu(normalization(spectral(value) + local(value)))
        return self.project(value)
