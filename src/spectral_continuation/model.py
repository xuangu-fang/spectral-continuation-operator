"""Matched neural backbones and output parameterizations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class QueryUNet(nn.Module):
    """Four-scale U-Net shared exactly by direct and carrier-relative outputs."""

    def __init__(
        self, input_channels: int, width: int = 16, output_channels: int = 2
    ):
        super().__init__()
        self.e1 = ConvBlock(input_channels, width)
        self.e2 = ConvBlock(width, 2 * width)
        self.e3 = ConvBlock(2 * width, 4 * width)
        self.e4 = ConvBlock(4 * width, 8 * width)
        self.mid = ConvBlock(8 * width, 8 * width)
        self.d4 = ConvBlock(16 * width, 8 * width)
        self.d3 = ConvBlock(12 * width, 4 * width)
        self.d2 = ConvBlock(6 * width, 2 * width)
        self.d1 = ConvBlock(3 * width, width)
        self.out = nn.Conv2d(width, output_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(inputs)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        e4 = self.e4(F.avg_pool2d(e3, 2))
        value = self.mid(F.avg_pool2d(e4, 2))
        value = F.interpolate(value, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        value = self.d4(torch.cat((value, e4), dim=1))
        value = F.interpolate(value, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        value = self.d3(torch.cat((value, e3), dim=1))
        value = F.interpolate(value, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        value = self.d2(torch.cat((value, e2), dim=1))
        value = F.interpolate(value, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.d1(torch.cat((value, e1), dim=1)))


def reconstruct(
    raw: torch.Tensor,
    carrier: torch.Tensor,
    target_scale: float,
    parameterization: str,
) -> torch.Tensor:
    value = torch.complex(raw[:, 0], raw[:, 1]) * target_scale
    if parameterization == "direct":
        output = value
    elif parameterization == "carrier_relative":
        phase_reference = torch.complex(carrier[:, 0], carrier[:, 1])
        output = phase_reference * value
    else:
        raise ValueError(f"unknown parameterization: {parameterization}")
    return torch.stack((output.real, output.imag), dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
