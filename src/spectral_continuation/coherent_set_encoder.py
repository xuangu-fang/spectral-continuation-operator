"""Permutation-invariant, common-phase-invariant encoder of complex samples.

Predicts phase geometry only; absolute complex phase remains in analytic readout.
No query coordinates or values are encoder inputs.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class CoherentSetEncoder(nn.Module):
    def __init__(self, output_count=8, width=64, slots=4):
        super().__init__()
        self.register_buffer("position_scales", math.pi * torch.tensor([1.0, 2.0, 4.0]))
        self.tokenizer = nn.Sequential(nn.Linear(18, width), nn.GELU(), nn.Linear(width, width))
        self.slots = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(slots * width, 128), nn.GELU(), nn.Linear(128, output_count)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, y, weights):
        x, y, weights = x.float(), y.to(torch.complex64), weights.float()
        valid = weights > 0
        index = (y.abs() * valid).argmax(-1)
        reference = y.gather(1, index[:, None])
        relative = y * reference.conj() / reference.abs().clamp_min(1e-8)
        position = (x[..., None] * self.position_scales).flatten(-2)
        features = torch.cat(
            (
                x,
                position.sin(),
                position.cos(),
                relative.real[..., None],
                relative.imag[..., None],
                relative.abs()[..., None],
                weights[..., None],
            ),
            -1,
        )
        tokens = self.tokenizer(features)
        slots = self.slots[None].expand(len(x), -1, -1)
        encoded, _ = self.attention(
            slots, tokens, tokens, key_padding_mask=~valid, need_weights=False
        )
        return self.head(self.norm(slots + encoded).flatten(1))
