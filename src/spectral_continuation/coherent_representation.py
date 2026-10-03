"""Continuous complex representation audit; no query-truth coefficient fitting.

R0 models are fitted to explicitly provided points of ONE field. This is an
approximation/optimization diagnostic, not system generalization. Local complex
plane waves are classical PWPUM/PWNN baselines, not a novelty claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def complex_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p, t = np.asarray(prediction, dtype=np.complex128), np.asarray(target, dtype=np.complex128)
    if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("same shape, finite predictions and targets required")
    energy = np.abs(t) ** 2
    den = float(energy.sum())
    if den <= 0:
        raise ValueError("zero-energy reference has undefined relative metrics")
    angle = np.angle(p * t.conj())
    invalid = np.abs(p) <= max(float(np.sqrt(energy.mean())) * 1e-12, 1e-30)
    angle = np.where(invalid, np.pi, angle)
    return {
        "complex_rel_l2": float(np.sqrt(np.sum(np.abs(p - t) ** 2) / den)),
        "amplitude_rel_l2": float(np.sqrt(np.sum((np.abs(p) - np.abs(t)) ** 2) / den)),
        "energy_weighted_phase_rmse_rad": float(np.sqrt(np.sum(energy * angle**2) / den)),
        "energy_weighted_phase_cos": float(np.sum(energy * np.cos(angle)) / den),
        "zero_prediction_energy_fraction": float(np.sum(energy * invalid) / den),
        "complex_coherence": float(
            abs(np.vdot(t, p)) / max(np.linalg.norm(t) * np.linalg.norm(p), 1e-30)
        ),
    }


def regular_fit_mask(shape: tuple[int, int], stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("stride must be positive")
    iy = np.unique(np.r_[np.arange(0, shape[0], stride), shape[0] - 1])
    ix = np.unique(np.r_[np.arange(0, shape[1], stride), shape[1] - 1])
    mask = np.zeros(shape, dtype=bool)
    mask[np.ix_(iy, ix)] = True
    return mask


@dataclass
class PatchBatch:
    coordinates: torch.Tensor
    target: torch.Tensor
    weights: torch.Tensor
    centers: np.ndarray
    spacing: np.ndarray


def build_patches(
    coordinates: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    spacing: float | np.ndarray,
    device: str = "cuda",
) -> PatchBatch:
    """Hat-function partition of unity on physical coordinates, only fit labels."""
    coords = np.asarray(coordinates).reshape(-1, 2)
    lo, hi = coords.min(0), coords.max(0)
    counts = np.maximum(np.ceil((hi - lo) / np.asarray(spacing)).astype(int), 1)
    spacing = (hi - lo) / counts
    axes = [np.linspace(lo[d], hi[d], counts[d] + 1) for d in range(2)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    x, y = coords[fit_mask.ravel()], target.ravel()[fit_mask.ravel()]
    patches = []
    for c in centers:
        delta = (x - c) / spacing
        weights = np.prod(np.maximum(1 - np.abs(delta), 0), axis=-1)
        inside = weights > 1e-10
        if inside.sum() < 4:
            raise ValueError("too few fitted samples per patch; increase patch spacing")
        patches.append((delta[inside], y[inside], weights[inside]))
    n = max(len(p[0]) for p in patches)
    xx = np.zeros((len(patches), n, 2))
    yy = np.zeros((len(patches), n), dtype=np.complex128)
    ww = np.zeros((len(patches), n))
    for i, (x, y, w) in enumerate(patches):
        xx[i, : len(x)], yy[i, : len(y)], ww[i, : len(w)] = x, y, w
    return PatchBatch(
        torch.tensor(xx, device=device),
        torch.tensor(yy, device=device),
        torch.tensor(ww, device=device),
        centers,
        spacing,
    )


class LocalWaveCharts(nn.Module):
    """Differentiable variable projection for local coherent wave charts.

    Fits wavevectors (optional), eliminates complex amplitudes with ridge LS.
    Shared decoder formula; coefficients and vectors remain per-instance R0.
    """

    def __init__(
        self,
        patches: PatchBatch,
        wavenumber: float,
        branches: int,
        degree: int = 0,
        learnable: bool = False,
        curvature: bool = False,
        ridge: float = 1e-7,
    ):
        super().__init__()
        if degree not in (0, 1):
            raise ValueError("only constant/affine envelopes supported")
        self.degree, self.ridge, self.curved = degree, ridge, curvature
        self.register_buffer("centers", torch.tensor(patches.centers, device=patches.target.device))
        self.register_buffer("spacing", torch.tensor(patches.spacing, device=patches.target.device))
        angles = torch.arange(branches, device=self.spacing.device) * (2 * math.pi / branches)
        vectors = torch.stack((angles.cos(), angles.sin()), -1).double()
        vectors = vectors * self.spacing * wavenumber
        self.k = nn.Parameter(
            vectors[None].repeat(len(patches.centers), 1, 1), requires_grad=learnable
        )
        self.hessian = nn.Parameter(
            torch.zeros((*self.k.shape[:-1], 3), dtype=torch.float64, device=self.k.device),
            requires_grad=curvature,
        )
        self.register_buffer(
            "coefficients",
            torch.zeros(
                (len(patches.centers), branches * (1 if degree == 0 else 3)),
                dtype=torch.complex128,
                device=self.k.device,
            ),
        )

    def basis(self, x: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        k = self.k if indices is None else self.k[indices]
        phase = torch.einsum("pnd,pjd->pnj", x, k)
        if self.curved:
            quadratic = torch.stack((x[..., 0] ** 2, x[..., 0] * x[..., 1], x[..., 1] ** 2), -1)
            h = self.hessian if indices is None else self.hessian[indices]
            phase = phase + torch.einsum("pnd,pjd->pnj", quadratic, h)
        waves = torch.exp(1j * phase)
        if self.degree:
            poly = torch.cat((torch.ones_like(x[..., :1]), x), -1)
            waves = (waves[..., None] * poly[..., None, :]).flatten(-2)
        return waves

    def project(self, patches: PatchBatch) -> tuple[torch.Tensor, torch.Tensor]:
        basis = self.basis(patches.coordinates)
        weighted = basis * patches.weights[..., None].sqrt()
        labels = patches.target * patches.weights.sqrt()
        gram = weighted.mH @ weighted
        penalty = self.ridge * gram.diagonal(dim1=-2, dim2=-1).real.mean(-1)
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        coeff = torch.linalg.solve(
            gram + penalty[:, None, None] * eye, (weighted.mH @ labels[..., None])
        ).squeeze(-1)
        prediction = (basis * coeff[:, None]).sum(-1)
        loss = ((prediction - patches.target).abs().square() * patches.weights).sum()
        loss = loss / (patches.target.abs().square() * patches.weights).sum()
        return loss, coeff

    @torch.no_grad()
    def set_coefficients(self, patches: PatchBatch) -> float:
        loss, coefficients = self.project(patches)
        self.coefficients.copy_(coefficients)
        return float(loss)

    @torch.no_grad()
    def predict(self, coordinates: np.ndarray, chunk: int = 2048) -> np.ndarray:
        xy = torch.as_tensor(coordinates.reshape(-1, 2), device=self.k.device, dtype=torch.float64)
        out = []
        for start in range(0, len(xy), chunk):
            delta = (xy[start : start + chunk][None] - self.centers[:, None]) / self.spacing
            weights = (1 - delta.abs()).clamp_min(0).prod(-1)
            pred = (self.basis(delta) * self.coefficients[:, None]).sum(-1)
            value = (weights * pred).sum(0) / weights.sum(0).clamp_min(1e-12)
            out.append(value.cpu().numpy())
        return np.concatenate(out).reshape(coordinates.shape[:-1])

    def stored_real_scalars(self) -> int:
        # Includes non-trainable wavenumbers, coefficients, chart locations/scales.
        return (
            self.k.numel()
            + 2 * self.coefficients.numel()
            + self.centers.numel()
            + self.spacing.numel()
            + (self.hessian.numel() if self.curved else 0)
        )


class SineLayer(nn.Module):
    def __init__(self, ni: int, no: int, first: bool = False, omega: float = 30):
        super().__init__()
        self.linear, self.omega = nn.Linear(ni, no), omega
        limit = 1 / ni if first else math.sqrt(6 / ni) / omega
        nn.init.uniform_(self.linear.weight, -limit, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class ComplexINR(nn.Module):
    def __init__(self, kind: str, width: int, layers: int = 3, bandwidth: float = 30):
        super().__init__()
        self.kind = kind
        if kind == "siren":
            blocks = [SineLayer(2, width, first=True, omega=bandwidth)]
            blocks += [SineLayer(width, width) for _ in range(layers - 1)]
        elif kind == "fourier":
            self.register_buffer("features", torch.randn(2, width // 2) * bandwidth)
            blocks = [nn.Linear(width, width), nn.GELU()]
            for _ in range(layers - 1):
                blocks.extend((nn.Linear(width, width), nn.GELU()))
        else:
            raise ValueError(kind)
        final = nn.Linear(width, 2)
        if kind == "siren":
            nn.init.uniform_(final.weight, -math.sqrt(6 / width) / 30, math.sqrt(6 / width) / 30)
        self.net = nn.Sequential(*blocks, final)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "fourier":
            p = x @ self.features
            x = torch.cat((p.cos(), p.sin()), -1)
        y = self.net(x)
        return torch.complex(y[..., 0], y[..., 1])
