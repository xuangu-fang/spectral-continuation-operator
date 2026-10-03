"""Learn a phase-geometry initialization through a few context-only updates.

Inner objective is profiled complex Gaussian marginal likelihood. The only outer
training objective is held-coordinate complex error. Query values never enter
inference. This tests amortized inference, not a new meta-learning algorithm.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .learned_coherent_kernel import CoherentKernelEncoder


class UnrolledCoherentKernel(CoherentKernelEncoder):
    def __init__(self, inner_steps=2, initial_step=0.02, gradient_softness=1.0, **kwargs):
        super().__init__(**kwargs)
        self.inner_steps, self.gradient_softness = inner_steps, gradient_softness
        dimensions = 8 if self.curvature else 5
        if not 0 < initial_step < 0.1:
            raise ValueError("initial step must be between zero and 0.1")
        logit = math.log(initial_step / (0.1 - initial_step))
        self.step_logits = nn.Parameter(torch.full((inner_steps, dimensions), logit))

    def forward(self, x, y, query, weights=None):
        if weights is None:
            weights = torch.ones_like(y.real)
        raw = self.raw_geometry(x, y, weights).double()
        valid = (weights > 0).double()
        labels = y.to(torch.complex128) * valid
        count = valid.sum(-1)
        eye = torch.eye(x.shape[1], dtype=torch.complex128, device=x.device)
        diagonal = self.ridge * valid + (1 - valid)

        def gram(k, lower):
            kernel = self.covariance(x, x, k, lower) * valid[:, :, None] * valid[:, None, :]
            return kernel + diagonal[:, :, None] * eye

        with torch.enable_grad():
            if not self.training:
                raw = raw.detach().requires_grad_(True)
            for step in range(self.inner_steps):
                k, lower = self.geometry_from_raw(raw)
                kernel = gram(k, lower)
                coefficients = torch.linalg.solve(kernel, labels[..., None]).squeeze(-1)
                variance = (labels.conj() * coefficients).sum(-1).real / count
                objective = (
                    variance.clamp_min(1e-30).log() + torch.linalg.slogdet(kernel)[1] / count
                )
                gradient = torch.autograd.grad(objective.sum(), raw, create_graph=self.training)[0]
                direction = gradient / (gradient.square() + self.gradient_softness**2).sqrt()
                raw = raw - (0.1 * self.step_logits[step].sigmoid()) * direction
                if not self.training:
                    raw = raw.detach().requires_grad_(True)
        with torch.set_grad_enabled(self.training):
            k, lower = self.geometry_from_raw(raw)
            coefficients = torch.linalg.solve(gram(k, lower), labels[..., None])
            result = (self.covariance(query, x, k, lower) * valid[:, None]) @ coefficients
            return result.squeeze(-1)
