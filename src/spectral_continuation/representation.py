"""Complex, phase-equivariant representations for oscillatory frequency fields."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import QueryUNet


class ComplexConv2d(nn.Module):
    """Bias-free complex convolution, equivariant to a global phase rotation."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = dilation * (kernel_size // 2)
        self.dilation = dilation
        shape = (output_channels, input_channels, kernel_size, kernel_size)
        scale = 1.0 / math.sqrt(input_channels * kernel_size * kernel_size)
        self.weight_real = nn.Parameter(scale * torch.randn(shape))
        self.weight_imag = nn.Parameter(scale * torch.randn(shape))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        real = F.conv2d(
            value.real,
            self.weight_real,
            padding=self.padding,
            dilation=self.dilation,
        ) - F.conv2d(
            value.imag,
            self.weight_imag,
            padding=self.padding,
            dilation=self.dilation,
        )
        imag = F.conv2d(
            value.real,
            self.weight_imag,
            padding=self.padding,
            dilation=self.dilation,
        ) + F.conv2d(
            value.imag,
            self.weight_real,
            padding=self.padding,
            dilation=self.dilation,
        )
        return torch.complex(real, imag)


class ComplexModReLU(nn.Module):
    """Magnitude nonlinearity that leaves the complex phase equivariant."""

    def __init__(self, channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        magnitude = value.abs().clamp_min(1e-8)
        return value * (F.relu(magnitude + self.bias) / magnitude)


class ComplexResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels, 3, dilation=dilation)
        self.activation = ComplexModReLU(channels)
        self.conv2 = ComplexConv2d(channels, channels, 3, dilation=dilation)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.conv2(self.activation(self.conv1(value)))
        return value + self.residual_scale * update


class ComplexDelayOperator(nn.Module):
    """Encode a frequency set on a delay dictionary and query it continuously.

    The hidden state is a full-resolution complex field over delay channels.  Its
    analysis and synthesis phases use the same learned delays, while every learned
    complex operation is globally phase equivariant.  There is deliberately no
    spatial pooling.
    """

    def __init__(
        self,
        visible_frequencies: tuple[float, ...],
        environment_channels: int,
        *,
        modes: int = 32,
        blocks: int = 4,
        max_delay: float = 0.65,
        frequency_scale: float = 200.0,
    ):
        super().__init__()
        if len(visible_frequencies) < 2:
            raise ValueError("the delay representation requires at least two frequencies")
        self.register_buffer(
            "visible_frequencies",
            torch.tensor(visible_frequencies, dtype=torch.float32),
        )
        self.reference_frequency = float(max(visible_frequencies))
        self.max_delay = float(max_delay)
        self.frequency_scale = float(frequency_scale)
        initial = torch.linspace(0.01, 0.99, modes)
        self.delay_logits = nn.Parameter(torch.logit(initial))
        self.analysis_gain = nn.Parameter(torch.ones(modes))
        self.decay = nn.Parameter(torch.zeros(modes))
        self.blocks = nn.ModuleList(
            ComplexResidualBlock(modes, dilation=(1, 2, 4, 1)[index % 4])
            for index in range(blocks)
        )
        # The latent state must be query-independent.  The final context channel
        # contains the query frequency and is intentionally excluded here.
        context_channels = environment_channels + 2
        self.context = nn.Sequential(
            nn.Conv2d(context_channels, modes, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(modes, modes, 3, padding=1),
        )
        self.output_gain = nn.Parameter(torch.tensor(1.0))

    @property
    def delays(self) -> torch.Tensor:
        return self.max_delay * torch.sigmoid(self.delay_logits)

    def forward(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequency: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_complex(fields):
            raise TypeError("fields must be a complex tensor")
        if fields.shape[1] != len(self.visible_frequencies):
            raise ValueError("visible field count does not match the configured frequency set")
        delay = self.delays
        offsets = self.visible_frequencies.to(fields.device) - self.reference_frequency
        analysis_phase = -2.0 * torch.pi * delay[:, None] * offsets[None, :]
        analysis = torch.polar(torch.ones_like(analysis_phase), analysis_phase)
        latent = torch.einsum("kf,bfhw->bkhw", analysis, fields) / fields.shape[1]
        latent = latent * self.analysis_gain.view(1, -1, 1, 1)
        invariant_gate = 2.0 * torch.sigmoid(
            self.context(environment_and_coordinates[:, :-1])
        )
        latent = latent * invariant_gate
        for block in self.blocks:
            latent = block(latent)
        delta = query_frequency - self.reference_frequency
        phase = 2.0 * torch.pi * delta[:, None] * delay[None, :]
        amplitude = torch.exp(
            torch.clamp(
                delta[:, None] * self.decay[None, :] / self.frequency_scale,
                min=-2.0,
                max=2.0,
            )
        )
        synthesis = amplitude * torch.polar(torch.ones_like(phase), phase)
        output = torch.einsum("bk,bkhw->bhw", synthesis, latent)
        return self.output_gain * output / latent.shape[1]


class ImageQueryOperator(nn.Module):
    """Conventional real/imag image baseline with the existing U-Net backbone."""

    def __init__(
        self,
        visible_count: int,
        environment_channels: int,
        *,
        width: int = 16,
        frequency_scale: float = 200.0,
    ):
        super().__init__()
        self.frequency_scale = float(frequency_scale)
        self.backbone = QueryUNet(2 * visible_count + environment_channels + 3, width)

    def forward(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequency: torch.Tensor,
    ) -> torch.Tensor:
        real_fields = torch.cat((fields.real, fields.imag), dim=1)
        height, width = fields.shape[-2:]
        query_map = (query_frequency / self.frequency_scale).view(-1, 1, 1, 1)
        query_map = query_map.expand(-1, 1, height, width)
        inputs = torch.cat((real_fields, environment_and_coordinates[:, :-1], query_map), dim=1)
        raw = self.backbone(inputs)
        return torch.complex(raw[:, 0], raw[:, 1])


class PronyFrequencyOperator(nn.Module):
    """Per-location finite-rank spectral recurrence used as an analytic POC.

    A sum of ``order`` damped complex exponentials obeys an order-``order``
    linear recurrence.  Fitting that recurrence is a direct diagnostic of whether
    the observed frequency band identifies a compact propagation state.
    """

    def __init__(
        self,
        visible_frequencies: tuple[float, ...],
        order: int = 5,
        ridge: float = 1e-5,
    ):
        super().__init__()
        if len(visible_frequencies) <= 2 * order:
            raise ValueError("Prony fitting requires more than twice as many samples as modes")
        gaps = torch.diff(torch.tensor(visible_frequencies))
        if not torch.allclose(gaps, gaps[0]):
            raise ValueError("Prony POC currently requires uniformly spaced frequencies")
        self.visible_frequencies = visible_frequencies
        self.order = order
        self.ridge = ridge
        self.frequency_step = float(gaps[0])
        self.reference_frequency = float(visible_frequencies[-1])

    def forward(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequency: torch.Tensor,
    ) -> torch.Tensor:
        del environment_and_coordinates
        batch, count, height, width = fields.shape
        # Recurrence fitting is ill-conditioned by construction; complex128 is
        # required even when the stored wave fields are complex64.
        sequence = fields.permute(0, 2, 3, 1).reshape(-1, count).to(torch.complex128)
        design = torch.stack(
            [
                torch.stack(
                    [sequence[:, index - lag - 1] for lag in range(self.order)],
                    dim=-1,
                )
                for index in range(self.order, count)
            ],
            dim=1,
        )
        target = sequence[:, self.order :]
        adjoint = design.mH
        gram = adjoint @ design
        identity = torch.eye(self.order, dtype=sequence.dtype, device=fields.device)
        coefficients = torch.linalg.solve(
            gram + self.ridge * identity[None],
            adjoint @ target[..., None],
        )[..., 0]
        step_count = torch.round(
            (query_frequency - self.reference_frequency) / self.frequency_step
        ).to(torch.long)
        reconstructed_query = self.reference_frequency + step_count * self.frequency_step
        if not torch.allclose(reconstructed_query, query_frequency):
            raise ValueError("query frequencies must lie on the observed frequency lattice")
        per_pixel_steps = step_count[:, None, None].expand(batch, height, width).reshape(-1)
        history = [sequence[:, index] for index in range(count)]
        selected = torch.empty_like(history[-1])
        for step in range(1, int(step_count.max()) + 1):
            prediction = sum(
                coefficients[:, lag] * history[-lag - 1] for lag in range(self.order)
            )
            history.append(prediction)
            mask = per_pixel_steps == step
            selected[mask] = prediction[mask]
        return selected.reshape(batch, height, width).to(fields.dtype)


class RealResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.scale * self.net(value)


class StableSpectralStateOperator(nn.Module):
    """Learn an invariant local recurrence and propagate the complex field state.

    The coefficient network sees magnitudes and adjacent cross-frequency phase
    increments, all invariant to a common phase rotation.  The recurrence acts on
    the complex observations themselves, making the complete operator equivariant.
    An l1 constraint on recurrence coefficients prevents exponentially growing
    characteristic roots.
    """

    def __init__(
        self,
        visible_frequencies: tuple[float, ...],
        environment_channels: int,
        *,
        order: int = 3,
        width: int = 64,
        blocks: int = 4,
        stability_radius: float = 1.0,
    ):
        super().__init__()
        if len(visible_frequencies) <= order:
            raise ValueError("the visible band must contain more samples than the recurrence order")
        gaps = torch.diff(torch.tensor(visible_frequencies))
        if not torch.allclose(gaps, gaps[0]):
            raise ValueError("the recurrence POC requires a uniform frequency lattice")
        self.visible_frequencies = visible_frequencies
        self.reference_frequency = float(visible_frequencies[-1])
        self.frequency_step = float(gaps[0])
        self.order = order
        self.stability_radius = stability_radius
        invariant_channels = len(visible_frequencies) + 2 * (len(visible_frequencies) - 1)
        invariant_channels += environment_channels + 2
        self.stem = nn.Sequential(
            nn.Conv2d(invariant_channels, width, 3, padding=1),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            RealResidualBlock(width, dilation=(1, 2, 4, 1)[index % 4])
            for index in range(blocks)
        )
        self.output = nn.Conv2d(width, 2 * order, 1)

    def recurrence_coefficients(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        magnitudes = fields.abs()
        adjacent = fields[:, 1:] * torch.conj(fields[:, :-1])
        adjacent = adjacent / adjacent.abs().clamp_min(1e-8)
        invariant = torch.cat(
            (
                magnitudes,
                adjacent.real,
                adjacent.imag,
                environment_and_coordinates[:, :-1],
            ),
            dim=1,
        )
        hidden = self.stem(invariant)
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.output(hidden)
        coefficients = torch.complex(raw[:, : self.order], raw[:, self.order :])
        coefficient_l1 = coefficients.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
        coefficients = coefficients * torch.clamp(
            self.stability_radius / coefficient_l1,
            max=1.0,
        )
        return coefficients

    def visible_reconstruction_loss(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Identify the one-step generator from the observed frequency band."""

        coefficients = self.recurrence_coefficients(fields, environment_and_coordinates)
        losses = []
        for index in range(self.order, fields.shape[1]):
            prediction = sum(
                coefficients[:, lag] * fields[:, index - lag - 1]
                for lag in range(self.order)
            )
            numerator = (prediction - fields[:, index]).abs().square().flatten(1).sum(1)
            denominator = fields[:, index].abs().square().flatten(1).sum(1).clamp_min(1e-8)
            losses.append(numerator / denominator)
        return torch.stack(losses, dim=1).mean()

    def forward(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequency: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = self.recurrence_coefficients(fields, environment_and_coordinates)
        step_count = torch.round(
            (query_frequency - self.reference_frequency) / self.frequency_step
        ).to(torch.long)
        reconstructed_query = self.reference_frequency + step_count * self.frequency_step
        if not torch.allclose(reconstructed_query, query_frequency):
            raise ValueError("query frequencies must lie on the observed frequency lattice")
        history = [fields[:, index] for index in range(fields.shape[1])]
        selected = torch.empty_like(history[-1])
        for step in range(1, int(step_count.max()) + 1):
            prediction = sum(
                coefficients[:, lag] * history[-lag - 1] for lag in range(self.order)
            )
            history.append(prediction)
            mask = step_count == step
            selected[mask] = prediction[mask]
        return selected


def phase_equivariance_error(
    model: nn.Module,
    fields: torch.Tensor,
    context: torch.Tensor,
    query: torch.Tensor,
    angle: float = 0.731,
) -> float:
    """Relative error under a common input/output phase rotation."""

    with torch.no_grad():
        reference = model(fields, context, query)
        rotation = torch.polar(
            torch.ones((), device=fields.device),
            torch.tensor(angle, device=fields.device),
        )
        rotated = model(rotation * fields, context, query)
        numerator = torch.linalg.vector_norm(rotated - rotation * reference)
        denominator = torch.linalg.vector_norm(reference).clamp_min(1e-8)
    return float((numerator / denominator).cpu())
