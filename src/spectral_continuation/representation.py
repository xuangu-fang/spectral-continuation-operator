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


class NormalizedRealResidualBlock(nn.Module):
    """Group-normalized block for heterogeneous invariant state features."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.conv1(value)
        update = F.silu(self.norm1(update))
        update = self.norm2(self.conv2(update))
        return value + self.scale * update


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

    def forward_queries(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequencies: tuple[float, ...],
    ) -> torch.Tensor:
        """Evaluate several lattice queries from one encoded recurrence state."""

        coefficients = self.recurrence_coefficients(fields, environment_and_coordinates)
        query = torch.tensor(
            query_frequencies, device=fields.device, dtype=fields.real.dtype
        )
        step_count = torch.round(
            (query - self.reference_frequency) / self.frequency_step
        ).to(torch.long)
        reconstructed = self.reference_frequency + step_count * self.frequency_step
        if not torch.allclose(reconstructed, query):
            raise ValueError("query frequencies must lie on the observed frequency lattice")
        history = [fields[:, index] for index in range(fields.shape[1])]
        predictions: dict[int, torch.Tensor] = {}
        requested = set(step_count.tolist())
        for step in range(1, int(step_count.max()) + 1):
            prediction = sum(
                coefficients[:, lag] * history[-lag - 1]
                for lag in range(self.order)
            )
            history.append(prediction)
            if step in requested:
                predictions[step] = prediction
        return torch.stack([predictions[int(step)] for step in step_count], dim=1)


class CarrierGaugedPoleOperator(nn.Module):
    """Direct continuous query through a carrier-gauged pole--residue state.

    A phase carrier is estimated analytically from the visible frequency band.
    The learned encoder only sees quantities invariant to a common complex phase
    rotation and emits a query-independent set of local poles and residues. The
    decoder evaluates those poles directly at the requested frequency, avoiding
    accumulated error from a one-Hz recurrence rollout.
    """

    def __init__(
        self,
        visible_frequencies: tuple[float, ...],
        environment_channels: int,
        *,
        poles: int = 4,
        width: int = 64,
        blocks: int = 4,
        max_growth_per_hz: float = 0.025,
        max_residual_angle_per_hz: float = math.pi,
        residue_scale: float = 1.0,
        state_backbone: str = "dilated",
        residue_mode: str = "predicted",
        residue_ridge: float = 1e-3,
        pole_growth: str = "signed",
        learn_carrier_correction: bool = False,
        max_carrier_correction_per_hz: float = 0.15,
        state_normalization: bool = False,
        pole_anchor: str = "reference_residual",
        carrier_estimator: str = "band_ols",
        max_curvature_per_hz2: float = 0.0,
        global_latent_channels: int = 0,
        hyper_carrier_correction: bool = False,
        max_hyper_slope_per_hz: float = 0.08,
        max_hyper_curvature_per_hz2: float = 0.005,
    ):
        super().__init__()
        if len(visible_frequencies) < 3:
            raise ValueError("carrier-gauged poles require at least three visible frequencies")
        frequencies = torch.tensor(visible_frequencies, dtype=torch.float32)
        gaps = torch.diff(frequencies)
        if not torch.allclose(gaps, gaps[0]):
            raise ValueError("carrier regression currently requires a uniform frequency lattice")
        if poles < 1:
            raise ValueError("poles must be positive")
        self.register_buffer("visible_frequency_tensor", frequencies)
        self.visible_frequencies = tuple(map(float, visible_frequencies))
        self.reference_frequency = float(visible_frequencies[-1])
        self.frequency_step = float(gaps[0])
        self.poles = int(poles)
        self.max_growth_per_hz = float(max_growth_per_hz)
        self.max_residual_angle_per_hz = float(max_residual_angle_per_hz)
        self.residue_scale = float(residue_scale)
        self.state_backbone = state_backbone
        self.residue_mode = residue_mode
        self.residue_ridge = float(residue_ridge)
        self.pole_growth = pole_growth
        self.learn_carrier_correction = bool(learn_carrier_correction)
        self.max_carrier_correction_per_hz = float(max_carrier_correction_per_hz)
        self.state_normalization = bool(state_normalization)
        self.pole_anchor = pole_anchor
        self.carrier_estimator = carrier_estimator
        self.max_curvature_per_hz2 = float(max_curvature_per_hz2)
        self.global_latent_channels = int(global_latent_channels)
        self.hyper_carrier_correction = bool(hyper_carrier_correction)
        self.max_hyper_slope_per_hz = float(max_hyper_slope_per_hz)
        self.max_hyper_curvature_per_hz2 = float(max_hyper_curvature_per_hz2)
        if self.hyper_carrier_correction and self.global_latent_channels <= 0:
            raise ValueError("hyper carrier correction requires a positive global latent")
        if self.global_latent_channels > 0 and state_backbone != "dilated":
            raise ValueError("global hyper conditioning currently requires the dilated backbone")
        if residue_mode not in {"predicted", "least_squares"}:
            raise ValueError(f"unknown residue mode: {residue_mode}")
        if pole_growth not in {"signed", "stable"}:
            raise ValueError(f"unknown pole growth mode: {pole_growth}")
        if pole_anchor not in {"reference_residual", "residue_sum"}:
            raise ValueError(f"unknown pole anchor: {pole_anchor}")
        if carrier_estimator not in {"band_ols", "endpoint_increment"}:
            raise ValueError(f"unknown carrier estimator: {carrier_estimator}")

        visible_count = len(visible_frequencies)
        invariant_channels = 3 * visible_count + 2 * (visible_count - 1)
        invariant_channels += environment_channels + 2
        if self.global_latent_channels > 0:
            latent = self.global_latent_channels
            self.global_encoder = nn.Sequential(
                nn.Conv2d(invariant_channels, 32, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv2d(64, latent, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.global_film = nn.Linear(latent, 2 * width * (blocks + 1))
            nn.init.zeros_(self.global_film.weight)
            nn.init.zeros_(self.global_film.bias)
            self.hyper_carrier_head = (
                nn.Linear(latent, 12) if self.hyper_carrier_correction else None
            )
            if self.hyper_carrier_head is not None:
                nn.init.zeros_(self.hyper_carrier_head.weight)
                nn.init.zeros_(self.hyper_carrier_head.bias)
        else:
            self.global_encoder = None
            self.global_film = None
            self.hyper_carrier_head = None
        if state_backbone == "dilated":
            groups = min(8, width)
            while width % groups:
                groups -= 1
            self.stem = nn.Sequential(
                nn.Conv2d(invariant_channels, width, 3, padding=1),
                *([nn.GroupNorm(groups, width)] if state_normalization else []),
                nn.SiLU(),
            )
            block_type = (
                NormalizedRealResidualBlock if state_normalization else RealResidualBlock
            )
            self.blocks = nn.ModuleList(
                block_type(width, dilation=(1, 2, 4, 1)[index % 4])
                for index in range(blocks)
            )
            self.residue_head = (
                nn.Conv2d(width, 2 * poles, 1)
                if residue_mode == "predicted"
                else None
            )
            self.pole_head = nn.Conv2d(width, 2 * poles, 1)
            self.curvature_head = (
                nn.Conv2d(width, poles, 1) if self.max_curvature_per_hz2 > 0 else None
            )
            self.carrier_head = (
                nn.Conv2d(width, 1, 1) if learn_carrier_correction else None
            )
            if self.residue_head is not None:
                nn.init.normal_(self.residue_head.weight, std=1e-3)
                nn.init.zeros_(self.residue_head.bias)
            nn.init.zeros_(self.pole_head.weight)
            nn.init.zeros_(self.pole_head.bias)
            if self.curvature_head is not None:
                nn.init.zeros_(self.curvature_head.weight)
                nn.init.zeros_(self.curvature_head.bias)
            if self.carrier_head is not None:
                nn.init.zeros_(self.carrier_head.weight)
                nn.init.zeros_(self.carrier_head.bias)
            angle_bias = self.pole_head.bias[poles:]
        elif state_backbone == "unet":
            state_channels = (4 if residue_mode == "predicted" else 2) * poles
            state_channels += poles if self.max_curvature_per_hz2 > 0 else 0
            state_channels += int(learn_carrier_correction)
            self.backbone = QueryUNet(
                invariant_channels, width=width, output_channels=state_channels
            )
            nn.init.normal_(self.backbone.out.weight, std=1e-3)
            nn.init.zeros_(self.backbone.out.bias)
            angle_start = (3 if residue_mode == "predicted" else 1) * poles
            angle_bias = self.backbone.out.bias[angle_start : angle_start + poles]
        else:
            raise ValueError(f"unknown state backbone: {state_backbone}")

        # Start close to analytic carrier extrapolation while breaking the pole
        # permutation symmetry so residual frequencies can specialize early.
        initial_angles = torch.linspace(-0.65, 0.65, poles).clamp(-0.95, 0.95)
        with torch.no_grad():
            angle_bias.copy_(torch.atanh(initial_angles))

    def carrier_slope(self, fields: torch.Tensor) -> torch.Tensor:
        """Estimate the local phase generator from principal increments.

        ``band_ols`` averages the complete visible band. ``endpoint_increment``
        uses the last secant as a one-sided estimate at the continuation
        boundary, which is preferable when phase has appreciable curvature.
        """

        unit = fields / fields.abs().clamp_min(1e-8)
        increments = torch.angle(unit[:, 1:] * torch.conj(unit[:, :-1]))
        if self.carrier_estimator == "endpoint_increment":
            return increments[:, -1] / self.frequency_step
        relative_phase = torch.cat(
            (torch.zeros_like(increments[:, :1]), torch.cumsum(increments, dim=1)),
            dim=1,
        )
        frequencies = self.visible_frequency_tensor.to(fields.device, fields.real.dtype)
        offsets = frequencies - frequencies.mean()
        centered_phase = relative_phase - relative_phase.mean(dim=1, keepdim=True)
        return torch.sum(
            centered_phase * offsets.view(1, -1, 1, 1), dim=1
        ) / offsets.square().sum().clamp_min(1e-8)

    def carrier(
        self,
        fields: torch.Tensor,
        query_frequency: torch.Tensor,
        slope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if slope is None:
            slope = self.carrier_slope(fields)
        reference = fields[:, -1] / fields[:, -1].abs().clamp_min(1e-8)
        delta = query_frequency.to(fields.real.dtype) - self.reference_frequency
        rotation = torch.polar(
            torch.ones_like(slope), slope * delta.view(-1, 1, 1)
        )
        return reference * rotation

    def spectral_state(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.is_complex(fields):
            raise TypeError("fields must be a complex tensor")
        if fields.shape[1] != len(self.visible_frequencies):
            raise ValueError("visible field count does not match the configured frequency set")
        slope = self.carrier_slope(fields)
        frequency_delta = (
            self.visible_frequency_tensor.to(fields.device) - self.reference_frequency
        )
        reference_unit = fields[:, -1] / fields[:, -1].abs().clamp_min(1e-8)
        visible_carrier = reference_unit[:, None] * torch.polar(
            torch.ones_like(fields.real),
            slope[:, None] * frequency_delta.view(1, -1, 1, 1),
        )
        gauged = fields * torch.conj(visible_carrier)
        adjacent = fields[:, 1:] * torch.conj(fields[:, :-1])
        adjacent = adjacent / adjacent.abs().clamp_min(1e-8)
        invariant = torch.cat(
            (
                gauged.real,
                gauged.imag,
                fields.abs(),
                adjacent.real,
                adjacent.imag,
                environment_and_coordinates[:, :-1],
            ),
            dim=1,
        )
        global_state = (
            self.global_encoder(invariant).flatten(1)
            if self.global_encoder is not None
            else None
        )
        film = None
        if global_state is not None and self.global_film is not None:
            film = self.global_film(global_state).view(
                fields.shape[0], 2, len(self.blocks) + 1, -1, 1, 1
            )

        def modulate(value: torch.Tensor, index: int) -> torch.Tensor:
            if film is None:
                return value
            scale = 0.25 * torch.tanh(film[:, 0, index])
            shift = 0.25 * film[:, 1, index]
            return value * (1.0 + scale) + shift

        if self.state_backbone == "dilated":
            hidden = modulate(self.stem(invariant), 0)
            for index, block in enumerate(self.blocks):
                hidden = modulate(block(hidden), index + 1)
            pole_raw = self.pole_head(hidden)
            carrier_raw = (
                self.carrier_head(hidden)[:, 0]
                if self.carrier_head is not None
                else None
            )
            residue_raw = (
                self.residue_head(hidden) if self.residue_head is not None else None
            )
        else:
            raw = self.backbone(invariant)
            state_channel_count = (
                (4 if self.residue_mode == "predicted" else 2) * self.poles
            )
            carrier_raw = (
                raw[:, state_channel_count]
                if self.learn_carrier_correction
                else None
            )
            raw = raw[:, :state_channel_count]
            if self.residue_mode == "predicted":
                residue_raw = raw[:, : 2 * self.poles]
                pole_raw = raw[:, 2 * self.poles : 4 * self.poles]
                curvature_raw = (
                    raw[:, 4 * self.poles : 5 * self.poles]
                    if self.max_curvature_per_hz2 > 0
                    else None
                )
            else:
                residue_raw = None
                pole_raw = raw[:, : 2 * self.poles]
                curvature_raw = (
                    raw[:, 2 * self.poles : 3 * self.poles]
                    if self.max_curvature_per_hz2 > 0
                    else None
                )
        if self.pole_growth == "stable":
            # Negative real parts make every learned mode non-growing toward
            # higher frequencies. The offset initializes close to the unit circle.
            growth = -self.max_growth_per_hz * torch.sigmoid(
                pole_raw[:, : self.poles] - 3.0
            )
        else:
            growth = self.max_growth_per_hz * torch.tanh(
                pole_raw[:, : self.poles]
            )
        angle = self.max_residual_angle_per_hz * torch.tanh(pole_raw[:, self.poles :])
        pole = torch.complex(growth, angle)
        if self.state_backbone == "dilated":
            curvature_raw = (
                self.curvature_head(hidden) if self.curvature_head is not None else None
            )
        curvature = (
            self.max_curvature_per_hz2 * torch.tanh(curvature_raw)
            if curvature_raw is not None
            else torch.zeros_like(growth)
        )
        if carrier_raw is not None:
            slope = slope + self.max_carrier_correction_per_hz * torch.tanh(
                carrier_raw
            )
        if self.hyper_carrier_head is not None and global_state is not None:
            coefficients = self.hyper_carrier_head(global_state).view(
                fields.shape[0], 2, 6, 1, 1
            )
            yy = environment_and_coordinates[:, -3]
            xx = environment_and_coordinates[:, -2]
            basis = torch.stack(
                (
                    torch.ones_like(xx),
                    xx,
                    yy,
                    xx.square(),
                    xx * yy,
                    yy.square(),
                ),
                dim=1,
            )
            hyper_slope = torch.sum(coefficients[:, 0] * basis, dim=1)
            slope = slope + self.max_hyper_slope_per_hz * torch.tanh(hyper_slope)
            hyper_curvature = torch.sum(coefficients[:, 1] * basis, dim=1)
            curvature = curvature + self.max_hyper_curvature_per_hz2 * torch.tanh(
                hyper_curvature[:, None]
            )
        if residue_raw is not None:
            residue = torch.complex(
                torch.tanh(residue_raw[:, : self.poles]),
                torch.tanh(residue_raw[:, self.poles :]),
            )
            local_scale = fields.abs().mean(dim=1, keepdim=True)
            residue = self.residue_scale * local_scale * residue / math.sqrt(self.poles)
            if self.pole_anchor == "residue_sum":
                anchor = fields[:, -1].abs()[:, None]
                residue = residue + (anchor - residue.sum(dim=1, keepdim=True)) / self.poles
        else:
            residue = self.solve_residues(gauged, pole)
        return fields[:, -1].abs(), residue, pole, slope, curvature

    def solve_residues(
        self, gauged_fields: torch.Tensor, pole: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable ridge fit of residues to the complete visible band."""

        batch, _, height, width = gauged_fields.shape
        delta = (
            self.visible_frequency_tensor.to(gauged_fields.device)
            - self.reference_frequency
        )
        design = torch.exp(
            pole[:, None] * delta.view(1, -1, 1, 1, 1)
        )
        design = design.permute(0, 3, 4, 1, 2).reshape(-1, len(delta), self.poles)
        target = gauged_fields.permute(0, 2, 3, 1).reshape(-1, len(delta), 1)
        adjoint = design.mH
        gram = adjoint @ design
        identity = torch.eye(
            self.poles, device=gram.device, dtype=gram.dtype
        ).expand(gram.shape[0], -1, -1)
        right_hand_side = adjoint @ target
        residue = torch.linalg.solve(
            gram + self.residue_ridge * identity, right_hand_side
        )[..., 0]
        return residue.reshape(batch, height, width, self.poles).permute(0, 3, 1, 2)

    def decode(
        self,
        reference_amplitude: torch.Tensor,
        residue: torch.Tensor,
        pole: torch.Tensor,
        fields: torch.Tensor,
        query_frequency: torch.Tensor,
        slope: torch.Tensor,
        curvature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = (
            query_frequency.to(fields.real.dtype) - self.reference_frequency
        ).view(-1, 1, 1, 1)
        exponent = pole * delta
        if curvature is not None:
            exponent = exponent + torch.complex(
                torch.zeros_like(exponent.real), 0.5 * curvature * delta.square()
            )
        exponent = torch.complex(exponent.real.clamp(-6.0, 6.0), exponent.imag)
        if self.residue_mode == "predicted" and self.pole_anchor == "reference_residual":
            basis = torch.exp(exponent) - 1.0
            gauged = reference_amplitude + torch.sum(residue * basis, dim=1)
        else:
            basis = torch.exp(exponent)
            gauged = torch.sum(residue * basis, dim=1)
        return self.carrier(fields, query_frequency, slope) * gauged

    def forward(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequency: torch.Tensor,
    ) -> torch.Tensor:
        reference_amplitude, residue, pole, slope, curvature = self.spectral_state(
            fields, environment_and_coordinates
        )
        return self.decode(
            reference_amplitude,
            residue,
            pole,
            fields,
            query_frequency,
            slope,
            curvature,
        )

    def forward_queries(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequencies: tuple[float, ...],
    ) -> torch.Tensor:
        """Evaluate arbitrary real queries from one pole--residue state."""

        state = self.spectral_state(fields, environment_and_coordinates)
        outputs = []
        for frequency in query_frequencies:
            query = torch.full(
                (fields.shape[0],),
                frequency,
                device=fields.device,
                dtype=fields.real.dtype,
            )
            outputs.append(self.decode(*state[:3], fields, query, state[3], state[4]))
        return torch.stack(outputs, dim=1)

    def visible_reconstruction_loss(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Fit one query-independent state to every non-anchor visible field."""

        reference_amplitude, residue, pole, slope, curvature = self.spectral_state(
            fields, environment_and_coordinates
        )
        losses = []
        for index, frequency in enumerate(self.visible_frequencies[:-1]):
            query = torch.full(
                (fields.shape[0],), frequency, device=fields.device, dtype=fields.real.dtype
            )
            prediction = self.decode(
                reference_amplitude, residue, pole, fields, query, slope, curvature
            )
            numerator = (prediction - fields[:, index]).abs().square().flatten(1).sum(1)
            denominator = fields[:, index].abs().square().flatten(1).sum(1).clamp_min(1e-8)
            losses.append(numerator / denominator)
        return torch.stack(losses, dim=1).mean()


class EquivariantSpectralGenerator(nn.Module):
    """Phase-equivariant multi-component affine frequency generator.

    A complex encoder decomposes the visible band into coherent endpoint
    components.  A separate invariant encoder predicts a stable affine generator
    for each component.  The exact endpoint projection makes the components sum
    to the last visible field, while the affine imaginary generator captures
    first-order dispersion curvature without a rollout or auxiliary loss.
    """

    def __init__(
        self,
        visible_frequencies: tuple[float, ...],
        environment_channels: int,
        *,
        components: int = 8,
        complex_width: int = 24,
        state_width: int = 64,
        blocks: int = 4,
        max_growth_per_hz: float = 0.025,
        max_angle_per_hz: float = math.pi,
        max_curvature_per_hz2: float = 0.01,
        carrier_mode: str = "endpoint",
        carrier_kernel: int = 5,
        parameter_smoothing: int = 1,
        stable_growth: bool = True,
    ):
        super().__init__()
        if len(visible_frequencies) < 3:
            raise ValueError("the component generator requires at least three fields")
        frequencies = torch.tensor(visible_frequencies, dtype=torch.float32)
        gaps = torch.diff(frequencies)
        if not torch.allclose(gaps, gaps[0]):
            raise ValueError("the component generator currently requires a uniform lattice")
        if components < 1:
            raise ValueError("components must be positive")
        if carrier_mode not in {"none", "endpoint", "coherent_endpoint"}:
            raise ValueError(f"unknown carrier mode: {carrier_mode}")
        if carrier_kernel % 2 != 1 or carrier_kernel < 1:
            raise ValueError("carrier kernel must be a positive odd integer")
        if parameter_smoothing % 2 != 1 or parameter_smoothing < 1:
            raise ValueError("parameter smoothing must be a positive odd integer")
        self.register_buffer("visible_frequency_tensor", frequencies)
        self.visible_frequencies = tuple(map(float, visible_frequencies))
        self.reference_frequency = float(visible_frequencies[-1])
        self.frequency_step = float(gaps[0])
        self.components = int(components)
        self.max_growth_per_hz = float(max_growth_per_hz)
        self.max_angle_per_hz = float(max_angle_per_hz)
        self.max_curvature_per_hz2 = float(max_curvature_per_hz2)
        self.carrier_mode = carrier_mode
        self.carrier_kernel = int(carrier_kernel)
        self.parameter_smoothing = int(parameter_smoothing)
        self.stable_growth = bool(stable_growth)

        visible_count = len(visible_frequencies)
        self.complex_stem = ComplexConv2d(visible_count, complex_width, 3)
        self.complex_activation = ComplexModReLU(complex_width)
        self.complex_blocks = nn.ModuleList(
            ComplexResidualBlock(complex_width, dilation=(1, 2, 4, 1)[index % 4])
            for index in range(blocks)
        )
        self.context_gate = nn.Sequential(
            nn.Conv2d(environment_channels + 2, complex_width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(complex_width, complex_width, 1),
        )
        self.component_head = ComplexConv2d(complex_width, components, 1)
        nn.init.zeros_(self.component_head.weight_real)
        nn.init.zeros_(self.component_head.weight_imag)

        # |u|, confidence-weighted adjacent correlations, correlations with the
        # endpoint, and phase-invariant environment/coordinate channels.
        invariant_channels = visible_count + 2 * (visible_count - 1) + 2 * visible_count
        invariant_channels += environment_channels + 2
        groups = min(8, state_width)
        while state_width % groups:
            groups -= 1
        self.generator_stem = nn.Sequential(
            nn.Conv2d(invariant_channels, state_width, 3, padding=1),
            nn.GroupNorm(groups, state_width),
            nn.SiLU(),
        )
        self.generator_blocks = nn.ModuleList(
            NormalizedRealResidualBlock(
                state_width, dilation=(1, 2, 4, 1)[index % 4]
            )
            for index in range(blocks)
        )
        self.generator_head = nn.Conv2d(state_width, 3 * components, 1)
        nn.init.zeros_(self.generator_head.weight)
        nn.init.zeros_(self.generator_head.bias)
        # Break component permutation symmetry while starting around the
        # analytic carrier.  Residual phase rates remain well inside the bound.
        initial = torch.linspace(-0.30, 0.30, components).clamp(-0.95, 0.95)
        with torch.no_grad():
            self.generator_head.bias[components : 2 * components].copy_(
                torch.atanh(initial)
            )

    @staticmethod
    def _confidence_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cross = left * right.conj()
        denominator = left.abs().square() + right.abs().square() + 1e-8
        return 2.0 * cross / denominator

    def carrier_slope(self, fields: torch.Tensor) -> torch.Tensor:
        if self.carrier_mode == "none":
            return torch.zeros_like(fields[:, 0].real)
        cross = fields[:, -1] * fields[:, -2].conj()
        if self.carrier_mode == "coherent_endpoint" and self.carrier_kernel > 1:
            padding = self.carrier_kernel // 2
            real = F.avg_pool2d(
                F.pad(cross.real[:, None], (padding,) * 4, mode="replicate"),
                self.carrier_kernel,
                stride=1,
            )[:, 0]
            imag = F.avg_pool2d(
                F.pad(cross.imag[:, None], (padding,) * 4, mode="replicate"),
                self.carrier_kernel,
                stride=1,
            )[:, 0]
            cross = torch.complex(real, imag)
        return torch.angle(cross) / self.frequency_step

    def invariant_features(
        self, fields: torch.Tensor, environment_and_coordinates: torch.Tensor
    ) -> torch.Tensor:
        adjacent = self._confidence_correlation(fields[:, 1:], fields[:, :-1])
        endpoint = fields[:, -1:]
        relative = self._confidence_correlation(fields, endpoint)
        return torch.cat(
            (
                fields.abs(),
                adjacent.real,
                adjacent.imag,
                relative.real,
                relative.imag,
                environment_and_coordinates[:, :-1],
            ),
            dim=1,
        )

    def spectral_state(
        self, fields: torch.Tensor, environment_and_coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.is_complex(fields):
            raise TypeError("fields must be a complex tensor")
        hidden = self.complex_activation(self.complex_stem(fields))
        gate = 2.0 * torch.sigmoid(self.context_gate(environment_and_coordinates[:, :-1]))
        hidden = hidden * gate
        for block in self.complex_blocks:
            hidden = block(hidden)
        component = self.component_head(hidden)
        # Exact endpoint anchor without selecting a potentially singular local
        # phase gauge.  The correction is itself complex phase equivariant.
        component = component + (
            fields[:, -1:] - component.sum(dim=1, keepdim=True)
        ) / self.components

        invariant = self.invariant_features(fields, environment_and_coordinates)
        generator_hidden = self.generator_stem(invariant)
        for block in self.generator_blocks:
            generator_hidden = block(generator_hidden)
        raw = self.generator_head(generator_hidden)
        if self.parameter_smoothing > 1:
            padding = self.parameter_smoothing // 2
            raw = F.avg_pool2d(
                F.pad(raw, (padding,) * 4, mode="replicate"),
                self.parameter_smoothing,
                stride=1,
            )
        if self.stable_growth:
            growth = -self.max_growth_per_hz * torch.sigmoid(
                raw[:, : self.components] - 3.0
            )
        else:
            growth = self.max_growth_per_hz * torch.tanh(raw[:, : self.components])
        residual_angle = self.max_angle_per_hz * torch.tanh(
            raw[:, self.components : 2 * self.components]
        )
        curvature = self.max_curvature_per_hz2 * torch.tanh(
            raw[:, 2 * self.components :]
        )
        return component, growth, residual_angle, curvature

    def decode(
        self,
        fields: torch.Tensor,
        query_frequency: torch.Tensor,
        component: torch.Tensor,
        growth: torch.Tensor,
        residual_angle: torch.Tensor,
        curvature: torch.Tensor,
    ) -> torch.Tensor:
        delta = (
            query_frequency.to(fields.real.dtype) - self.reference_frequency
        ).view(-1, 1, 1, 1)
        slope = self.carrier_slope(fields)[:, None]
        angle = slope + residual_angle
        exponent = torch.complex(
            (growth * delta).clamp(-6.0, 6.0),
            angle * delta + 0.5 * curvature * delta.square(),
        )
        return torch.sum(component * torch.exp(exponent), dim=1)

    def forward(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequency: torch.Tensor,
    ) -> torch.Tensor:
        state = self.spectral_state(fields, environment_and_coordinates)
        return self.decode(fields, query_frequency, *state)

    def forward_queries(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequencies: tuple[float, ...],
    ) -> torch.Tensor:
        state = self.spectral_state(fields, environment_and_coordinates)
        outputs = []
        for frequency in query_frequencies:
            query = torch.full(
                (fields.shape[0],),
                frequency,
                device=fields.device,
                dtype=fields.real.dtype,
            )
            outputs.append(self.decode(fields, query, *state))
        return torch.stack(outputs, dim=1)


class PhaseEquivariantNeuralODE(nn.Module):
    """State-dependent carrier integrator along the frequency axis.

    The vector field is represented in the instantaneous carrier gauge, so the
    real U-Net only sees phase-invariant quantities.  A one-Hz update is reused
    recurrently, while fractional queries use the same vector field with a
    shorter integration step.  Unlike the fixed spectral recurrence, the update
    depends on the evolving complex field, environment, coordinates, and current
    frequency.
    """

    def __init__(
        self,
        visible_frequencies: tuple[float, ...],
        environment_channels: int,
        *,
        width: int = 8,
        frequency_scale: float = 200.0,
        max_update_rate: float = 1.0,
        carrier_mode: str = "endpoint",
        carrier_kernel: int = 5,
    ):
        super().__init__()
        if len(visible_frequencies) < 2:
            raise ValueError("the neural ODE requires at least two visible fields")
        frequencies = torch.tensor(visible_frequencies, dtype=torch.float32)
        gaps = torch.diff(frequencies)
        if not torch.allclose(gaps, gaps[0]):
            raise ValueError("the neural ODE currently requires a uniform visible lattice")
        if carrier_mode not in {"endpoint", "coherent_endpoint"}:
            raise ValueError(f"unknown carrier mode: {carrier_mode}")
        if carrier_kernel % 2 != 1 or carrier_kernel < 1:
            raise ValueError("carrier kernel must be a positive odd integer")
        self.register_buffer("visible_frequency_tensor", frequencies)
        self.visible_frequencies = tuple(map(float, visible_frequencies))
        self.reference_frequency = float(visible_frequencies[-1])
        self.frequency_step = float(gaps[-1])
        self.frequency_scale = float(frequency_scale)
        self.max_update_rate = float(max_update_rate)
        self.carrier_mode = carrier_mode
        self.carrier_kernel = int(carrier_kernel)
        visible_count = len(visible_frequencies)
        # Static band: magnitudes and confidence-weighted phase increments.
        static_channels = visible_count + 2 * (visible_count - 1)
        # Dynamic state: previous/current magnitudes, their correlation, and
        # correlation to the original endpoint, plus current f and step size.
        dynamic_channels = 2 + 2 + 2 + 2
        input_channels = static_channels + dynamic_channels + environment_channels + 2
        self.vector_field = QueryUNet(input_channels, width=width, output_channels=2)
        nn.init.zeros_(self.vector_field.out.weight)
        nn.init.zeros_(self.vector_field.out.bias)

    @staticmethod
    def _confidence_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cross = left * right.conj()
        return 2.0 * cross / (left.abs().square() + right.abs().square() + 1e-8)

    def static_features(self, fields: torch.Tensor) -> torch.Tensor:
        adjacent = self._confidence_correlation(fields[:, 1:], fields[:, :-1])
        return torch.cat((fields.abs(), adjacent.real, adjacent.imag), dim=1)

    def slope(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        cross = current * previous.conj()
        if self.carrier_mode == "coherent_endpoint" and self.carrier_kernel > 1:
            padding = self.carrier_kernel // 2
            real = F.avg_pool2d(
                F.pad(cross.real[:, None], (padding,) * 4, mode="replicate"),
                self.carrier_kernel,
                stride=1,
            )[:, 0]
            imag = F.avg_pool2d(
                F.pad(cross.imag[:, None], (padding,) * 4, mode="replicate"),
                self.carrier_kernel,
                stride=1,
            )[:, 0]
            cross = torch.complex(real, imag)
        return torch.angle(cross) / self.frequency_step

    def step(
        self,
        fields: torch.Tensor,
        static: torch.Tensor,
        previous: torch.Tensor,
        current: torch.Tensor,
        context: torch.Tensor,
        current_frequency: float,
        step_size: float,
    ) -> torch.Tensor:
        slope = self.slope(current, previous)
        rotation = torch.polar(torch.ones_like(slope), slope * step_size)
        carrier_unit = current / current.abs().clamp_min(1e-8) * rotation
        dynamic_cross = self._confidence_correlation(current, previous)
        endpoint_cross = self._confidence_correlation(current, fields[:, -1])
        batch, _, height, width = fields.shape
        frequency_map = torch.full(
            (batch, 1, height, width),
            current_frequency / self.frequency_scale,
            device=fields.device,
            dtype=fields.real.dtype,
        )
        step_map = torch.full_like(frequency_map, step_size / self.frequency_step)
        invariant = torch.cat(
            (
                static,
                previous.abs()[:, None],
                current.abs()[:, None],
                dynamic_cross.real[:, None],
                dynamic_cross.imag[:, None],
                endpoint_cross.real[:, None],
                endpoint_cross.imag[:, None],
                context[:, :-1],
                frequency_map,
                step_map,
            ),
            dim=1,
        )
        raw = torch.tanh(self.vector_field(invariant))
        local_scale = fields.abs().mean(dim=1)
        correction = torch.complex(raw[:, 0], raw[:, 1])
        coefficient = current.abs() + (
            step_size * self.max_update_rate * local_scale * correction
        )
        return carrier_unit * coefficient

    def forward_queries(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequencies: tuple[float, ...],
    ) -> torch.Tensor:
        if min(query_frequencies) < self.reference_frequency:
            raise ValueError("the neural ODE only continues toward higher frequencies")
        static = self.static_features(fields)
        previous = fields[:, -2]
        current = fields[:, -1]
        current_frequency = self.reference_frequency
        integer_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
            round(current_frequency): (previous, current)
        }
        maximum_integer = math.floor(max(query_frequencies))
        while current_frequency < maximum_integer:
            next_field = self.step(
                fields,
                static,
                previous,
                current,
                environment_and_coordinates,
                current_frequency,
                self.frequency_step,
            )
            previous, current = current, next_field
            current_frequency += self.frequency_step
            integer_states[round(current_frequency)] = (previous, current)
        outputs = []
        for query_frequency in query_frequencies:
            lower = math.floor(query_frequency)
            fractional = query_frequency - lower
            state_previous, state_current = integer_states[lower]
            if fractional < 1e-7:
                outputs.append(state_current)
            else:
                outputs.append(
                    self.step(
                        fields,
                        static,
                        state_previous,
                        state_current,
                        environment_and_coordinates,
                        float(lower),
                        fractional,
                    )
                )
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        fields: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
        query_frequency: torch.Tensor,
    ) -> torch.Tensor:
        unique = torch.unique(query_frequency.detach().cpu())
        if len(unique) == 1:
            return self.forward_queries(
                fields, environment_and_coordinates, (float(unique[0]),)
            )[:, 0]
        outputs = []
        for index, frequency in enumerate(query_frequency):
            outputs.append(
                self.forward_queries(
                    fields[index : index + 1],
                    environment_and_coordinates[index : index + 1],
                    (float(frequency),),
                )[0, 0]
            )
        return torch.stack(outputs)

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
