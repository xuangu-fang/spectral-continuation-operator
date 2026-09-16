"""Invariant amortized reconstruction of sparse spectral calibration slices."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import QueryUNet


def confidence_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    cross = left * right.conj()
    denominator = left.abs().square() + right.abs().square() + 1e-8
    return 2.0 * cross / denominator


def normalized_sensor_splat(
    evidence: torch.Tensor,
    mask: torch.Tensor,
    token_grid: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate an unordered sparse set into normalized local token cells."""

    pooled_mask = F.adaptive_avg_pool2d(mask, token_grid)
    pooled_evidence = F.adaptive_avg_pool2d(evidence, token_grid)
    pooled_evidence = pooled_evidence / pooled_mask.clamp_min(1e-6)
    occupied = (pooled_mask > 0).to(evidence.dtype)
    pooled_evidence = pooled_evidence * occupied
    shape = evidence.shape[-2:]
    return (
        F.interpolate(pooled_evidence, size=shape, mode="nearest"),
        F.interpolate(occupied, size=shape, mode="nearest"),
    )


class InvariantSpectralCalibrator(nn.Module):
    """Correct dense calibration priors from sparse complex observations.

    Every feature is invariant to a common phase rotation. The network predicts
    amplitude and phase corrections which multiply the frozen complex prior, so
    the resulting calibration fields remain globally phase equivariant.
    """

    def __init__(
        self,
        visible_count: int,
        environment_channels: int,
        *,
        calibration_count: int = 2,
        width: int = 24,
        sensor_token_grid: tuple[int, int] = (4, 8),
        max_log_amplitude_correction: float = 1.5,
        max_phase_correction: float = math.pi,
        max_slope_correction_per_hz: float = math.pi,
        calibration_frequency_step: float = 1.0,
    ):
        super().__init__()
        self.visible_count = int(visible_count)
        self.calibration_count = int(calibration_count)
        self.sensor_token_grid = tuple(map(int, sensor_token_grid))
        self.max_log_amplitude_correction = float(max_log_amplitude_correction)
        self.max_phase_correction = float(max_phase_correction)
        self.max_slope_correction_per_hz = float(max_slope_correction_per_hz)
        self.calibration_frequency_step = float(calibration_frequency_step)
        if calibration_count != 2:
            raise ValueError("the coupled spectral-state decoder requires two slices")
        # Visible magnitudes and adjacent normalized correlations.
        channels = visible_count + 2 * (visible_count - 1)
        # Prior magnitudes and prior-to-visible-endpoint correlations.
        channels += calibration_count + 2 * calibration_count
        # Raw sparse evidence: complex correlation, log-amplitude ratio, mask.
        evidence_channels = 3 * calibration_count
        channels += evidence_channels + 1
        # Locally set-aggregated evidence and occupancy.
        channels += evidence_channels + 1
        # Globally averaged instance-calibration evidence.
        channels += evidence_channels
        # Active medium channels and spatial coordinates.
        channels += environment_channels + 2
        # Two log-amplitude corrections, one reference-phase correction and one
        # shared spectral-slope correction. This has the same output size as two
        # independent complex corrections but cannot invent inconsistent phases.
        self.backbone = QueryUNet(channels, width=width, output_channels=4)
        nn.init.zeros_(self.backbone.out.weight)
        nn.init.zeros_(self.backbone.out.bias)

    def invariant_features(
        self,
        visible_fields: torch.Tensor,
        calibration_prior: torch.Tensor,
        sparse_observations: torch.Tensor,
        sensor_mask: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        adjacent = confidence_correlation(visible_fields[:, 1:], visible_fields[:, :-1])
        endpoint = visible_fields[:, -1:]
        prior_relative = confidence_correlation(calibration_prior, endpoint)
        observed_relative = confidence_correlation(sparse_observations, calibration_prior)
        log_amplitude_ratio = torch.log(
            (sparse_observations.abs() + 1e-6) / (calibration_prior.abs() + 1e-6)
        ).clamp(-4.0, 4.0)
        observed_relative = observed_relative * sensor_mask
        log_amplitude_ratio = log_amplitude_ratio * sensor_mask
        evidence = torch.cat(
            (
                observed_relative.real,
                observed_relative.imag,
                log_amplitude_ratio,
            ),
            dim=1,
        )
        pooled_evidence, occupied = normalized_sensor_splat(
            evidence, sensor_mask[:, :1], self.sensor_token_grid
        )
        sensor_count = sensor_mask[:, :1].sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        global_evidence = evidence.sum(dim=(2, 3), keepdim=True) / sensor_count
        global_evidence = global_evidence.expand(-1, -1, *evidence.shape[-2:])
        return torch.cat(
            (
                visible_fields.abs(),
                adjacent.real,
                adjacent.imag,
                calibration_prior.abs(),
                prior_relative.real,
                prior_relative.imag,
                evidence,
                sensor_mask[:, :1],
                pooled_evidence,
                occupied,
                global_evidence,
                environment_and_coordinates[:, :-1],
            ),
            dim=1,
        )

    def forward(
        self,
        visible_fields: torch.Tensor,
        calibration_prior: torch.Tensor,
        sparse_observations: torch.Tensor,
        sensor_mask: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        features = self.invariant_features(
            visible_fields,
            calibration_prior,
            sparse_observations,
            sensor_mask,
            environment_and_coordinates,
        )
        raw = self.backbone(features)
        log_amplitude = self.max_log_amplitude_correction * torch.tanh(raw[:, :2])
        reference_phase_delta = self.max_phase_correction * torch.tanh(raw[:, 2])
        slope_delta = self.max_slope_correction_per_hz * torch.tanh(raw[:, 3])
        base_slope = (
            torch.angle(calibration_prior[:, 1] * calibration_prior[:, 0].conj())
            / self.calibration_frequency_step
        )
        slope = base_slope + slope_delta
        reference_phase = torch.angle(calibration_prior[:, 1]) + reference_phase_delta
        phase = torch.stack(
            (
                reference_phase - self.calibration_frequency_step * slope,
                reference_phase,
            ),
            dim=1,
        )
        amplitude = calibration_prior.abs() * torch.exp(log_amplitude)
        return amplitude * torch.polar(torch.ones_like(phase), phase)


def enforce_sensor_consistency(
    prediction: torch.Tensor,
    observations: torch.Tensor,
    sensor_mask: torch.Tensor,
) -> torch.Tensor:
    return prediction * (1.0 - sensor_mask) + observations * sensor_mask


def endpoint_carrier_query(
    calibration_fields: torch.Tensor,
    calibration_frequencies: tuple[float, float],
    query_frequency: float | torch.Tensor,
) -> torch.Tensor:
    """Query a constant-envelope local carrier from two adjacent slices."""

    if calibration_fields.shape[1] != 2:
        raise ValueError("endpoint carrier requires exactly two calibration fields")
    frequency_step = calibration_frequencies[1] - calibration_frequencies[0]
    if frequency_step <= 0:
        raise ValueError("calibration frequencies must be strictly increasing")
    slope = torch.angle(calibration_fields[:, 1] * calibration_fields[:, 0].conj()) / frequency_step
    if not torch.is_tensor(query_frequency):
        query_frequency = torch.full(
            (calibration_fields.shape[0],),
            float(query_frequency),
            device=calibration_fields.device,
            dtype=calibration_fields.real.dtype,
        )
    delta = query_frequency - calibration_frequencies[1]
    reference = calibration_fields[:, 1]
    return reference.abs() * torch.polar(
        torch.ones_like(reference.real),
        torch.angle(reference) + delta[:, None, None] * slope,
    )


def fit_stable_ar1(fields: torch.Tensor, ridge: float = 0.01) -> torch.Tensor:
    """Fit a non-growing complex one-step pole along the frequency axis."""

    if fields.shape[1] < 2:
        raise ValueError("AR(1) fitting requires at least two frequency slices")
    numerator = (fields[:, 1:] * fields[:, :-1].conj()).sum(dim=1)
    denominator = fields[:, :-1].abs().square().sum(dim=1).clamp_min(1e-8)
    coefficient = numerator / ((1.0 + ridge) * denominator)
    return coefficient * torch.clamp(1.0 / coefficient.abs().clamp_min(1e-8), max=1.0)


def ar1_query(
    reference: torch.Tensor,
    coefficient: torch.Tensor,
    reference_frequency: float,
    query_frequency: float | torch.Tensor,
) -> torch.Tensor:
    """Continuously query a stable complex pole from its reference state."""

    if not torch.is_tensor(query_frequency):
        query_frequency = torch.full(
            (reference.shape[0],),
            float(query_frequency),
            device=reference.device,
            dtype=reference.real.dtype,
        )
    delta = query_frequency - reference_frequency
    log_magnitude = torch.log(coefficient.abs().clamp_min(1e-8))
    phase = torch.angle(coefficient)
    factor = torch.exp(delta[:, None, None] * log_magnitude) * torch.polar(
        torch.ones_like(phase), delta[:, None, None] * phase
    )
    return reference * factor


class InvariantAR1Calibrator(nn.Module):
    """Infer one reference field and a stable complex spectral pole from sensors."""

    def __init__(
        self,
        visible_count: int,
        environment_channels: int,
        *,
        calibration_count: int = 6,
        width: int = 32,
        sensor_token_grid: tuple[int, int] = (2, 4),
        max_log_amplitude_correction: float = 1.5,
        max_phase_correction: float = math.pi,
        max_log_pole_correction: float = 0.25,
        max_pole_angle_correction: float = math.pi,
        ar_ridge: float = 0.01,
    ):
        super().__init__()
        self.calibration_count = int(calibration_count)
        self.sensor_token_grid = tuple(map(int, sensor_token_grid))
        self.max_log_amplitude_correction = float(max_log_amplitude_correction)
        self.max_phase_correction = float(max_phase_correction)
        self.max_log_pole_correction = float(max_log_pole_correction)
        self.max_pole_angle_correction = float(max_pole_angle_correction)
        self.ar_ridge = float(ar_ridge)
        channels = visible_count + 2 * (visible_count - 1)
        channels += calibration_count + 2 * calibration_count
        evidence_channels = 3 * calibration_count
        channels += evidence_channels + 1
        channels += evidence_channels + 1
        channels += evidence_channels
        channels += environment_channels + 2
        self.backbone = QueryUNet(channels, width=width, output_channels=4)
        nn.init.zeros_(self.backbone.out.weight)
        nn.init.zeros_(self.backbone.out.bias)

    def invariant_features(
        self,
        visible_fields: torch.Tensor,
        calibration_prior: torch.Tensor,
        sparse_observations: torch.Tensor,
        sensor_mask: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        adjacent = confidence_correlation(visible_fields[:, 1:], visible_fields[:, :-1])
        endpoint = visible_fields[:, -1:]
        prior_relative = confidence_correlation(calibration_prior, endpoint)
        observed_relative = (
            confidence_correlation(sparse_observations, calibration_prior) * sensor_mask
        )
        log_amplitude_ratio = (
            torch.log((sparse_observations.abs() + 1e-6) / (calibration_prior.abs() + 1e-6)).clamp(
                -4.0, 4.0
            )
            * sensor_mask
        )
        evidence = torch.cat(
            (
                observed_relative.real,
                observed_relative.imag,
                log_amplitude_ratio,
            ),
            dim=1,
        )
        pooled_evidence, occupied = normalized_sensor_splat(
            evidence, sensor_mask[:, :1], self.sensor_token_grid
        )
        sensor_count = sensor_mask[:, :1].sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        global_evidence = evidence.sum(dim=(2, 3), keepdim=True) / sensor_count
        global_evidence = global_evidence.expand(-1, -1, *evidence.shape[-2:])
        return torch.cat(
            (
                visible_fields.abs(),
                adjacent.real,
                adjacent.imag,
                calibration_prior.abs(),
                prior_relative.real,
                prior_relative.imag,
                evidence,
                sensor_mask[:, :1],
                pooled_evidence,
                occupied,
                global_evidence,
                environment_and_coordinates[:, :-1],
            ),
            dim=1,
        )

    def forward(
        self,
        visible_fields: torch.Tensor,
        calibration_prior: torch.Tensor,
        sparse_observations: torch.Tensor,
        sensor_mask: torch.Tensor,
        environment_and_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.backbone(
            self.invariant_features(
                visible_fields,
                calibration_prior,
                sparse_observations,
                sensor_mask,
                environment_and_coordinates,
            )
        )
        prior_reference = calibration_prior[:, -1]
        reference_amplitude = prior_reference.abs() * torch.exp(
            self.max_log_amplitude_correction * torch.tanh(raw[:, 0])
        )
        reference_phase = torch.angle(prior_reference) + self.max_phase_correction * torch.tanh(
            raw[:, 1]
        )
        reference = reference_amplitude * torch.polar(
            torch.ones_like(reference_phase), reference_phase
        )
        prior_pole = fit_stable_ar1(calibration_prior, self.ar_ridge)
        log_magnitude = torch.log(prior_pole.abs().clamp_min(1e-8))
        log_magnitude = log_magnitude + self.max_log_pole_correction * torch.tanh(raw[:, 2])
        magnitude = torch.exp(torch.minimum(log_magnitude, torch.zeros_like(log_magnitude)))
        pole_phase = torch.angle(prior_pole) + self.max_pole_angle_correction * torch.tanh(
            raw[:, 3]
        )
        pole = magnitude * torch.polar(torch.ones_like(pole_phase), pole_phase)
        return reference, pole
