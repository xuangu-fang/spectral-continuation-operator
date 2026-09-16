"""Test-time sparse calibration of a frozen continuous spectral state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CalibrationConfig:
    token_grid: tuple[int, int] = (4, 8)
    max_slope_correction_per_hz: float = 0.2
    max_curvature_correction_per_hz2: float = 0.0
    steps: int = 200
    learning_rate: float = 0.08


@dataclass
class CalibratedState:
    slope_token: torch.Tensor
    curvature_token: torch.Tensor | None
    loss_history: list[float]


def nested_sensor_indices(
    batch_size: int,
    point_count: int,
    maximum_sensors: int,
    seed: int,
) -> np.ndarray:
    """Return independent nested random sensor orders for each environment."""

    if not 0 < maximum_sensors <= point_count:
        raise ValueError("maximum_sensors must lie in [1, point_count]")
    generator = np.random.default_rng(seed)
    return np.stack(
        [generator.permutation(point_count)[:maximum_sensors] for _ in range(batch_size)]
    )


def stratified_energy_sensor_indices(
    calibration_prior: torch.Tensor,
    sensor_count: int,
    grid: tuple[int, int],
) -> torch.Tensor:
    """Choose high-confidence points while preserving spatial coverage."""

    batch, _, height, width = calibration_prior.shape
    rows, columns = grid
    if sensor_count < rows * columns:
        raise ValueError("sensor_count must cover every stratification cell")
    if height % rows or width % columns:
        raise ValueError("stratification grid must divide the spatial shape")
    score = calibration_prior.abs().amin(dim=1)
    quota, remainder = divmod(sensor_count, rows * columns)
    selections = []
    for batch_index in range(batch):
        current = []
        cell_index = 0
        for row in range(rows):
            row_start = row * height // rows
            row_end = (row + 1) * height // rows
            for column in range(columns):
                column_start = column * width // columns
                column_end = (column + 1) * width // columns
                local = score[batch_index, row_start:row_end, column_start:column_end].flatten()
                count = quota + int(cell_index < remainder)
                local_indices = torch.topk(local, count).indices
                local_width = column_end - column_start
                local_row = torch.div(local_indices, local_width, rounding_mode="floor")
                local_column = local_indices % local_width
                current.append((local_row + row_start) * width + local_column + column_start)
                cell_index += 1
        selections.append(torch.cat(current))
    return torch.stack(selections)


def _token_field(
    raw: torch.Tensor,
    output_shape: tuple[int, int],
    bound: float,
) -> torch.Tensor:
    return (
        bound
        * torch.tanh(F.interpolate(raw, size=output_shape, mode="bilinear", align_corners=False))[
            :, 0
        ]
    )


def calibrated_components(
    base_slope: torch.Tensor,
    base_curvature: torch.Tensor,
    calibration: CalibratedState,
    config: CalibrationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply bounded common-mode slope/curvature corrections to a pole state."""

    spatial_shape = tuple(base_slope.shape[-2:])
    slope_delta = _token_field(
        calibration.slope_token,
        spatial_shape,
        config.max_slope_correction_per_hz,
    )
    slope = base_slope + slope_delta
    curvature = base_curvature
    if calibration.curvature_token is not None:
        curvature_delta = _token_field(
            calibration.curvature_token,
            spatial_shape,
            config.max_curvature_correction_per_hz2,
        )
        curvature = curvature + curvature_delta[:, None]
    return slope, curvature


def decode_calibrated(
    model,
    state: tuple[torch.Tensor, ...],
    fields: torch.Tensor,
    query_frequency: torch.Tensor,
    calibration: CalibratedState,
    config: CalibrationConfig,
) -> torch.Tensor:
    slope, curvature = calibrated_components(state[3], state[4], calibration, config)
    return model.decode(*state[:3], fields, query_frequency, slope, curvature)


def _gather_points(field: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flattened = field.flatten(1)
    return torch.gather(flattened, 1, indices)


def sparse_calibration_loss(
    model,
    state: tuple[torch.Tensor, ...],
    fields: torch.Tensor,
    calibration_targets: torch.Tensor,
    calibration_frequencies: tuple[float, ...],
    sensor_indices: torch.Tensor,
    calibration: CalibratedState,
    config: CalibrationConfig,
) -> torch.Tensor:
    """Observed complex relative squared error, averaged over environments."""

    predictions = []
    for frequency in calibration_frequencies:
        query = torch.full(
            (fields.shape[0],),
            frequency,
            device=fields.device,
            dtype=fields.real.dtype,
        )
        predictions.append(decode_calibrated(model, state, fields, query, calibration, config))
    prediction = torch.stack(predictions, dim=1)
    expanded_indices = sensor_indices[:, None].expand(-1, len(calibration_frequencies), -1)
    observed_prediction = torch.gather(prediction.flatten(2), 2, expanded_indices)
    observed_target = torch.gather(calibration_targets.flatten(2), 2, expanded_indices)
    numerator = (observed_prediction - observed_target).abs().square().sum(dim=(1, 2))
    denominator = observed_target.abs().square().sum(dim=(1, 2)).clamp_min(1e-8)
    return (numerator / denominator).mean()


def fit_sparse_phase_calibration(
    model,
    state: tuple[torch.Tensor, ...],
    fields: torch.Tensor,
    calibration_targets: torch.Tensor,
    calibration_frequencies: tuple[float, ...],
    sensor_indices: torch.Tensor,
    config: CalibrationConfig,
) -> CalibratedState:
    """Fit only instance-specific low-resolution phase-generator tokens."""

    batch_size = fields.shape[0]
    slope_token = torch.zeros(
        batch_size,
        1,
        *config.token_grid,
        device=fields.device,
        dtype=fields.real.dtype,
        requires_grad=True,
    )
    curvature_token = None
    parameters = [slope_token]
    if config.max_curvature_correction_per_hz2 > 0:
        curvature_token = torch.zeros_like(slope_token, requires_grad=True)
        parameters.append(curvature_token)
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    loss_history = []
    detached_state = tuple(value.detach() for value in state)
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        calibration = CalibratedState(slope_token, curvature_token, loss_history)
        loss = sparse_calibration_loss(
            model,
            detached_state,
            fields,
            calibration_targets,
            calibration_frequencies,
            sensor_indices,
            calibration,
            config,
        )
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))
    return CalibratedState(
        slope_token.detach(),
        curvature_token.detach() if curvature_token is not None else None,
        loss_history,
    )
