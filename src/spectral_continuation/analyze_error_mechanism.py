"""Attribute continuation error to range, interference nodes, and phase curvature."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_ocean_npz
from .representation_data import FrequencyBandDataset
from .train_representation_poc import build_model, coordinates


def _aggregate(mask: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mask = mask.to(target.real.dtype)
    numerator = ((prediction - target).abs().square() * mask).sum()
    denominator = (target.abs().square() * mask).sum().clamp_min(1e-12)
    phase = torch.angle(prediction * target.conj())
    weight = target.abs() * mask
    weight_sum = weight.sum().clamp_min(1e-12)
    return {
        "complex_rel_l2": float((numerator / denominator).sqrt()),
        "phase_coherence": float((weight * phase.cos()).sum() / weight_sum),
        "target_energy_fraction": float(denominator / target.abs().square().sum()),
        "squared_error_fraction": float(numerator / (prediction - target).abs().square().sum()),
        "pixel_fraction": float(mask.mean()),
    }


def _quantile_bins(
    value: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, count: int = 4
) -> tuple[list[dict[str, float]], list[float]]:
    boundaries = torch.quantile(
        value.flatten(), torch.linspace(0, 1, count + 1, device=value.device)
    )
    rows = []
    for index in range(count):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = value >= lower
        mask &= value <= upper if index == count - 1 else value < upper
        rows.append(
            {
                "bin": index + 1,
                "lower": float(lower),
                "upper": float(upper),
                **_aggregate(mask, prediction, target),
            }
        )
    return rows, boundaries.cpu().tolist()


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        args.run_summary.parent / "best.pt", map_location="cpu", weights_only=False
    )
    bundle = load_ocean_npz(args.dataset)
    positions = np.flatnonzero(bundle.split == "val")
    visible = tuple(float(value) for value in checkpoint["visible"])
    if visible != tuple(map(float, range(74, 81))):
        raise ValueError(
            "this attribution probe is defined for the frozen 74--80 Hz extreme protocol"
        )
    dataset = FrequencyBandDataset(
        bundle,
        positions,
        visible,
        (args.query,),
        checkpoint["normalization"],
        field_only=bool(summary["args"].get("field_only", False)),
    )
    model_args = Namespace(**summary["args"])
    model = build_model(
        model_args,
        visible,
        dataset.environment_channels,
        checkpoint["normalization"].frequency_scale,
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions, targets = [], []
    for start in range(0, len(dataset), args.batch_size):
        items = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
        fields = torch.stack([item[0] for item in items]).to(args.device)
        environment = torch.stack([item[1] for item in items]).to(args.device)
        target = torch.stack([item[2][0] for item in items]).to(args.device)
        query = torch.full((len(items),), args.query, device=args.device)
        context = coordinates(environment, query, checkpoint["normalization"].frequency_scale)
        prediction = model(fields, context, query)
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)

    raw_fields = {
        frequency: torch.complex(
            bundle.fields[frequency][positions, 0], bundle.fields[frequency][positions, 1]
        )
        for frequency in bundle.frequencies
        if 74 <= frequency <= 100
    }
    endpoint_magnitude = raw_fields[80.0].abs()
    visible_ratio_previous = raw_fields[79.0] * raw_fields[78.0].conj()
    visible_ratio_endpoint = raw_fields[80.0] * raw_fields[79.0].conj()
    visible_curvature = torch.angle(
        visible_ratio_endpoint * visible_ratio_previous.conj()
    ).abs()
    future_curvatures = []
    for frequency in range(82, 101):
        previous = raw_fields[float(frequency - 1)] * raw_fields[float(frequency - 2)].conj()
        current = raw_fields[float(frequency)] * raw_fields[float(frequency - 1)].conj()
        future_curvatures.append(torch.angle(current * previous.conj()).abs())
    future_curvature = torch.stack(future_curvatures).mean(dim=0)

    with np.load(args.dataset, allow_pickle=False) as source:
        ranges = source["ranges_m"].copy()
    edges = np.linspace(ranges.min(), ranges.max(), 5)
    range_rows = []
    for index in range(4):
        columns = torch.from_numpy(
            (ranges >= edges[index])
            & ((ranges <= edges[index + 1]) if index == 3 else (ranges < edges[index + 1]))
        )
        mask = columns.view(1, 1, -1).expand_as(target.real)
        range_rows.append(
            {
                "range_min_m": float(edges[index]),
                "range_max_m": float(edges[index + 1]),
                **_aggregate(mask, prediction, target),
            }
        )

    target_magnitude_rows, _ = _quantile_bins(target.abs(), prediction, target)
    endpoint_magnitude_rows, _ = _quantile_bins(endpoint_magnitude, prediction, target)
    visible_curvature_rows, _ = _quantile_bins(visible_curvature, prediction, target)
    future_curvature_rows, _ = _quantile_bins(future_curvature, prediction, target)

    error_by_environment_range = (
        (prediction - target).abs().square().sum(dim=1)
        / target.abs().square().sum(dim=1).clamp_min(1e-12)
    ).sqrt()
    phase_by_environment_range = (
        (target.abs() * torch.angle(prediction * target.conj()).cos()).sum(dim=1)
        / target.abs().sum(dim=1).clamp_min(1e-12)
    )
    range_grid = torch.from_numpy(ranges).view(1, -1).expand_as(error_by_environment_range)
    future_curve_by_range = (
        (endpoint_magnitude * future_curvature).sum(dim=1)
        / endpoint_magnitude.sum(dim=1).clamp_min(1e-12)
    )

    def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
        left = left.float().flatten()
        right = right.float().flatten()
        left = left - left.mean()
        right = right - right.mean()
        return float((left * right).mean() / (left.std() * right.std()).clamp_min(1e-12))

    global_metrics = _aggregate(torch.ones_like(target.real, dtype=torch.bool), prediction, target)
    payload = {
        "schema_version": 1,
        "status": "frozen-checkpoint error attribution",
        "checkpoint": str(args.run_summary.parent / "best.pt"),
        "query_hz": args.query,
        "global": global_metrics,
        "range_bins": range_rows,
        "target_magnitude_quartiles": target_magnitude_rows,
        "visible_endpoint_magnitude_quartiles": endpoint_magnitude_rows,
        "visible_phase_curvature_quartiles": visible_curvature_rows,
        "future_phase_curvature_quartiles": future_curvature_rows,
        "correlations_over_environment_range_cells": {
            "range_vs_rel_error": correlation(range_grid, error_by_environment_range),
            "range_vs_phase_coherence": correlation(range_grid, phase_by_environment_range),
            "future_curvature_vs_rel_error": correlation(
                future_curve_by_range, error_by_environment_range
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "error_attribution.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].plot(
        [(row["range_min_m"] + row["range_max_m"]) / 2 for row in range_rows],
        [row["complex_rel_l2"] for row in range_rows],
        marker="o",
    )
    axes[0].set(xlabel="Range (m)", ylabel="Restricted complex RelL2", title="Range")
    axes[1].bar(
        np.arange(1, 5), [row["squared_error_fraction"] for row in target_magnitude_rows]
    )
    axes[1].set(
        xlabel="Target magnitude quartile (low to high)",
        ylabel="Fraction of squared error",
        title="Interference nodes",
    )
    axes[2].plot(
        np.arange(1, 5),
        [row["complex_rel_l2"] for row in future_curvature_rows],
        marker="o",
    )
    axes[2].set(
        xlabel="True phase-curvature quartile",
        ylabel="Restricted complex RelL2",
        title="Spectral curvature",
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "error_attribution.png", dpi=220)
    figure.savefig(args.output_dir / "error_attribution.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query", type=float, default=110.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
