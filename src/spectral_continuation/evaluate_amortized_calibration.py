"""Evaluate whether sparse calibration transfers to entirely unobserved frequencies."""

from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from .data import load_ocean_npz
from .metrics import metric_vectors, paired_bootstrap
from .sparse_calibration import (
    nested_sensor_indices,
    stratified_energy_sensor_indices,
)
from .spectral_calibrator import (
    InvariantSpectralCalibrator,
    endpoint_carrier_query,
    enforce_sensor_consistency,
)
from .train_representation_poc import build_model, coordinates, sha256_file


def _complex_fields(bundle, frequencies, positions, scale):
    return (
        torch.stack(
            [
                torch.complex(
                    bundle.fields[frequency][positions, 0],
                    bundle.fields[frequency][positions, 1],
                )
                for frequency in frequencies
            ],
            dim=1,
        )
        / scale
    )


def _metric_vectors(prediction, target):
    return metric_vectors(
        torch.stack((prediction.real, prediction.imag), dim=1).cpu(),
        torch.stack((target.real, target.imag), dim=1).cpu(),
    )


def _mean_metrics(vectors):
    return {name: float(values.mean()) for name, values in vectors.items()}


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    calibrator_summary = json.loads(args.calibrator_summary.read_text(encoding="utf-8"))
    calibrator_args = Namespace(**calibrator_summary["args"])
    calibrator_checkpoint_path = args.calibrator_summary.parent / "best.pt"
    calibrator_checkpoint = torch.load(
        calibrator_checkpoint_path, map_location="cpu", weights_only=False
    )
    base_summary_path = Path(calibrator_summary["base_run_summary"])
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    base_checkpoint_path = base_summary_path.parent / "best.pt"
    base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    bundle = load_ocean_npz(args.dataset)
    positions = np.flatnonzero(bundle.split == args.split)
    if args.split == "train" and args.train_tail_only:
        positions = positions[calibrator_summary["fit_environment_count"] :]
    if args.max_environments is not None:
        positions = positions[: args.max_environments]
    index = torch.as_tensor(positions, dtype=torch.long)
    normalization = base_checkpoint["normalization"]
    visible_frequencies = tuple(map(float, base_checkpoint["visible"]))
    calibration_frequencies = tuple(map(float, calibrator_summary["calibration_hz"]))
    queries = tuple(map(float, args.queries.split(",")))
    if set(calibration_frequencies).intersection(queries):
        raise ValueError("query frequencies must have zero calibration observations")

    visible = _complex_fields(bundle, visible_frequencies, index, normalization.field_scale).to(
        device
    )
    calibration_target = _complex_fields(
        bundle, calibration_frequencies, index, normalization.field_scale
    ).to(device)
    query_target = _complex_fields(bundle, queries, index, normalization.field_scale).to(device)
    environment = bundle.environment[index]
    environment = (environment - normalization.environment_mean[0]) / normalization.environment_std[
        0
    ]
    query_stub = torch.full((len(index),), visible_frequencies[-1], device=device)
    context = coordinates(environment.to(device), query_stub, normalization.frequency_scale)

    base_model = build_model(
        Namespace(**base_summary["args"]),
        visible_frequencies,
        bundle.environment.shape[1],
        normalization.frequency_scale,
    ).to(device)
    base_model.load_state_dict(base_checkpoint["model"])
    base_model.eval()
    state = base_model.spectral_state(visible, context)

    def base_decode(frequency):
        query = torch.full((len(index),), frequency, device=device, dtype=visible.real.dtype)
        return base_model.decode(*state[:3], visible, query, state[3], state[4])

    calibration_prior = torch.stack(
        [base_decode(frequency) for frequency in calibration_frequencies], dim=1
    )
    base_query = torch.stack([base_decode(frequency) for frequency in queries], dim=1)
    prior_carrier = torch.stack(
        [
            endpoint_carrier_query(calibration_prior, calibration_frequencies, frequency)
            for frequency in queries
        ],
        dim=1,
    )
    dense_oracle = torch.stack(
        [
            endpoint_carrier_query(calibration_target, calibration_frequencies, frequency)
            for frequency in queries
        ],
        dim=1,
    )

    calibrator = InvariantSpectralCalibrator(
        len(visible_frequencies),
        bundle.environment.shape[1],
        calibration_count=2,
        width=calibrator_args.width,
        sensor_token_grid=tuple(calibrator_args.sensor_token_grid),
        max_log_amplitude_correction=calibrator_args.max_log_amplitude_correction,
        max_phase_correction=calibrator_args.max_phase_correction,
        max_slope_correction_per_hz=calibrator_args.max_slope_correction_per_hz,
        calibration_frequency_step=(calibration_frequencies[1] - calibration_frequencies[0]),
    ).to(device)
    calibrator.load_state_dict(calibrator_checkpoint["model"])
    calibrator.eval()

    baseline_vectors = {}
    records = []
    vector_payload = {}

    def record(method, budget, mask_seed, prediction):
        for query_index, frequency in enumerate(queries):
            vectors = _metric_vectors(prediction[:, query_index], query_target[:, query_index])
            if method == "hyper_pole":
                baseline_vectors[f"{frequency:g}"] = vectors
            baseline = baseline_vectors.get(f"{frequency:g}")
            paired = (
                paired_bootstrap(
                    baseline["complex_rel_l2"],
                    vectors["complex_rel_l2"],
                    seed=int(frequency * 100) + max(mask_seed, 0),
                )
                if baseline is not None
                else {}
            )
            records.append(
                {
                    "method": method,
                    "budget_per_frequency": budget,
                    "total_complex_observations": budget * len(calibration_frequencies),
                    "mask_seed": mask_seed,
                    "query_hz": frequency,
                    **_mean_metrics(vectors),
                    **{f"paired_{name}": value for name, value in paired.items()},
                }
            )
            key = f"{method}_k{budget}_mask{mask_seed}_q{frequency:g}"
            for metric_name, values in vectors.items():
                vector_payload[f"{key}_{metric_name}"] = values

    record("hyper_pole", 0, -1, base_query)
    record("prior_endpoint_carrier", 0, -1, prior_carrier)
    record("dense_calibration_oracle", visible.shape[-2] * visible.shape[-1], -1, dense_oracle)
    zero_mask = torch.zeros(len(index), 1, *visible.shape[-2:], device=device)
    zero_prediction = calibrator(
        visible,
        calibration_prior,
        torch.zeros_like(calibration_target),
        zero_mask,
        context,
    )
    zero_query = torch.stack(
        [
            endpoint_carrier_query(zero_prediction, calibration_frequencies, frequency)
            for frequency in queries
        ],
        dim=1,
    )
    record("amortized_no_sensors", 0, -1, zero_query)

    budgets = tuple(map(int, args.sensor_budgets.split(",")))
    mask_seeds = tuple(map(int, args.mask_seeds.split(",")))
    point_count = visible.shape[-2] * visible.shape[-1]
    for mask_seed in mask_seeds:
        order = nested_sensor_indices(len(index), point_count, max(budgets), mask_seed)
        for budget in budgets:
            sensor_indices = torch.as_tensor(order[:, :budget], dtype=torch.long, device=device)
            mask = torch.zeros(len(index), point_count, device=device)
            mask.scatter_(1, sensor_indices, 1.0)
            mask = mask.reshape(len(index), 1, *visible.shape[-2:])
            sparse = calibration_target * mask
            prediction = calibrator(visible, calibration_prior, sparse, mask, context)
            prediction = enforce_sensor_consistency(prediction, sparse, mask)
            query_prediction = torch.stack(
                [
                    endpoint_carrier_query(prediction, calibration_frequencies, frequency)
                    for frequency in queries
                ],
                dim=1,
            )
            record("amortized_calibration", budget, mask_seed, query_prediction)

            shuffled_sparse = calibration_target.roll(1, dims=0) * mask
            shuffled = calibrator(visible, calibration_prior, shuffled_sparse, mask, context)
            shuffled_query = torch.stack(
                [
                    endpoint_carrier_query(shuffled, calibration_frequencies, frequency)
                    for frequency in queries
                ],
                dim=1,
            )
            record("shuffled_instance", budget, mask_seed, shuffled_query)

    if args.include_energy_layout:
        grid = tuple(calibrator_args.sensor_token_grid)
        for budget in budgets:
            if budget < grid[0] * grid[1]:
                continue
            sensor_indices = stratified_energy_sensor_indices(calibration_prior, budget, grid)
            mask = torch.zeros(len(index), point_count, device=device)
            mask.scatter_(1, sensor_indices, 1.0)
            mask = mask.reshape(len(index), 1, *visible.shape[-2:])
            sparse = calibration_target * mask
            prediction = calibrator(visible, calibration_prior, sparse, mask, context)
            prediction = enforce_sensor_consistency(prediction, sparse, mask)
            query_prediction = torch.stack(
                [
                    endpoint_carrier_query(prediction, calibration_frequencies, frequency)
                    for frequency in queries
                ],
                dim=1,
            )
            record("amortized_calibration_energy", budget, 0, query_prediction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "metric_vectors.npz", **vector_payload)
    with (args.output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "schema_version": 1,
        "status": "frozen sparse spectral calibration evaluation",
        "calibrator_summary": str(args.calibrator_summary),
        "calibrator_checkpoint_sha256": sha256_file(calibrator_checkpoint_path),
        "base_checkpoint_sha256": sha256_file(base_checkpoint_path),
        "dataset": str(args.dataset),
        "split": args.split,
        "environment_count": len(index),
        "visible_hz": visible_frequencies,
        "calibration_hz": calibration_frequencies,
        "query_hz": queries,
        "sensor_budgets_per_frequency": budgets,
        "mask_seeds": mask_seeds,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrator-summary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--train-tail-only", action="store_true")
    parser.add_argument("--max-environments", type=int)
    parser.add_argument("--queries", default="107,108,109,110,111,112")
    parser.add_argument("--sensor-budgets", default="23,46,92,184,369")
    parser.add_argument("--mask-seeds", default="4201,4301,4401")
    parser.add_argument("--include-energy-layout", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
