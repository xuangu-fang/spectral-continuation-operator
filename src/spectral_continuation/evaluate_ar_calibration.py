"""Evaluate sparse broadband AR calibration on unseen future frequencies."""

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
from .sparse_calibration import nested_sensor_indices
from .spectral_calibrator import InvariantAR1Calibrator, ar1_query, fit_stable_ar1
from .train_representation_poc import build_model, coordinates, sha256_file
from .train_spectral_calibrator import _complex_fields


def _vectors(prediction, target):
    return metric_vectors(
        torch.stack((prediction.real, prediction.imag), dim=1).cpu(),
        torch.stack((target.real, target.imag), dim=1).cpu(),
    )


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    calibrator_summary = json.loads(args.calibrator_summary.read_text(encoding="utf-8"))
    calibrator_args = Namespace(**calibrator_summary["args"])
    calibrator_path = args.calibrator_summary.parent / "best.pt"
    calibrator_checkpoint = torch.load(calibrator_path, map_location="cpu", weights_only=False)
    base_summary_path = Path(calibrator_summary["base_run_summary"])
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    base_path = base_summary_path.parent / "best.pt"
    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
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
    reference_frequency = calibration_frequencies[-1]
    queries = tuple(map(float, args.queries.split(",")))
    if set(calibration_frequencies).intersection(queries):
        raise ValueError("future queries must not contain calibration frequencies")
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
    context = coordinates(
        environment.to(device),
        torch.full((len(index),), visible_frequencies[-1], device=device),
        normalization.frequency_scale,
    )
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
    prior_pole = fit_stable_ar1(calibration_prior, calibrator_args.ar_ridge)
    prior_query = torch.stack(
        [
            ar1_query(calibration_prior[:, -1], prior_pole, reference_frequency, frequency)
            for frequency in queries
        ],
        dim=1,
    )
    oracle_pole = fit_stable_ar1(calibration_target, calibrator_args.ar_ridge)
    oracle_query = torch.stack(
        [
            ar1_query(calibration_target[:, -1], oracle_pole, reference_frequency, frequency)
            for frequency in queries
        ],
        dim=1,
    )
    calibrator = InvariantAR1Calibrator(
        len(visible_frequencies),
        bundle.environment.shape[1],
        calibration_count=len(calibration_frequencies),
        width=calibrator_args.width,
        sensor_token_grid=tuple(calibrator_args.sensor_token_grid),
        max_log_amplitude_correction=calibrator_args.max_log_amplitude_correction,
        max_phase_correction=calibrator_args.max_phase_correction,
        max_log_pole_correction=calibrator_args.max_log_pole_correction,
        max_pole_angle_correction=calibrator_args.max_pole_angle_correction,
        ar_ridge=calibrator_args.ar_ridge,
    ).to(device)
    calibrator.load_state_dict(calibrator_checkpoint["model"])
    calibrator.eval()

    records = []
    payload = {}
    baseline_vectors = {}

    def record(method, budget, mask_seed, prediction):
        for query_index, frequency in enumerate(queries):
            vectors = _vectors(prediction[:, query_index], query_target[:, query_index])
            if method == "hyper_pole":
                baseline_vectors[f"{frequency:g}"] = vectors
            baseline = baseline_vectors.get(f"{frequency:g}")
            paired = (
                paired_bootstrap(
                    baseline["complex_rel_l2"],
                    vectors["complex_rel_l2"],
                    seed=max(mask_seed, 0) + int(frequency * 100),
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
                    **{name: float(values.mean()) for name, values in vectors.items()},
                    **{f"paired_{name}": value for name, value in paired.items()},
                }
            )
            key = f"{method}_k{budget}_mask{mask_seed}_q{frequency:g}"
            for name, values in vectors.items():
                payload[f"{key}_{name}"] = values

    record("hyper_pole", 0, -1, base_query)
    record("prior_ar1", 0, -1, prior_query)
    record("dense_ar1_oracle", visible.shape[-2] * visible.shape[-1], -1, oracle_query)
    zero_mask = torch.zeros(len(index), 1, *visible.shape[-2:], device=device)
    zero_reference, zero_pole = calibrator(
        visible,
        calibration_prior,
        torch.zeros_like(calibration_target),
        zero_mask,
        context,
    )
    zero_query = torch.stack(
        [
            ar1_query(zero_reference, zero_pole, reference_frequency, frequency)
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
            reference, pole = calibrator(visible, calibration_prior, sparse, mask, context)
            prediction = torch.stack(
                [
                    ar1_query(reference, pole, reference_frequency, frequency)
                    for frequency in queries
                ],
                dim=1,
            )
            record("amortized_ar1", budget, mask_seed, prediction)
            shuffled = calibration_target.roll(1, dims=0) * mask
            shuffled_reference, shuffled_pole = calibrator(
                visible, calibration_prior, shuffled, mask, context
            )
            shuffled_query = torch.stack(
                [
                    ar1_query(shuffled_reference, shuffled_pole, reference_frequency, frequency)
                    for frequency in queries
                ],
                dim=1,
            )
            record("shuffled_instance", budget, mask_seed, shuffled_query)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "metric_vectors.npz", **payload)
    with (args.output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "schema_version": 1,
        "status": "frozen sparse broadband AR calibration evaluation",
        "calibrator_summary": str(args.calibrator_summary),
        "calibrator_checkpoint_sha256": sha256_file(calibrator_path),
        "base_checkpoint_sha256": sha256_file(base_path),
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
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
