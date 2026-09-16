"""Evaluate sparse spectral calibration on a frozen pole--residue checkpoint."""

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
    CalibrationConfig,
    decode_calibrated,
    fit_sparse_phase_calibration,
    nested_sensor_indices,
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


def _vectors(prediction: torch.Tensor, target: torch.Tensor):
    pred = torch.stack((prediction.real, prediction.imag), dim=1).cpu()
    true = torch.stack((target.real, target.imag), dim=1).cpu()
    return metric_vectors(pred, true)


def _summarize(vectors):
    return {name: float(np.mean(value)) for name, value in vectors.items()}


@torch.no_grad()
def _predict_queries(model, state, fields, queries, calibration=None, config=None):
    outputs = []
    for frequency in queries:
        query = torch.full(
            (fields.shape[0],), frequency, device=fields.device, dtype=fields.real.dtype
        )
        if calibration is None:
            outputs.append(model.decode(*state[:3], fields, query, state[3], state[4]))
        else:
            outputs.append(decode_calibrated(model, state, fields, query, calibration, config))
    return torch.stack(outputs, dim=1)


def _evaluate_prediction(prediction, targets, queries):
    vectors = {}
    metrics = {}
    for index, frequency in enumerate(queries):
        current = _vectors(prediction[:, index], targets[:, index])
        vectors[f"{frequency:g}"] = current
        metrics[f"{frequency:g}"] = _summarize(current)
    metric_names = next(iter(vectors.values()))
    metrics["macro_average"] = {
        name: float(np.mean([metrics[f"{frequency:g}"][name] for frequency in queries]))
        for name in metric_names
    }
    return metrics, vectors


def run(args):
    device = torch.device(args.device)
    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    checkpoint_path = args.run_summary.parent / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle = load_ocean_npz(args.dataset)
    split = np.flatnonzero(bundle.split == args.split)
    if args.max_environments is not None:
        split = split[: args.max_environments]
    visible = tuple(map(float, checkpoint["visible"]))
    calibration_frequencies = tuple(map(float, args.calibration_frequencies.split(",")))
    queries = tuple(map(float, args.queries.split(",")))
    if set(calibration_frequencies).intersection(queries):
        raise ValueError("calibration and query frequencies must be disjoint")
    normalization = checkpoint["normalization"]
    model = build_model(
        Namespace(**run_summary["args"]),
        visible,
        bundle.environment.shape[1],
        normalization.frequency_scale,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    positions = torch.as_tensor(split, dtype=torch.long)
    fields = _complex_fields(bundle, visible, positions, normalization.field_scale).to(device)
    calibration_targets = _complex_fields(
        bundle, calibration_frequencies, positions, normalization.field_scale
    ).to(device)
    targets = _complex_fields(bundle, queries, positions, normalization.field_scale).to(device)
    environment = bundle.environment[positions]
    environment = (environment - normalization.environment_mean[0]) / normalization.environment_std[
        0
    ]
    query_stub = torch.full((len(split),), visible[-1], dtype=torch.float32)
    context = coordinates(
        environment.to(device), query_stub.to(device), normalization.frequency_scale
    )
    with torch.no_grad():
        state = tuple(value.detach() for value in model.spectral_state(fields, context))
        baseline_prediction = _predict_queries(model, state, fields, queries)
    baseline_metrics, baseline_vectors = _evaluate_prediction(baseline_prediction, targets, queries)

    budgets = tuple(map(int, args.sensor_budgets.split(",")))
    mask_seeds = tuple(map(int, args.mask_seeds.split(",")))
    maximum_sensors = max(budgets)
    configs = []
    for grid_text in args.token_grids.split(","):
        height, width = map(int, grid_text.lower().split("x"))
        configs.append(
            CalibrationConfig(
                token_grid=(height, width),
                max_slope_correction_per_hz=args.max_slope_correction_per_hz,
                max_curvature_correction_per_hz2=args.max_curvature_correction_per_hz2,
                steps=args.steps,
                learning_rate=args.learning_rate,
            )
        )

    records = []
    vector_payload = {}
    point_count = fields.shape[-2] * fields.shape[-1]
    for mask_seed in mask_seeds:
        order = nested_sensor_indices(len(split), point_count, maximum_sensors, mask_seed)
        for budget in budgets:
            sensor_indices = torch.as_tensor(order[:, :budget], dtype=torch.long, device=device)
            for config in configs:
                calibration = fit_sparse_phase_calibration(
                    model,
                    state,
                    fields,
                    calibration_targets,
                    calibration_frequencies,
                    sensor_indices,
                    config,
                )
                prediction = _predict_queries(model, state, fields, queries, calibration, config)
                metrics, vectors = _evaluate_prediction(prediction, targets, queries)
                method = f"phase_{config.token_grid[0]}x{config.token_grid[1]}"
                key = f"{method}_k{budget}_mask{mask_seed}"
                for frequency in queries:
                    current = vectors[f"{frequency:g}"]
                    baseline = baseline_vectors[f"{frequency:g}"]
                    stats = paired_bootstrap(
                        baseline["complex_rel_l2"],
                        current["complex_rel_l2"],
                        seed=mask_seed + int(frequency * 100),
                    )
                    records.append(
                        {
                            "method": method,
                            "budget_per_frequency": budget,
                            "total_complex_observations": budget * len(calibration_frequencies),
                            "mask_seed": mask_seed,
                            "query_hz": frequency,
                            **metrics[f"{frequency:g}"],
                            **{f"paired_{name}": value for name, value in stats.items()},
                            "final_calibration_loss": calibration.loss_history[-1],
                        }
                    )
                    for metric_name, value in current.items():
                        vector_payload[f"{key}_q{frequency:g}_{metric_name}"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "metric_vectors.npz", **vector_payload)
    with (args.output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    output = {
        "schema_version": 1,
        "status": "sparse spectral calibration POC",
        "run_summary": str(args.run_summary),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset": str(args.dataset),
        "split": args.split,
        "environment_count": len(split),
        "visible_hz": visible,
        "calibration_hz": calibration_frequencies,
        "query_hz": queries,
        "sensor_budgets_per_frequency": budgets,
        "mask_seeds": mask_seeds,
        "baseline_metrics": baseline_metrics,
        "configs": [config.__dict__ for config in configs],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--max-environments", type=int)
    parser.add_argument("--calibration-frequencies", default="105,106")
    parser.add_argument("--queries", default="107,108,109,110,111,112")
    parser.add_argument("--sensor-budgets", default="23,46,92,184,369")
    parser.add_argument("--mask-seeds", default="4201,4301,4401")
    parser.add_argument("--token-grids", default="1x1,2x4,4x8")
    parser.add_argument("--max-slope-correction-per-hz", type=float, default=0.2)
    parser.add_argument("--max-curvature-correction-per-hz2", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
