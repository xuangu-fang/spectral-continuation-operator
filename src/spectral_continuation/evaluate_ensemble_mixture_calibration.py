"""Replay sparse complex ensemble-mixture calibration on a frozen seed ensemble."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_horizon_features import _load_bundle, _same_normalization
from .horizon import complex_relative_error
from .representation_data import FrequencyBandDataset
from .sparse_calibration import nested_sensor_indices
from .train_representation_poc import build_model, coordinates, sha256_file


def _fit_sum_one_mixture(
    predictions: np.ndarray,
    target: np.ndarray,
    sensor_indices: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Fit complex seed weights with an exact sum-to-one parameterization."""
    seed_count, environment_count = predictions.shape[:2]
    if seed_count < 2:
        raise ValueError("ensemble mixture calibration requires at least two seeds")
    basis = np.zeros((seed_count, seed_count - 1), dtype=np.complex128)
    basis[:-1] = np.eye(seed_count - 1)
    basis[-1] = -1.0
    uniform = np.full(seed_count, 1.0 / seed_count, dtype=np.complex128)
    weights = np.empty((environment_count, seed_count), dtype=np.complex64)
    for environment in range(environment_count):
        selected = sensor_indices[environment]
        design = predictions[:, environment].reshape(seed_count, predictions.shape[2], -1)
        design = design[:, :, selected].transpose(1, 2, 0).reshape(-1, seed_count)
        response = target[environment].reshape(target.shape[1], -1)[:, selected].reshape(-1)
        centered = design @ basis
        residual = response - design @ uniform
        scale = max(len(response), 1)
        gram = centered.conj().T @ centered / scale
        gram += ridge * np.eye(seed_count - 1)
        rhs = centered.conj().T @ residual / scale
        coefficients = np.linalg.solve(gram, rhs)
        weights[environment] = uniform + basis @ coefficients
    return weights


def _weighted_prediction(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("es,seqhw->eqhw", weights, predictions, optimize=True)


def _weighted_disagreement(predictions: np.ndarray, weighted: np.ndarray) -> dict[str, np.ndarray]:
    energy = np.maximum(np.sum(np.abs(weighted) ** 2, axis=(-2, -1)), 1e-12)
    complex_value = np.sqrt(
        np.mean(np.sum(np.abs(predictions - weighted[None]) ** 2, axis=(-2, -1)), axis=0) / energy
    )
    amplitude_value = np.sqrt(
        np.mean(
            np.sum(
                (np.abs(predictions) - np.abs(weighted)[None]) ** 2,
                axis=(-2, -1),
            ),
            axis=0,
        )
        / energy
    )
    cross = predictions * np.conj(weighted)[None]
    phase_cosine = np.real(cross) / np.maximum(np.abs(cross), 1e-12)
    spatial_weight = np.abs(weighted)
    phase_value = np.mean(
        np.sum(spatial_weight[None] * (1.0 - phase_cosine), axis=(-2, -1))
        / np.maximum(np.sum(spatial_weight, axis=(-2, -1))[None], 1e-12),
        axis=0,
    )
    return {"complex": complex_value, "amplitude": amplitude_value, "phase": phase_value}


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.run_summary]
    first = summaries[0]
    models = {summary["model"] for summary in summaries}
    visible_sets = {tuple(summary["visible_frequencies_hz"]) for summary in summaries}
    domains = {summary["args"]["domain"] for summary in summaries}
    if len(models) != 1 or len(visible_sets) != 1 or len(domains) != 1:
        raise ValueError("seed ensemble must share model, visible band, and domain")
    visible = tuple(map(float, first["visible_frequencies_hz"]))
    queries = tuple(map(float, args.queries.split(",")))
    endpoints = tuple(map(float, args.endpoints.split(",")))
    calibration_windows = {
        endpoint: tuple(endpoint - offset for offset in range(args.window - 1, -1, -1))
        for endpoint in endpoints
    }
    all_frequencies = tuple(
        sorted(set(queries).union(*(set(values) for values in calibration_windows.values())))
    )
    bundle = _load_bundle(first, all_frequencies)
    positions = np.flatnonzero(bundle.split == args.split)
    if args.max_environments is not None:
        positions = positions[: args.max_environments]
    device = torch.device(args.device)

    seed_predictions = []
    target_batches = []
    checkpoint_hashes = []
    reference_normalization = None
    for seed_index, (summary_path, summary) in enumerate(
        zip(args.run_summary, summaries, strict=True)
    ):
        checkpoint_path = summary_path.parent / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        normalization = checkpoint["normalization"]
        if reference_normalization is None:
            reference_normalization = normalization
        elif not _same_normalization(reference_normalization, normalization):
            raise RuntimeError("seed checkpoints use different normalizations")
        dataset = FrequencyBandDataset(
            bundle,
            positions,
            visible,
            all_frequencies,
            normalization,
            field_only=bool(summary["args"].get("field_only", False)),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = build_model(
            Namespace(**summary["args"]),
            visible,
            dataset.environment_channels,
            normalization.frequency_scale,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        prediction_batches = []
        for fields, environment, target in loader:
            fields = fields.to(device)
            environment = environment.to(device)
            anchor = torch.full(
                (len(fields),), all_frequencies[0], dtype=fields.real.dtype, device=device
            )
            context = coordinates(environment, anchor, normalization.frequency_scale)
            prediction_batches.append(
                model.forward_queries(fields, context, all_frequencies).cpu().numpy()
            )
            if seed_index == 0:
                target_batches.append(target.numpy())
        seed_predictions.append(np.concatenate(prediction_batches))
        checkpoint_hashes.append(sha256_file(checkpoint_path))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = np.stack(seed_predictions)
    targets = np.concatenate(target_batches)
    frequency_lookup = {frequency: index for index, frequency in enumerate(all_frequencies)}
    query_indices = [frequency_lookup[frequency] for frequency in queries]
    query_predictions = predictions[:, :, query_indices]
    query_target = targets[:, query_indices]
    base_prediction = query_predictions.mean(axis=0)
    base_error = complex_relative_error(base_prediction, query_target)

    height, width = query_target.shape[-2:]
    point_count = height * width
    sensor_indices = nested_sensor_indices(
        len(positions), point_count, args.sensor_budget, args.mask_seed
    )
    dense_indices = np.broadcast_to(np.arange(point_count), (len(positions), point_count))
    shape = (len(positions), len(endpoints), len(queries))
    calibrated_error = np.empty(shape, dtype=np.float32)
    shuffled_error = np.empty(shape, dtype=np.float32)
    dense_oracle_error = np.empty(shape, dtype=np.float32)
    complex_disagreement = np.empty(shape, dtype=np.float32)
    amplitude_disagreement = np.empty(shape, dtype=np.float32)
    phase_disagreement = np.empty(shape, dtype=np.float32)

    for endpoint_index, endpoint in enumerate(endpoints):
        calibration_indices = [
            frequency_lookup[frequency] for frequency in calibration_windows[endpoint]
        ]
        calibration_prediction = predictions[:, :, calibration_indices]
        calibration_target = targets[:, calibration_indices]
        weights = _fit_sum_one_mixture(
            calibration_prediction, calibration_target, sensor_indices, args.ridge
        )
        shuffled_weights = _fit_sum_one_mixture(
            calibration_prediction,
            np.roll(calibration_target, 1, axis=0),
            sensor_indices,
            args.ridge,
        )
        dense_weights = _fit_sum_one_mixture(
            calibration_prediction, calibration_target, dense_indices, args.ridge
        )
        calibrated_prediction = _weighted_prediction(query_predictions, weights)
        shuffled_prediction = _weighted_prediction(query_predictions, shuffled_weights)
        dense_prediction = _weighted_prediction(query_predictions, dense_weights)
        calibrated_error[:, endpoint_index] = complex_relative_error(
            calibrated_prediction, query_target
        )
        shuffled_error[:, endpoint_index] = complex_relative_error(
            shuffled_prediction, query_target
        )
        dense_oracle_error[:, endpoint_index] = complex_relative_error(
            dense_prediction, query_target
        )
        state_features = _weighted_disagreement(query_predictions, calibrated_prediction)
        complex_disagreement[:, endpoint_index] = state_features["complex"]
        amplitude_disagreement[:, endpoint_index] = state_features["amplitude"]
        phase_disagreement[:, endpoint_index] = state_features["phase"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "sequential_errors.npz",
        environment_ids=np.asarray(bundle.ids[positions]).astype(str),
        positions=positions,
        query_frequencies_hz=np.asarray(queries),
        calibration_endpoints_hz=np.asarray(endpoints),
        base_error=base_error,
        calibrated_error=calibrated_error,
        shuffled_error=shuffled_error,
        dense_oracle_error=dense_oracle_error,
        calibrated_seed_complex_disagreement=complex_disagreement,
        calibrated_seed_amplitude_disagreement=amplitude_disagreement,
        calibrated_seed_phase_disagreement=phase_disagreement,
        sensor_indices=sensor_indices,
    )
    output = {
        "schema_version": 1,
        "status": "sparse complex sum-to-one ensemble-mixture calibration replay",
        "domain": next(iter(domains)),
        "model": next(iter(models)),
        "environment_count": len(positions),
        "query_hz": queries,
        "calibration_endpoints_hz": endpoints,
        "window": args.window,
        "sensor_budget_per_frequency": args.sensor_budget,
        "spatial_observation_fraction": args.sensor_budget / point_count,
        "total_complex_observations_per_trigger": args.sensor_budget * args.window,
        "ridge": args.ridge,
        "run_summaries": [str(path) for path in args.run_summary],
        "checkpoint_sha256": checkpoint_hashes,
        "base_macro_complex_rel_l2": float(base_error.mean()),
        "endpoint_macro_complex_rel_l2": {
            f"{endpoint:g}": {
                "sparse_calibration": float(calibrated_error[:, index].mean()),
                "shuffled_instance": float(shuffled_error[:, index].mean()),
                "dense_calibration_ceiling": float(dense_oracle_error[:, index].mean()),
            }
            for index, endpoint in enumerate(endpoints)
        },
        "best_endpoint_macro_complex_rel_l2": {
            "sparse_calibration": float(calibrated_error.mean(axis=(0, 2)).min()),
            "dense_calibration_ceiling": float(dense_oracle_error.mean(axis=(0, 2)).min()),
        },
        "leakage_note": (
            "mixture weights use only the rolling sparse calibration window; future query "
            "targets are used only for scoring"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", default="19,19.5,20,21,22,23,24")
    parser.add_argument("--endpoints", default="18,19,20,21,22,23,24")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--sensor-budget", type=int, default=10)
    parser.add_argument("--mask-seed", type=int, default=4201)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-environments", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
