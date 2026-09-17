"""Replay one risk-triggered sparse calibration sweep with frozen AR calibrators."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from .data import load_ocean_npz
from .sparse_calibration import nested_sensor_indices
from .spectral_calibrator import InvariantAR1Calibrator, ar1_query, fit_stable_ar1
from .train_representation_poc import build_model, coordinates, sha256_file
from .train_spectral_calibrator import _complex_fields


def _relative_error(prediction: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    numerator = (prediction - target).abs().square().flatten(2).sum(2)
    denominator = target.abs().square().flatten(2).sum(2).clamp_min(1e-8)
    return (numerator / denominator).sqrt().cpu().numpy()


def _base_decode(model, state, visible, frequency: float) -> torch.Tensor:
    query = torch.full(
        (len(visible),),
        frequency,
        device=visible.device,
        dtype=visible.real.dtype,
    )
    return model.decode(*state[:3], visible, query, state[3], state[4])


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.calibrator_summary]
    bundle = load_ocean_npz(args.dataset)
    positions = np.flatnonzero(bundle.split == args.split)
    if args.max_environments is not None:
        positions = positions[: args.max_environments]
    index = torch.as_tensor(positions, dtype=torch.long)
    queries = tuple(map(float, args.queries.split(",")))
    endpoints = tuple(map(float, args.endpoints.split(",")))
    window = int(args.window)
    required = set(queries)
    for endpoint in endpoints:
        required.update(endpoint - offset for offset in range(window - 1, -1, -1))
    missing = required.difference(bundle.frequencies)
    if missing:
        raise ValueError(f"dataset is missing rolling calibration frequencies {sorted(missing)}")

    base_predictions = []
    priors_by_seed = []
    calibrators = []
    visible_by_seed = []
    context_by_seed = []
    normalizations = []
    checkpoint_hashes = []
    all_prior_frequencies = tuple(sorted(required))
    for calibrator_summary_path, calibrator_summary in zip(
        args.calibrator_summary, summaries, strict=True
    ):
        calibrator_path = calibrator_summary_path.parent / "best.pt"
        calibrator_checkpoint = torch.load(calibrator_path, map_location="cpu", weights_only=False)
        base_summary_path = Path(calibrator_summary["base_run_summary"])
        base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
        base_path = base_summary_path.parent / "best.pt"
        base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
        normalization = base_checkpoint["normalization"]
        visible_frequencies = tuple(map(float, base_checkpoint["visible"]))
        visible = _complex_fields(bundle, visible_frequencies, index, normalization.field_scale).to(
            device
        )
        environment = bundle.environment[index]
        environment = (
            environment - normalization.environment_mean[0]
        ) / normalization.environment_std[0]
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

        prior = torch.stack(
            [
                _base_decode(base_model, state, visible, frequency)
                for frequency in all_prior_frequencies
            ],
            dim=1,
        )
        lookup = {frequency: offset for offset, frequency in enumerate(all_prior_frequencies)}
        priors_by_seed.append((prior, lookup))
        base_predictions.append(
            torch.stack([prior[:, lookup[frequency]] for frequency in queries], dim=1)
        )
        calibrator_args = Namespace(**calibrator_summary["args"])
        calibrator = InvariantAR1Calibrator(
            len(visible_frequencies),
            bundle.environment.shape[1],
            calibration_count=window,
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
        calibrators.append(calibrator)
        visible_by_seed.append(visible)
        context_by_seed.append(context)
        normalizations.append(normalization)
        checkpoint_hashes.append(
            {
                "calibrator": sha256_file(calibrator_path),
                "base": sha256_file(base_path),
            }
        )

    field_scales = {normalization.field_scale for normalization in normalizations}
    if len(field_scales) != 1:
        raise RuntimeError("seed base models use different field scales")
    field_scale = next(iter(field_scales))
    query_target = _complex_fields(bundle, queries, index, field_scale).to(device)
    base_ensemble = torch.stack(base_predictions).mean(dim=0)
    base_error = _relative_error(base_ensemble, query_target)

    height, width = query_target.shape[-2:]
    point_count = height * width
    order = nested_sensor_indices(len(index), point_count, args.sensor_budget, args.mask_seed)
    sensor_indices = torch.as_tensor(order, dtype=torch.long, device=device)
    mask = torch.zeros(len(index), point_count, device=device)
    mask.scatter_(1, sensor_indices, 1.0)
    mask = mask.reshape(len(index), 1, height, width)

    calibrated_error = np.empty((len(index), len(endpoints), len(queries)), dtype=np.float32)
    shuffled_error = np.empty_like(calibrated_error)
    dense_oracle_error = np.empty_like(calibrated_error)
    for endpoint_index, endpoint in enumerate(endpoints):
        calibration_frequencies = tuple(endpoint - offset for offset in range(window - 1, -1, -1))
        calibration_target = _complex_fields(
            bundle, calibration_frequencies, index, field_scale
        ).to(device)
        correct_predictions = []
        shuffled_predictions = []
        for seed_index, calibrator in enumerate(calibrators):
            prior, lookup = priors_by_seed[seed_index]
            calibration_prior = torch.stack(
                [prior[:, lookup[frequency]] for frequency in calibration_frequencies],
                dim=1,
            )
            reference, pole = calibrator(
                visible_by_seed[seed_index],
                calibration_prior,
                calibration_target * mask,
                mask,
                context_by_seed[seed_index],
            )
            shuffled_reference, shuffled_pole = calibrator(
                visible_by_seed[seed_index],
                calibration_prior,
                calibration_target.roll(1, dims=0) * mask,
                mask,
                context_by_seed[seed_index],
            )
            correct_predictions.append(
                torch.stack(
                    [ar1_query(reference, pole, endpoint, frequency) for frequency in queries],
                    dim=1,
                )
            )
            shuffled_predictions.append(
                torch.stack(
                    [
                        ar1_query(shuffled_reference, shuffled_pole, endpoint, frequency)
                        for frequency in queries
                    ],
                    dim=1,
                )
            )
        calibrated_ensemble = torch.stack(correct_predictions).mean(dim=0)
        shuffled_ensemble = torch.stack(shuffled_predictions).mean(dim=0)
        calibrated_error[:, endpoint_index] = _relative_error(calibrated_ensemble, query_target)
        shuffled_error[:, endpoint_index] = _relative_error(shuffled_ensemble, query_target)
        dense_pole = fit_stable_ar1(calibration_target, calibrator_args.ar_ridge)
        dense_prediction = torch.stack(
            [
                ar1_query(calibration_target[:, -1], dense_pole, endpoint, frequency)
                for frequency in queries
            ],
            dim=1,
        )
        dense_oracle_error[:, endpoint_index] = _relative_error(dense_prediction, query_target)
        print(f"endpoint={endpoint:g} complete", flush=True)

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
        sensor_indices=order,
    )
    output = {
        "schema_version": 1,
        "status": "M2a one-trigger rolling sparse-sweep replay",
        "dataset": str(args.dataset),
        "split": args.split,
        "environment_count": len(index),
        "query_hz": queries,
        "calibration_endpoints_hz": endpoints,
        "window": window,
        "sensor_budget_per_frequency": args.sensor_budget,
        "total_complex_observations_per_trigger": args.sensor_budget * window,
        "mask_seed": args.mask_seed,
        "calibrator_summaries": [str(path) for path in args.calibrator_summary],
        "checkpoint_sha256": checkpoint_hashes,
        "base_macro_complex_rel_l2": float(base_error.mean()),
        "evaluation_note": (
            "full-field RelL2 remains evaluated at sweep frequencies; observed sensor points "
            "are not removed or force-copied, making the score conservative"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrator-summary", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-environments", type=int)
    parser.add_argument(
        "--queries",
        default="107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125",
    )
    parser.add_argument(
        "--endpoints", default="106,107,108,109,110,111,112,113,114,115,116,117,118,119,120"
    )
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--sensor-budget", type=int, default=92)
    parser.add_argument("--mask-seed", type=int, default=4201)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
