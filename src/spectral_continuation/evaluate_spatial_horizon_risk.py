"""Evaluate patch-level target-free failure maps from frozen seed predictions."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_horizon_features import _load_bundle, _same_normalization
from .horizon import reliable_horizon
from .representation_data import FrequencyBandDataset
from .risk import (
    RidgeLogisticRegression,
    average_precision,
    expected_calibration_error,
    prefix_horizon_from_risk,
    roc_auc,
    spearman_correlation,
)
from .train_representation_poc import build_model, coordinates, sha256_file

MODEL_FEATURES = {
    "distance_only": ("query_offset_hz",),
    "global_disagreement": ("query_offset_hz", "global_seed_disagreement"),
    "local_disagreement": ("query_offset_hz", "local_seed_disagreement"),
    "local_combo": (
        "query_offset_hz",
        "global_seed_disagreement",
        "local_seed_disagreement",
        "local_amplitude_disagreement",
        "local_phase_disagreement",
        "predicted_patch_energy_ratio",
        "patch_y",
        "patch_x",
    ),
}


def _patch_sum(values: np.ndarray, patch_grid: tuple[int, int]) -> np.ndarray:
    rows, columns = patch_grid
    height, width = values.shape[-2:]
    if height % rows or width % columns:
        raise ValueError("field shape must be divisible by patch grid")
    reshaped = values.reshape(*values.shape[:-2], rows, height // rows, columns, width // columns)
    return reshaped.sum(axis=(-3, -1))


def _spatial_features(
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: np.ndarray,
    reference_frequency: float,
    patch_grid: tuple[int, int],
    energy_floor_fraction: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return target-free patch features and target-derived local error labels."""
    ensemble = predictions.mean(axis=0)
    patch_count = patch_grid[0] * patch_grid[1]
    target_patch_energy = _patch_sum(np.abs(targets) ** 2, patch_grid)
    target_total_energy = np.sum(np.abs(targets) ** 2, axis=(-2, -1))
    target_floor = energy_floor_fraction * target_total_energy[..., None, None] / patch_count
    error_patch_energy = _patch_sum(np.abs(ensemble - targets) ** 2, patch_grid)
    local_error = np.sqrt(error_patch_energy / np.maximum(target_patch_energy, target_floor))

    predicted_patch_energy = _patch_sum(np.abs(ensemble) ** 2, patch_grid)
    predicted_total_energy = np.sum(np.abs(ensemble) ** 2, axis=(-2, -1))
    predicted_floor = energy_floor_fraction * predicted_total_energy[..., None, None] / patch_count
    seed_error_energy = _patch_sum(np.abs(predictions - ensemble[None]) ** 2, patch_grid).mean(
        axis=0
    )
    local_disagreement = np.sqrt(
        seed_error_energy / np.maximum(predicted_patch_energy, predicted_floor)
    )
    amplitude_error_energy = _patch_sum(
        (np.abs(predictions) - np.abs(ensemble)[None]) ** 2, patch_grid
    ).mean(axis=0)
    amplitude_disagreement = np.sqrt(
        amplitude_error_energy / np.maximum(predicted_patch_energy, predicted_floor)
    )
    cross = predictions * np.conj(ensemble)[None]
    phase_cosine = np.real(cross) / np.maximum(np.abs(cross), 1e-12)
    phase_numerator = _patch_sum(np.abs(ensemble)[None] * (1.0 - phase_cosine), patch_grid).mean(
        axis=0
    )
    phase_denominator = _patch_sum(np.abs(ensemble), patch_grid)
    phase_disagreement = phase_numerator / np.maximum(phase_denominator, 1e-12)

    global_seed_error = np.sum(np.abs(predictions - ensemble[None]) ** 2, axis=(-2, -1)).mean(
        axis=0
    )
    global_disagreement = np.sqrt(global_seed_error / np.maximum(predicted_total_energy, 1e-12))
    patch_energy_ratio = (
        patch_count
        * predicted_patch_energy
        / np.maximum(predicted_total_energy[..., None, None], 1e-12)
    )
    rows, columns = patch_grid
    patch_y = np.broadcast_to(np.linspace(-1.0, 1.0, rows)[None, None, :, None], local_error.shape)
    patch_x = np.broadcast_to(
        np.linspace(-1.0, 1.0, columns)[None, None, None, :], local_error.shape
    )
    offset = np.broadcast_to(
        (frequencies - reference_frequency)[None, :, None, None], local_error.shape
    )
    global_feature = np.broadcast_to(global_disagreement[..., None, None], local_error.shape)
    features = np.stack(
        (
            offset,
            global_feature,
            local_disagreement,
            amplitude_disagreement,
            phase_disagreement,
            patch_energy_ratio,
            patch_y,
            patch_x,
        ),
        axis=-1,
    )
    names = (
        "query_offset_hz",
        "global_seed_disagreement",
        "local_seed_disagreement",
        "local_amplitude_disagreement",
        "local_phase_disagreement",
        "predicted_patch_energy_ratio",
        "patch_y",
        "patch_x",
    )
    return features, local_error, names


def _within_group_auc(labels: np.ndarray, scores: np.ndarray, axis: str) -> float:
    values = []
    if axis == "frequency":
        groups = (
            (labels[:, index].ravel(), scores[:, index].ravel()) for index in range(labels.shape[1])
        )
    elif axis == "field":
        groups = (
            (labels[environment, query].ravel(), scores[environment, query].ravel())
            for environment in range(labels.shape[0])
            for query in range(labels.shape[1])
        )
    else:
        raise ValueError(f"unknown grouping axis {axis}")
    for current_labels, current_scores in groups:
        if 0 < current_labels.sum() < len(current_labels):
            values.append(roc_auc(current_labels, current_scores))
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.run_summary]
    first = summaries[0]
    queries = (
        tuple(map(float, args.queries.split(",")))
        if args.queries
        else tuple(map(float, first["test_query_frequencies_hz"]))
    )
    visible = tuple(map(float, first["visible_frequencies_hz"]))
    reference_frequency = (
        args.reference_frequency
        if args.reference_frequency is not None
        else float(first["selection_query_frequencies_hz"][-1])
    )
    bundle = _load_bundle(first, queries)
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
            queries,
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
            anchor = torch.full((len(fields),), queries[0], dtype=fields.real.dtype, device=device)
            context = coordinates(environment, anchor, normalization.frequency_scale)
            prediction_batches.append(model.forward_queries(fields, context, queries).cpu().numpy())
            if seed_index == 0:
                target_batches.append(target.numpy())
        seed_predictions.append(np.concatenate(prediction_batches))
        checkpoint_hashes.append(sha256_file(checkpoint_path))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = np.stack(seed_predictions)
    targets = np.concatenate(target_batches)
    features, local_error, feature_names = _spatial_features(
        predictions,
        targets,
        np.asarray(queries),
        reference_frequency,
        tuple(args.patch_grid),
        args.energy_floor_fraction,
    )
    labels = local_error > args.error_threshold
    fold_ids = np.arange(len(positions)) % args.folds
    results = {}
    probabilities_by_method = {}
    for model_name, names in MODEL_FEATURES.items():
        indices = [feature_names.index(name) for name in names]
        probabilities = np.empty(local_error.shape, dtype=float)
        for fold in range(args.folds):
            train = fold_ids != fold
            test = fold_ids == fold
            model = RidgeLogisticRegression(ridge=args.ridge)
            model.fit(
                features[train][..., indices].reshape(-1, len(indices)), labels[train].ravel()
            )
            probabilities[test] = model.predict_proba(
                features[test][..., indices].reshape(-1, len(indices))
            ).reshape(local_error[test].shape)
        flat_label = labels.ravel()
        flat_error = local_error.ravel()
        flat_probability = probabilities.ravel()
        accepted = flat_probability <= args.safe_probability
        patch_error_by_frequency = local_error.transpose(0, 2, 3, 1).reshape(-1, len(queries))
        patch_probability_by_frequency = probabilities.transpose(0, 2, 3, 1).reshape(
            -1, len(queries)
        )
        true_horizon = reliable_horizon(
            patch_error_by_frequency,
            np.asarray(queries),
            reference_frequency,
            args.error_threshold,
        )[0]
        predicted_horizon = prefix_horizon_from_risk(
            patch_probability_by_frequency,
            np.asarray(queries),
            reference_frequency,
            args.safe_probability,
        )
        top_count = max(1, int(0.2 * len(flat_label)))
        results[model_name] = {
            "features": names,
            "failure_auroc": roc_auc(flat_label, flat_probability),
            "failure_auprc": average_precision(flat_label, flat_probability),
            "error_spearman": spearman_correlation(flat_error, flat_probability),
            "brier_score": float(np.mean((flat_probability - flat_label) ** 2)),
            "expected_calibration_error": expected_calibration_error(flat_label, flat_probability),
            "within_frequency_auroc": _within_group_auc(labels, probabilities, "frequency"),
            "within_environment_frequency_spatial_auroc": _within_group_auc(
                labels, probabilities, "field"
            ),
            "accepted_patch_fraction_at_safe_probability": float(accepted.mean()),
            "empirical_safe_rate_at_safe_probability": (
                float(1.0 - flat_label[accepted].mean()) if accepted.any() else float("nan")
            ),
            "top_risk_20_percent_failure_recall": float(
                flat_label[np.argsort(flat_probability)[-top_count:]].sum()
                / max(1, flat_label.sum())
            ),
            "patch_horizon_mae_hz": float(np.mean(np.abs(predicted_horizon - true_horizon))),
            "patch_unsafe_overreach_rate": float(np.mean(predicted_horizon > true_horizon)),
        }
        probabilities_by_method[model_name] = probabilities

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "spatial_risk_predictions.npz",
        environment_ids=np.asarray(bundle.ids[positions]).astype(str),
        query_frequencies_hz=np.asarray(queries),
        patch_grid=np.asarray(args.patch_grid),
        local_error=local_error,
        labels=labels,
        fold_ids=fold_ids,
        **{f"{name}_failure_probability": value for name, value in probabilities_by_method.items()},
    )
    output = {
        "schema_version": 1,
        "status": "environment-fold cross-validated target-free patch risk map",
        "domain": first["args"]["domain"],
        "model": first["model"],
        "run_summaries": [str(path) for path in args.run_summary],
        "checkpoint_sha256": checkpoint_hashes,
        "environment_count": len(positions),
        "queries_hz": queries,
        "reference_frequency_hz": reference_frequency,
        "patch_grid": args.patch_grid,
        "error_threshold": args.error_threshold,
        "energy_floor_fraction": args.energy_floor_fraction,
        "local_error_definition": (
            "patch complex RelL2 with denominator floored by energy_floor_fraction times "
            "the uniform per-patch target energy"
        ),
        "results": results,
        "leakage_note": (
            "risk features use only frozen seed predictions, query offsets and patch coordinates; "
            "query targets only define evaluation labels"
        ),
    }
    (args.output_dir / "spatial_risk_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    method = "local_combo"
    count = labels.reshape(len(positions), len(queries), -1).sum(axis=2)
    usable = np.argwhere((count > 0) & (count < np.prod(args.patch_grid)))
    if len(usable):
        environment, query = usable[len(usable) // 2]
    else:
        environment, query = 0, len(queries) // 2
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True)
    images = [
        (local_error[environment, query], "True local RelL2", "magma", None, None),
        (
            probabilities_by_method[method][environment, query],
            "Predicted failure risk",
            "viridis",
            0.0,
            1.0,
        ),
        (labels[environment, query], "Failure mask", "gray_r", 0.0, 1.0),
    ]
    for axis, (value, title, cmap, lower, upper) in zip(axes, images, strict=True):
        image = axis.imshow(value, cmap=cmap, vmin=lower, vmax=upper, aspect="auto")
        axis.set(title=title, xlabel="Patch x", ylabel="Patch y")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(
        f"{first['args']['domain']} env {bundle.ids[positions[environment]]}, "
        f"query {queries[query]:g} Hz"
    )
    figure.savefig(args.output_dir / "spatial_risk_example.png", dpi=250)
    figure.savefig(args.output_dir / "spatial_risk_example.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries")
    parser.add_argument("--reference-frequency", type=float)
    parser.add_argument("--patch-grid", nargs=2, type=int, default=(4, 8))
    parser.add_argument("--error-threshold", type=float, default=0.5)
    parser.add_argument("--safe-probability", type=float, default=0.1)
    parser.add_argument("--energy-floor-fraction", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-environments", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
