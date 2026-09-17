"""Extract target-free horizon-risk features from frozen seed ensembles."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_ocean_npz
from .horizon import complex_relative_error, target_free_ensemble_features
from .representation_data import FrequencyBandDataset, load_representation_bundle
from .train_representation_poc import build_model, coordinates, sha256_file


def _same_normalization(left, right) -> bool:
    return (
        left.field_scale == right.field_scale
        and left.frequency_scale == right.frequency_scale
        and torch.equal(left.environment_mean, right.environment_mean)
        and torch.equal(left.environment_std, right.environment_std)
    )


def _load_bundle(summary: dict, queries: tuple[float, ...]):
    run_args = summary["args"]
    visible = tuple(map(float, summary["visible_frequencies_hz"]))
    train_queries = tuple(map(float, summary["selection_query_frequencies_hz"]))
    frequencies = tuple(sorted(set(visible + train_queries + queries)))
    if run_args["domain"] == "ocean":
        return load_ocean_npz(Path(run_args["dataset"]))
    return load_representation_bundle(
        "synthetic",
        None,
        train_count=int(run_args["synthetic_train"]),
        validation_count=int(run_args["synthetic_validation"]),
        height=int(run_args["synthetic_height"]),
        width=int(run_args["synthetic_width"]),
        frequencies=frequencies,
        seed=int(run_args["data_seed"]),
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.run_summary]
    first = summaries[0]
    models = {summary["model"] for summary in summaries}
    visible_sets = {tuple(summary["visible_frequencies_hz"]) for summary in summaries}
    domains = {summary["args"]["domain"] for summary in summaries}
    if len(models) != 1 or len(visible_sets) != 1 or len(domains) != 1:
        raise ValueError("seed ensemble must share model, visible band, and domain")
    queries = (
        tuple(map(float, args.queries.split(",")))
        if args.queries
        else tuple(map(float, first["test_query_frequencies_hz"]))
    )
    visible = tuple(map(float, first["visible_frequencies_hz"]))
    reference_frequency = (
        float(args.reference_frequency)
        if args.reference_frequency is not None
        else float(first["selection_query_frequencies_hz"][-1])
    )
    bundle = _load_bundle(first, queries)
    missing = set(visible + queries).difference(bundle.frequencies)
    if missing:
        raise ValueError(f"bundle is missing frequencies {sorted(missing)}")
    positions = np.flatnonzero(bundle.split == "val")
    if args.max_environments is not None:
        positions = positions[: args.max_environments]
    device = torch.device(args.device)

    seed_predictions = []
    target_batches = []
    visible_batches = []
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
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        model_args = Namespace(**summary["args"])
        model = build_model(
            model_args,
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
            query = torch.full(
                (fields.shape[0],), queries[0], dtype=fields.real.dtype, device=device
            )
            context = coordinates(environment, query, normalization.frequency_scale)
            if hasattr(model, "forward_queries"):
                prediction = model.forward_queries(fields, context, queries)
            else:
                prediction = torch.stack(
                    [
                        model(
                            fields,
                            coordinates(
                                environment,
                                torch.full_like(query, frequency),
                                normalization.frequency_scale,
                            ),
                            torch.full_like(query, frequency),
                        )
                        for frequency in queries
                    ],
                    dim=1,
                )
            prediction_batches.append(prediction.cpu())
            if seed_index == 0:
                target_batches.append(target.cpu())
                visible_batches.append(fields.cpu())
        seed_predictions.append(torch.cat(prediction_batches).numpy())
        checkpoint_hashes.append(sha256_file(checkpoint_path))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = np.stack(seed_predictions, axis=0)
    targets = torch.cat(target_batches).numpy()
    visible_fields = torch.cat(visible_batches).numpy()
    ensemble = predictions.mean(axis=0)
    features, feature_names = target_free_ensemble_features(
        predictions,
        visible_fields[:, -1],
        np.asarray(queries),
        reference_frequency,
    )
    ensemble_error = complex_relative_error(ensemble, targets)
    seed_error = np.stack(
        [complex_relative_error(prediction, targets) for prediction in predictions], axis=0
    )
    mean_seed_error = seed_error.mean(axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "horizon_features.npz",
        environment_ids=np.asarray(bundle.ids[positions]).astype(str),
        positions=positions,
        query_frequencies_hz=np.asarray(queries),
        feature_names=np.asarray(feature_names),
        features=features,
        ensemble_complex_rel_l2=ensemble_error,
        per_seed_complex_rel_l2=seed_error,
        mean_seed_complex_rel_l2=mean_seed_error,
    )
    output = {
        "schema_version": 1,
        "status": "frozen target-free seed-ensemble horizon features",
        "domain": next(iter(domains)),
        "model": next(iter(models)),
        "run_summaries": [str(path) for path in args.run_summary],
        "checkpoint_sha256": checkpoint_hashes,
        "environment_count": len(positions),
        "queries_hz": queries,
        "reference_frequency_hz": reference_frequency,
        "feature_names": feature_names,
        "target_definition": "raw complex RelL2 of pointwise complex seed-ensemble mean",
        "aggregate": {
            f"{frequency:g}": {
                "ensemble_complex_rel_l2": float(ensemble_error[:, index].mean()),
                "mean_seed_complex_rel_l2": float(mean_seed_error[:, index].mean()),
            }
            for index, frequency in enumerate(queries)
        },
        "lineage": bundle.lineage,
        "leakage_note": "features use only visible fields and frozen predictions; query targets only form labels",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, action="append", required=True)
    parser.add_argument("--queries")
    parser.add_argument("--reference-frequency", type=float)
    parser.add_argument("--max-environments", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
