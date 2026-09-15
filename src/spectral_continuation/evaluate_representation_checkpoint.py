"""Evaluate a frozen representation checkpoint on new continuous queries."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FieldBundle, load_ocean_npz
from .evaluate_horizon import verify_identity
from .representation_data import FrequencyBandDataset
from .train_representation_poc import build_model, evaluate, sha256_file


def _merged_bundle(source: FieldBundle, query: FieldBundle) -> FieldBundle:
    verify_identity(source, query)
    fields = dict(source.fields)
    for frequency, value in query.fields.items():
        if frequency in fields and not torch.equal(fields[frequency], value):
            raise RuntimeError(f"shared field at {frequency:g} Hz differs")
        fields[frequency] = value
    return FieldBundle(
        fields=fields,
        environment=source.environment,
        ids=source.ids,
        split=source.split,
        frequencies=tuple(sorted(fields)),
        lineage={"source": source.lineage, "query": query.lineage},
    )


def run(args) -> None:
    source_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    checkpoint_path = args.run_summary.parent / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_bundle = load_ocean_npz(args.source_dataset)
    query_bundle = (
        load_ocean_npz(args.query_dataset)
        if args.query_dataset is not None
        else source_bundle
    )
    bundle = _merged_bundle(source_bundle, query_bundle)
    queries = tuple(map(float, args.queries.split(",")))
    missing = set(queries).difference(bundle.frequencies)
    if missing:
        raise ValueError(f"query dataset is missing frequencies: {sorted(missing)}")
    visible = tuple(map(float, checkpoint["visible"]))
    validation = np.flatnonzero(bundle.split == "val")
    dataset = FrequencyBandDataset(
        bundle,
        validation,
        visible,
        queries,
        checkpoint["normalization"],
        field_only=bool(source_summary["args"].get("field_only", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model_args = Namespace(**source_summary["args"])
    model = build_model(
        model_args,
        visible,
        dataset.environment_channels,
        checkpoint["normalization"].frequency_scale,
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    metrics, vectors = evaluate(
        model, loader, queries, checkpoint["normalization"], torch.device(args.device)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        f"query_{frequency}_{metric}": values
        for frequency, current in vectors.items()
        for metric, values in current.items()
    }
    np.savez_compressed(args.output_dir / "validation_metrics.npz", **payload)
    output = {
        "schema_version": 1,
        "status": "frozen representation checkpoint evaluated on separately generated queries",
        "source_summary": str(args.run_summary),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "queries_hz": queries,
        "metrics": metrics,
        "lineage": bundle.lineage,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--query-dataset", type=Path)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
