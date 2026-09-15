"""Evaluate frozen query checkpoints once on a registered dense frequency horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MultiQueryDataset, load_ocean_npz
from .model import QueryUNet
from .train_query_operator import evaluate


def verify_identity(source, dense) -> None:
    if not np.array_equal(source.ids, dense.ids) or not np.array_equal(source.split, dense.split):
        raise RuntimeError("dense-horizon environment IDs or splits differ from source data")
    if not torch.equal(source.environment, dense.environment):
        raise RuntimeError("dense-horizon environment channels differ from source data")
    shared = sorted(set(source.frequencies).intersection(dense.frequencies))
    for frequency in shared:
        if not torch.equal(source.fields[frequency], dense.fields[frequency]):
            raise RuntimeError(f"shared field at {frequency:g} Hz is not bitwise identical")


def run(args):
    source_paths = sorted(args.run_root.glob(args.run_glob))
    if not source_paths:
        raise RuntimeError("no frozen source summaries matched")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in source_paths]
    datasets = {summary["args"]["dataset"] for summary in summaries}
    if len(datasets) != 1:
        raise RuntimeError("frozen checkpoints do not share one source dataset")
    source_bundle = load_ocean_npz(Path(datasets.pop()))
    dense_bundle = load_ocean_npz(args.dataset)
    verify_identity(source_bundle, dense_bundle)
    queries = tuple(map(float, args.queries.split(",")))
    validation = np.flatnonzero(dense_bundle.split == "val")
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path, source_summary in zip(source_paths, summaries, strict=True):
        source_run = path.parent
        checkpoint = torch.load(source_run / "best.pt", map_location="cpu", weights_only=False)
        normalization = checkpoint["normalization"]
        dataset = MultiQueryDataset(
            dense_bundle,
            validation,
            source_summary["visible_frequencies_hz"],
            queries,
            normalization,
            int(source_summary["args"]["max_slots"]),
            bool(source_summary["field_only"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        model = QueryUNet(dataset.input_channels, int(source_summary["args"]["width"])).to(device)
        model.load_state_dict(checkpoint["model"])
        extrapolation, vectors = evaluate(
            model,
            loader,
            queries,
            normalization.target_scale,
            source_summary["parameterization"],
            device,
        )
        output_dir = args.output_root / source_run.name
        output_dir.mkdir(parents=True, exist_ok=True)
        vector_payload = {}
        with np.load(source_run / "validation_metrics.npz", allow_pickle=False) as source_vectors:
            for key in source_vectors.files:
                if key.startswith("selection_"):
                    vector_payload[key] = np.asarray(source_vectors[key])
        for frequency, current in vectors.items():
            for metric, values in current.items():
                vector_payload[f"extrapolation_{frequency}_{metric}"] = values
        np.savez_compressed(output_dir / "validation_metrics.npz", **vector_payload)
        output = {
            **source_summary,
            "status": (
                "frozen checkpoint; dense-horizon labels evaluated once after 90/100-Hz selection"
            ),
            "extrapolation_query_frequencies_hz": queries,
            "extrapolation": extrapolation,
            "lineage": {
                "source_checkpoint_summary": str(path),
                "source_dataset": source_summary["lineage"],
                "dense_dataset": dense_bundle.lineage,
                "identity_check": "IDs, splits, active environment channels and all shared fields bitwise identical",
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        print(f"evaluated {source_run.name}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-glob", default="fine_query_formal_*/summary.json")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--queries", default="105,110,115,120,125,150,175,200")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
