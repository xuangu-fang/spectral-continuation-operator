"""Re-evaluate a frozen cross-domain query checkpoint on nested episodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .cross_domain_query_audit import _fit_subspaces
from .cross_domain_response_data import load_cross_domain_archive
from .train_cross_domain_query import (
    QueryNormalizer,
    ReceiverQueryOperator,
    TorchSubspacePredictor,
    _summary,
    evaluate,
)


def run(checkpoint_path: Path, dataset: Path, output: Path, device_name: str) -> dict:
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    training_args = checkpoint["args"]
    archive = load_cross_domain_archive(dataset)
    normalizer = QueryNormalizer(
        archive.responses, np.flatnonzero(archive.split == "train")
    )
    predictor = TorchSubspacePredictor(
        _fit_subspaces(archive, max_rank=32), archive.responses.shape[-1], device
    )
    model = ReceiverQueryOperator(
        archive.receivers,
        int(training_args["width"]),
        periodic=training_args["receiver_topology"] == "periodic",
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    rank_by_budget = {int(key): int(value) for key, value in checkpoint["rank_by_budget"].items()}
    rows = evaluate(
        model,
        archive,
        normalizer,
        predictor,
        rank_by_budget,
        "test",
        tuple(sorted(rank_by_budget)),
        3,
        20260924,
        device,
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "test_system_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": str(dataset.resolve()),
        "seed": int(training_args["seed"]),
        "checkpoint_step": int(checkpoint["step"]),
        "episode_protocol": "three frozen nested context orders per target; system macro-average",
        "test": _summary(rows),
    }
    (output / "nested_test_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    training_summary_path = output / "summary.json"
    if training_summary_path.exists():
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        training_summary["test"] = summary["test"]
        training_summary["final_evaluation_protocol"] = summary["episode_protocol"]
        training_summary_path.write_text(
            json.dumps(training_summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.dataset, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
