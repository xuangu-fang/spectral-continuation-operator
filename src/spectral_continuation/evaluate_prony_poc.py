"""Evaluate the finite-rank Prony spectral-state diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .representation import PronyFrequencyOperator
from .representation_data import (
    FrequencySetDataset,
    fit_representation_normalization,
    load_representation_bundle,
)
from .train_representation_poc import evaluate


def run(args):
    visible = tuple(map(float, args.visible.split(",")))
    selection_queries = tuple(map(float, args.selection_queries.split(",")))
    test_queries = tuple(map(float, args.test_queries.split(",")))
    all_frequencies = tuple(sorted(set(visible + selection_queries + test_queries)))
    bundle = load_representation_bundle(
        args.domain,
        args.dataset,
        train_count=args.synthetic_train,
        validation_count=args.synthetic_validation,
        height=args.synthetic_height,
        width=args.synthetic_width,
        frequencies=all_frequencies,
        seed=args.data_seed,
    )
    train = np.flatnonzero(bundle.split == "train")
    validation = np.flatnonzero(bundle.split == "val")
    normalization = fit_representation_normalization(
        bundle, train, visible, selection_queries
    )
    device = torch.device(args.device)
    model = PronyFrequencyOperator(visible, order=args.order, ridge=args.ridge).to(device)

    def run_queries(queries, current_bundle=bundle):
        dataset = FrequencySetDataset(
            current_bundle,
            validation,
            visible,
            queries,
            normalization,
            field_only=True,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        return evaluate(model, loader, queries, normalization, device)[0]

    selection = run_queries(selection_queries)
    extrapolation = run_queries(test_queries)
    cross_resolution = None
    if args.domain == "synthetic" and args.cross_resolution_factor > 1:
        high_bundle = load_representation_bundle(
            "synthetic",
            None,
            train_count=args.synthetic_train,
            validation_count=args.synthetic_validation,
            height=args.synthetic_height * args.cross_resolution_factor,
            width=args.synthetic_width * args.cross_resolution_factor,
            frequencies=all_frequencies,
            seed=args.data_seed,
        )
        cross_resolution = run_queries(test_queries, high_bundle)
    summary = {
        "schema_version": 1,
        "status": "analytic representation diagnostic; development data only",
        "domain": args.domain,
        "model": "prony_frequency_state",
        "visible_frequencies_hz": visible,
        "selection_query_frequencies_hz": selection_queries,
        "test_query_frequencies_hz": test_queries,
        "order": args.order,
        "ridge": args.ridge,
        "selection": selection,
        "extrapolation": extrapolation,
        "cross_resolution_extrapolation": cross_resolution,
        "lineage": bundle.lineage,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("synthetic", "ocean"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--visible", required=True)
    parser.add_argument("--selection-queries", required=True)
    parser.add_argument("--test-queries", required=True)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-seed", type=int, default=20260915)
    parser.add_argument("--synthetic-train", type=int, default=32)
    parser.add_argument("--synthetic-validation", type=int, default=128)
    parser.add_argument("--synthetic-height", type=int, default=32)
    parser.add_argument("--synthetic-width", type=int, default=64)
    parser.add_argument("--cross-resolution-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
