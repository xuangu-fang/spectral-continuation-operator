"""Train a continuous-frequency query operator and evaluate unseen high queries."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MultiQueryDataset, Normalization, load_ocean_npz
from .metrics import metric_vectors, relative_squared_loss
from .model import QueryUNet, parameter_count, reconstruct


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_multi_query_normalization(bundle, train, visible, all_queries):
    environment = bundle.environment[train].double()
    environment_mean = environment.mean(dim=(0, 2, 3), keepdim=True).float()
    environment_std = environment.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8).float()
    field_scales = {
        frequency: max(float(bundle.fields[frequency][train].double().square().mean().sqrt()), 1e-8)
        for frequency in visible
    }
    target_energy = torch.stack(
        [bundle.fields[frequency][train].double().square().mean() for frequency in all_queries]
    ).mean()
    return Normalization(
        environment_mean=environment_mean,
        environment_std=environment_std,
        field_scales=field_scales,
        target_scale=max(float(target_energy.sqrt()), 1e-8),
        frequency_scale=max(bundle.frequencies),
    )


@torch.no_grad()
def evaluate(model, loader, queries, scale, parameterization, device):
    model.eval()
    predictions = [[] for _ in queries]
    targets = [[] for _ in queries]
    for inputs, carrier, target, query_index in loader:
        output = reconstruct(
            model(inputs.to(device)), carrier.to(device), scale, parameterization
        ).cpu()
        for index in range(len(queries)):
            selected = query_index == index
            if selected.any():
                predictions[index].append(output[selected])
                targets[index].append(target[selected])
    result, vectors = {}, {}
    for index, frequency in enumerate(queries):
        current = metric_vectors(torch.cat(predictions[index]), torch.cat(targets[index]))
        key = f"{frequency:g}"
        result[key] = {name: float(values.mean()) for name, values in current.items()}
        vectors[key] = current
    result["macro_average"] = {
        metric: float(np.mean([result[f"{frequency:g}"][metric] for frequency in queries]))
        for metric in next(iter(result.values()))
    }
    return result, vectors


def run(args):
    started = time.perf_counter()
    bundle = load_ocean_npz(args.dataset)
    visible = tuple(map(float, args.visible.split(",")))
    train_queries = tuple(map(float, args.train_queries.split(",")))
    extrapolation_queries = tuple(map(float, args.extrapolation_queries.split(",")))
    train_positions = np.flatnonzero(bundle.split == "train")
    val_positions = np.flatnonzero(bundle.split == "val")
    if args.train_count is not None:
        train_positions = train_positions[: args.train_count]
    # Extrapolation labels do not influence normalization, optimization or selection.
    normalization = fit_multi_query_normalization(bundle, train_positions, visible, train_queries)
    train_dataset = MultiQueryDataset(
        bundle,
        train_positions,
        visible,
        train_queries,
        normalization,
        args.max_slots,
        args.field_only,
    )
    selection_dataset = MultiQueryDataset(
        bundle,
        val_positions,
        visible,
        train_queries,
        normalization,
        args.max_slots,
        args.field_only,
    )
    extrapolation_dataset = MultiQueryDataset(
        bundle,
        val_positions,
        visible,
        extrapolation_queries,
        normalization,
        args.max_slots,
        args.field_only,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        pin_memory=True,
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    extrapolation_loader = DataLoader(
        extrapolation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    seed_all(args.seed)
    device = torch.device(args.device)
    model = QueryUNet(train_dataset.input_channels, args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    selection, _ = evaluate(
        model,
        selection_loader,
        train_queries,
        normalization.target_scale,
        args.parameterization,
        device,
    )
    best_value = selection["macro_average"]["complex_rel_l2"]
    best_state = copy.deepcopy(model.state_dict())
    rows = [{"epoch": 0, "train_loss": None, "selection_complex_rel_l2": best_value}]
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, carrier, target, _ in train_loader:
            inputs, carrier, target = inputs.to(device), carrier.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = reconstruct(
                model(inputs), carrier, normalization.target_scale, args.parameterization
            )
            loss = relative_squared_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        selection, _ = evaluate(
            model,
            selection_loader,
            train_queries,
            normalization.target_scale,
            args.parameterization,
            device,
        )
        current = selection["macro_average"]["complex_rel_l2"]
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "selection_complex_rel_l2": current,
            }
        )
        if current < best_value:
            best_value = current
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:03d}/{args.epochs} train={np.mean(losses):.5f} select={current:.5f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    selection, selection_vectors = evaluate(
        model,
        selection_loader,
        train_queries,
        normalization.target_scale,
        args.parameterization,
        device,
    )
    # Crucial protocol: extrapolation fields are first evaluated only after selection is frozen.
    extrapolation, extrapolation_vectors = evaluate(
        model,
        extrapolation_loader,
        extrapolation_queries,
        normalization.target_scale,
        args.parameterization,
        device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best.pt"
    torch.save({"model": best_state, "normalization": normalization}, checkpoint)
    with (args.output_dir / "curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    vector_payload = {}
    for prefix, source in (
        ("selection", selection_vectors),
        ("extrapolation", extrapolation_vectors),
    ):
        for frequency, metrics in source.items():
            for metric, values in metrics.items():
                vector_payload[f"{prefix}_{frequency}_{metric}"] = values
    np.savez_compressed(args.output_dir / "validation_metrics.npz", **vector_payload)
    summary = {
        "schema_version": 1,
        "status": "development; extrapolation evaluated once after in-range selection",
        "visible_frequencies_hz": visible,
        "selection_query_frequencies_hz": train_queries,
        "extrapolation_query_frequencies_hz": extrapolation_queries,
        "parameterization": args.parameterization,
        "field_only": args.field_only,
        "seed": args.seed,
        "train_count": len(train_positions),
        "validation_count": len(val_positions),
        "parameters": parameter_count(model),
        "selection": selection,
        "extrapolation": extrapolation,
        "checkpoint_sha256": sha256_file(checkpoint),
        "lineage": bundle.lineage,
        "elapsed_seconds": time.perf_counter() - started,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visible", required=True)
    parser.add_argument("--train-queries", required=True)
    parser.add_argument("--extrapolation-queries", required=True)
    parser.add_argument("--parameterization", choices=("direct", "carrier_relative"), required=True)
    parser.add_argument("--max-slots", type=int, default=3)
    parser.add_argument("--field-only", action="store_true")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
