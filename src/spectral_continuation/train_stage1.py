"""Train one matched stage-1 spectral-continuation variant."""

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

from .data import (
    SpectralQueryDataset,
    fit_normalization,
    load_ocean_npz,
    load_wavebench_development,
)
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
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, loader, scale, parameterization, device):
    model.eval()
    predictions, targets = [], []
    for inputs, carrier, target in loader:
        prediction = reconstruct(
            model(inputs.to(device)), carrier.to(device), scale, parameterization
        )
        predictions.append(prediction.cpu())
        targets.append(target)
    vectors = metric_vectors(torch.cat(predictions), torch.cat(targets))
    return {name: float(values.mean()) for name, values in vectors.items()}, vectors


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    frequencies = tuple(map(float, args.frequencies.split(",")))
    visible = tuple(map(float, args.visible.split(",")))
    if args.domain == "wavebench":
        bundle = load_wavebench_development(args.source_run, frequencies, args.train_count)
    else:
        bundle = load_ocean_npz(args.dataset)
    if any(frequency not in bundle.fields for frequency in (*visible, args.target)):
        raise ValueError("visible or target frequency is absent from the dataset")
    train_positions = np.flatnonzero(bundle.split == "train")
    val_positions = np.flatnonzero(bundle.split == "val")
    if args.train_count is not None and args.domain == "ocean":
        train_positions = train_positions[: args.train_count]
    normalization = fit_normalization(bundle, train_positions, visible, args.target)
    train_dataset = SpectralQueryDataset(
        bundle, train_positions, visible, args.target, normalization, args.max_slots
    )
    val_dataset = SpectralQueryDataset(
        bundle, val_positions, visible, args.target, normalization, args.max_slots
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
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
    best_state = copy.deepcopy(model.state_dict())
    best_metrics, best_vectors = evaluate(
        model, val_loader, normalization.target_scale, args.parameterization, device
    )
    rows = [{"epoch": 0, "train_loss": None, **best_metrics, "lr": args.lr}]
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, carrier, target in train_loader:
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
        metrics, vectors = evaluate(
            model, val_loader, normalization.target_scale, args.parameterization, device
        )
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **metrics,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        if metrics["complex_rel_l2"] < best_metrics["complex_rel_l2"]:
            best_state, best_metrics, best_vectors = (
                copy.deepcopy(model.state_dict()),
                metrics,
                vectors,
            )
        print(
            f"epoch={epoch:03d}/{args.epochs} train={np.mean(losses):.5f} "
            f"val={metrics['complex_rel_l2']:.5f}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best.pt"
    torch.save(
        {
            "model": best_state,
            "input_channels": train_dataset.input_channels,
            "width": args.width,
            "parameterization": args.parameterization,
            "visible_frequencies": visible,
            "target_frequency": args.target,
            "target_scale": normalization.target_scale,
        },
        checkpoint,
    )
    write_csv(args.output_dir / "curve.csv", rows)
    np.savez_compressed(args.output_dir / "validation_metrics.npz", **best_vectors)
    summary = {
        "schema_version": 1,
        "status": "development validation only",
        "domain": args.domain,
        "visible_frequencies_hz": visible,
        "target_frequency_hz": args.target,
        "parameterization": args.parameterization,
        "model_seed": args.seed,
        "train_count": len(train_positions),
        "validation_count": len(val_positions),
        "model_parameters": parameter_count(model),
        "best_validation": best_metrics,
        "lineage": bundle.lineage,
        "normalization": {
            "field_scales": normalization.field_scales,
            "target_scale": normalization.target_scale,
            "frequency_scale": normalization.frequency_scale,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("wavebench", "ocean"), required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequencies", default="10,15,20,40")
    parser.add_argument("--visible", required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--parameterization", choices=("direct", "carrier_relative"), required=True)
    parser.add_argument("--max-slots", type=int, default=3)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.domain == "wavebench" and args.source_run is None:
        parser.error("--source-run is required for WaveBench")
    if args.domain == "ocean" and args.dataset is None:
        parser.error("--dataset is required for Ocean")
    return args


if __name__ == "__main__":
    run(parse_args())
