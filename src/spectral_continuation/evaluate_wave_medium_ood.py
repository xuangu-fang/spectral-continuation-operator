"""Evaluate frozen medium probes on an unseen scatterer geometry family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train_wave_medium_probe import MediumUNet, _input_tensor, _iou
from .train_wave_operator_contrastive import ContrastiveFieldBank


def evaluate(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = np.load(args.training_dataset, allow_pickle=True)
    train = np.flatnonzero(base["split"].astype(str) == "train")
    scale = np.sqrt(np.mean(np.abs(base["field"][train]) ** 2, axis=(0, 1, 3, 4)))
    loaded = np.load(args.ood_dataset, allow_pickle=True)
    split = loaded["split"].astype(str)
    indices = np.flatnonzero(split == "test")
    field = loaded["field"][indices].copy()
    if args.snr_db is not None:
        rng = np.random.default_rng(args.noise_seed)
        signal_rms = np.sqrt(np.mean(np.abs(field) ** 2, axis=(-2, -1), keepdims=True))
        noise = rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
        noise = noise / np.sqrt(2.0)
        field += noise * signal_rms * 10.0 ** (-args.snr_db / 20.0)
    ood = {
        "field": field,
        "speed": loaded["speed"][indices],
        "sources": loaded["sources"],
        "frequencies": loaded["frequencies"],
        "split": np.asarray(["test"] * len(indices)),
    }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    representation = checkpoint["summary"]["representation"]
    source_count = int(checkpoint["summary"].get("source_count", 6))
    bank = ContrastiveFieldBank(ood, representation, scale=scale)
    inputs = _input_tensor(bank, source_count)
    target = (1.0 - ood["speed"]).astype(np.float32)
    model = MediumUNet(inputs.shape[1], int(checkpoint["args"]["width"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs), args.batch_size):
            predictions.append(model(inputs[start : start + args.batch_size].to(device)).cpu())
    prediction = torch.cat(predictions)[:, 0].numpy()
    numerator = np.linalg.norm((prediction - target).reshape(len(target), -1), axis=1)
    denominator = np.linalg.norm(target.reshape(len(target), -1), axis=1)
    error = numerator / np.maximum(denominator, 1e-8)
    threshold = float(checkpoint["summary"]["selected_threshold"])
    iou = _iou(prediction, target, threshold)
    correlation = np.asarray(
        [np.corrcoef(prediction[index].ravel(), target[index].ravel())[0, 1]
         for index in range(len(target))]
    )
    summary = {
        "representation": representation,
        "seed": int(checkpoint["summary"]["seed"]),
        "source_count": source_count,
        "evaluation_set": args.label,
        "snr_db": args.snr_db,
        "contrast_rel_l2_mean": float(error.mean()),
        "contrast_rel_l2_std": float(error.std()),
        "inclusion_iou_mean": float(iou.mean()),
        "pixel_correlation_mean": float(np.nanmean(correlation)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    noise_tag = "clean" if args.snr_db is None else f"snr{args.snr_db:g}"
    stem = f"{args.label}_{noise_tag}_{representation}_seed{summary['seed']}"
    (args.output / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output / f"{stem}_predictions.npz",
        prediction=prediction, target=target, error=error, iou=iou,
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--ood-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--label", default="ood_ellipses")
    parser.add_argument("--snr-db", type=float)
    parser.add_argument("--noise-seed", type=int, default=20260922)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
