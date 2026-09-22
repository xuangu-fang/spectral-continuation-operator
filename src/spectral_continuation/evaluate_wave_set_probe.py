"""Evaluate a trained wave set operator under geometry, noise, or grid shifts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .probe_wave_set_encoder import evaluate
from .train_wave_medium_probe import _iou
from .wave_set_encoder import GROUP_LOW, FrozenMediumDecoder, WaveSetBank, WaveSetEncoder


def run(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    summary = checkpoint["summary"]
    representation = str(summary["representation"])
    loaded = np.load(args.dataset, allow_pickle=True)
    bank = WaveSetBank(
        loaded,
        representation,
        scale=np.asarray(checkpoint["field_scale"]),
        snr_db=args.snr_db,
        noise_seed=args.noise_seed,
    )
    indices = np.flatnonzero(loaded["split"].astype(str) == "test")
    if len(indices) == 0:
        indices = np.arange(len(loaded["field"]))
    encoder_state = checkpoint["encoder"]
    width = int(encoder_state["backbone.first.0.weight"].shape[0])
    dimension = int(encoder_state["project.2.weight"].shape[0])
    aggregation = str(summary.get("aggregation", "moments"))
    slots = int(summary.get("slots", 4))
    metadata_conditioning = str(summary.get("metadata_conditioning", "deep_add"))
    basis_dimension = int(summary.get("basis_dimension", 4))
    encoder = WaveSetEncoder(
        bank.channels,
        width,
        dimension,
        aggregation,
        slots,
        metadata_conditioning,
        basis_dimension,
    ).to(device)
    encoder.load_state_dict(encoder_state)
    feature_width = int(checkpoint["decoder"]["head.weight"].shape[1])
    decoder = FrozenMediumDecoder(feature_width).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    targets = torch.from_numpy((1.0 - loaded["speed"][:, None]).astype(np.float32))
    prediction, error = evaluate(
        encoder,
        decoder,
        bank,
        targets,
        indices,
        GROUP_LOW,
        device,
        args.batch_size,
    )
    target = targets[indices, 0].numpy()
    threshold = float(summary["selected_threshold"])
    iou = _iou(prediction, target, threshold)
    correlation = np.asarray(
        [
            np.corrcoef(prediction[position].ravel(), target[position].ravel())[0, 1]
            for position in range(len(indices))
        ]
    )
    group_metrics = {}
    if "boundary" in loaded.files:
        boundary = loaded["boundary"].astype(str)[indices]
        for name in sorted(set(boundary)):
            selected = boundary == name
            group_metrics[name] = {
                "count": int(selected.sum()),
                "contrast_rel_l2_mean": float(error[selected].mean()),
                "inclusion_iou_mean": float(iou[selected].mean()),
            }
    result = {
        "schema_version": 1,
        "task": "wave_set_operator_shift_evaluation",
        "label": args.label,
        "representation": representation,
        "seed": int(summary["probe_seed"]),
        "snr_db": args.snr_db,
        "systems": len(indices),
        "contrast_rel_l2_mean": float(error.mean()),
        "contrast_rel_l2_std": float(error.std()),
        "inclusion_iou_mean": float(iou.mean()),
        "pixel_correlation_mean": float(np.nanmean(correlation)),
        "groups": group_metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    noise_tag = "clean" if args.snr_db is None else f"snr{args.snr_db:g}"
    stem = f"{args.label}_{noise_tag}_{representation}_seed{result['seed']}"
    (args.output / f"{stem}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        args.output / f"{stem}_predictions.npz",
        system=indices,
        prediction=prediction,
        target=target,
        error=error,
        iou=iou,
        correlation=correlation,
    )
    with (args.output / f"{stem}_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("system", "contrast_rel_l2", "inclusion_iou", "pixel_correlation"),
        )
        writer.writeheader()
        for position, system in enumerate(indices):
            writer.writerow(
                {
                    "system": int(system),
                    "contrast_rel_l2": float(error[position]),
                    "inclusion_iou": float(iou[position]),
                    "pixel_correlation": float(correlation[position]),
                }
            )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--snr-db", type=float)
    parser.add_argument("--noise-seed", type=int, default=20260922)
    parser.add_argument("--batch-size", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
