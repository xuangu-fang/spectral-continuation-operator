"""Frozen spatial probes for self-supervised wave-system set encoders."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .train_wave_medium_probe import _iou
from .wave_set_encoder import (
    GROUP_HIGH,
    GROUP_LOW,
    FrozenMediumDecoder,
    ViewBudget,
    WaveSetBank,
    WaveSetEncoder,
    sample_rectangular_views,
)


def _fixed_views(
    sources: tuple[int, ...], frequencies: tuple[int, ...], budget: ViewBudget
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (source, frequency)
        for source in sources[: budget.source_count]
        for frequency in frequencies[-budget.frequency_count :]
    )


def _sample_training_views(
    rng: np.random.Generator,
    sources: tuple[int, ...],
    frequencies: tuple[int, ...],
    full_context_probability: float,
) -> tuple[tuple[int, int], ...]:
    if rng.random() < full_context_probability:
        source_count = len(sources)
        frequency_count = len(frequencies)
    else:
        source_count = int(rng.integers(1, len(sources) + 1))
        frequency_count = int(rng.integers(1, len(frequencies) + 1))
    return sample_rectangular_views(
        rng,
        sources=sources,
        frequencies=frequencies,
        source_count=source_count,
        frequency_count=frequency_count,
    )


@torch.no_grad()
def evaluate(
    encoder: WaveSetEncoder,
    decoder: FrozenMediumDecoder,
    bank: WaveSetBank,
    targets: torch.Tensor,
    indices: np.ndarray,
    views: tuple[tuple[int, int], ...],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    decoder.eval()
    predictions = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        value, metadata = bank.set_tensor(batch, views, device)
        predictions.append(decoder(encoder.spatial(value, metadata)).cpu())
    prediction = torch.cat(predictions)[:, 0].numpy()
    target = targets[indices, 0].numpy()
    numerator = np.linalg.norm((prediction - target).reshape(len(indices), -1), axis=1)
    denominator = np.linalg.norm(target.reshape(len(indices), -1), axis=1)
    return prediction, numerator / np.maximum(denominator, 1e-8)


def _budget_metrics(
    encoder: WaveSetEncoder,
    decoder: FrozenMediumDecoder,
    bank: WaveSetBank,
    targets: torch.Tensor,
    indices: np.ndarray,
    sources: tuple[int, ...],
    frequencies: tuple[int, ...],
    budgets: tuple[ViewBudget, ...],
    device: str,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    output = {}
    for budget in budgets:
        views = _fixed_views(sources, frequencies, budget)
        _, error = evaluate(
            encoder, decoder, bank, targets, indices, views, device, batch_size
        )
        output[f"s{budget.source_count}_f{budget.frequency_count}"] = {
            "views": len(views),
            "contrast_rel_l2_mean": float(error.mean()),
            "contrast_rel_l2_std": float(error.std()),
        }
    return output


def train(args: argparse.Namespace) -> dict:
    if args.reset_encoder and not args.finetune:
        raise ValueError("reset_encoder requires finetune")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    encoder_summary = checkpoint["summary"]
    representation = encoder_summary["representation"]
    data = np.load(args.dataset, allow_pickle=True)
    split = data["split"].astype(str)
    all_train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    if args.train_systems > len(all_train):
        raise ValueError("train_systems exceeds available systems")
    training = np.sort(rng.choice(all_train, args.train_systems, replace=False))
    bank = WaveSetBank(data, representation, scale=np.asarray(checkpoint["field_scale"]))
    width = int(encoder_summary["width"])
    dimension = int(encoder_summary["dimension"])
    aggregation = str(encoder_summary.get("aggregation", "moments"))
    slots = int(encoder_summary.get("slots", 4))
    metadata_conditioning = str(
        encoder_summary.get("metadata_conditioning", "deep_add")
    )
    basis_dimension = int(encoder_summary.get("basis_dimension", 4))
    encoder = WaveSetEncoder(
        bank.channels,
        width,
        dimension,
        aggregation,
        slots,
        metadata_conditioning,
        basis_dimension,
    ).to(device)
    if not args.reset_encoder:
        encoder.load_state_dict(checkpoint["encoder"])
    feature_width = int(encoder_summary.get("feature_width", width))
    decoder = FrozenMediumDecoder(feature_width).to(device)
    if not args.finetune:
        encoder.requires_grad_(False)
    parameters = list(decoder.parameters())
    if args.finetune:
        parameters.extend(encoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    targets = torch.from_numpy((1.0 - data["speed"][:, None]).astype(np.float32))
    selection_views = GROUP_LOW
    high_selection_views = GROUP_HIGH
    best_validation = float("inf")
    best_state = None
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        batch = rng.choice(
            training, args.batch_size, replace=len(training) < args.batch_size
        )
        if args.training_view_domain == "balanced":
            view_sets = (
                _sample_training_views(
                    rng, tuple(range(6)), tuple(range(4)), args.full_context_probability
                ),
                _sample_training_views(
                    rng,
                    tuple(range(6, 9)),
                    tuple(range(4, 8)),
                    args.full_context_probability,
                ),
            )
        else:
            use_high_group = args.training_view_domain == "mixed" and rng.random() < 0.5
            training_sources = tuple(range(6, 9)) if use_high_group else tuple(range(6))
            training_frequencies = tuple(range(4, 8)) if use_high_group else tuple(range(4))
            view_sets = (
                _sample_training_views(
                    rng,
                    training_sources,
                    training_frequencies,
                    args.full_context_probability,
                ),
            )
        target = targets[batch].to(device)
        encoder.train(args.finetune)
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        weight = 1.0 + 4.0 * (target > 1e-6)
        losses = []
        for views in view_sets:
            value, metadata = bank.set_tensor(batch, views, device)
            prediction = decoder(encoder.spatial(value, metadata))
            losses.append(((prediction - target).square() * weight).mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.evaluate_every == 0 or step == args.steps:
            _, error = evaluate(
                encoder,
                decoder,
                bank,
                targets,
                validation,
                selection_views,
                device,
                args.eval_batch_size,
            )
            validation_error = float(error.mean())
            if args.training_view_domain != "low":
                _, high_error = evaluate(
                    encoder,
                    decoder,
                    bank,
                    targets,
                    validation,
                    high_selection_views,
                    device,
                    args.eval_batch_size,
                )
                high_validation_error = float(high_error.mean())
                selection_error = 0.5 * (validation_error + high_validation_error)
            else:
                high_validation_error = None
                selection_error = validation_error
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "validation_rel_l2": validation_error,
                    "high_validation_rel_l2": high_validation_error,
                    "selection_rel_l2": selection_error,
                    "views": sum(len(views) for views in view_sets),
                }
            )
            print(
                f"{representation} labels {len(training):3d} step {step:5d} "
                f"loss {float(loss.detach()):.5f} val {validation_error:.4f}",
                flush=True,
            )
            if selection_error < best_validation:
                best_validation = selection_error
                best_state = {
                    "encoder": {
                        name: value.detach().cpu() for name, value in encoder.state_dict().items()
                    },
                    "decoder": {
                        name: value.detach().cpu() for name, value in decoder.state_dict().items()
                    },
                }
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    validation_prediction, _ = evaluate(
        encoder,
        decoder,
        bank,
        targets,
        validation,
        selection_views,
        device,
        args.eval_batch_size,
    )
    validation_target = targets[validation, 0].numpy()
    threshold_grid = np.linspace(0.02, 0.30, 29)
    threshold = max(
        threshold_grid,
        key=lambda current: float(
            _iou(validation_prediction, validation_target, current).mean()
        ),
    )
    prediction, errors = evaluate(
        encoder,
        decoder,
        bank,
        targets,
        test,
        selection_views,
        device,
        args.eval_batch_size,
    )
    target = targets[test, 0].numpy()
    iou = _iou(prediction, target, float(threshold))
    correlation = np.asarray(
        [
            np.corrcoef(prediction[index].ravel(), target[index].ravel())[0, 1]
            for index in range(len(test))
        ]
    )
    low_budgets = tuple(
        ViewBudget(source_count, frequency_count)
        for source_count in (1, 2, 4, 6)
        for frequency_count in (1, 2, 4)
    )
    high_budgets = tuple(
        ViewBudget(source_count, frequency_count)
        for source_count in (1, 2, 3)
        for frequency_count in (1, 2, 4)
    )
    summary = {
        "schema_version": 1,
        "task": (
            "finetuned_spatial_medium_probe"
            if args.finetune
            else "frozen_spatial_medium_probe"
        ),
        "representation": representation,
        "encoder_seed": int(encoder_summary["seed"]),
        "probe_seed": args.seed,
        "train_systems": len(training),
        "finetune": args.finetune,
        "reset_encoder": args.reset_encoder,
        "full_context_probability": args.full_context_probability,
        "training_view_domain": args.training_view_domain,
        "steps": args.steps,
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "aggregation": aggregation,
        "slots": slots,
        "metadata_conditioning": metadata_conditioning,
        "basis_dimension": basis_dimension,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "selected_threshold": float(threshold),
        "primary_low_s6_f4": {
            "contrast_rel_l2_mean": float(errors.mean()),
            "contrast_rel_l2_std": float(errors.std()),
            "inclusion_iou_mean": float(iou.mean()),
            "pixel_correlation_mean": float(np.nanmean(correlation)),
        },
        "low_group_budgets": _budget_metrics(
            encoder,
            decoder,
            bank,
            targets,
            test,
            tuple(range(6)),
            tuple(range(4)),
            low_budgets,
            device,
            args.eval_batch_size,
        ),
        "disjoint_high_group_budgets": _budget_metrics(
            encoder,
            decoder,
            bank,
            targets,
            test,
            tuple(range(6, 9)),
            tuple(range(4, 8)),
            high_budgets,
            device,
            args.eval_batch_size,
        ),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f"{representation}_enc{encoder_summary['seed']}_labels{len(training)}_seed{args.seed}"
    if args.finetune:
        tag += "_finetune"
    checkpoint_path = args.output / f"set_probe_{tag}.pt"
    torch.save(
        {
            **best_state,
            "summary": summary,
            "args": vars(args),
            "field_scale": bank.scale,
        },
        checkpoint_path,
    )
    summary["checkpoint"] = str(checkpoint_path)
    (args.output / f"set_probe_{tag}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    rows = [
        {
            "system": int(system),
            "contrast_rel_l2": float(errors[position]),
            "inclusion_iou": float(iou[position]),
            "pixel_correlation": float(correlation[position]),
        }
        for position, system in enumerate(test)
    ]
    with (args.output / f"set_probe_{tag}_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-systems", type=int, default=112)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--reset-encoder", action="store_true")
    parser.add_argument("--full-context-probability", type=float, default=0.5)
    parser.add_argument(
        "--training-view-domain", choices=("low", "mixed", "balanced"), default="low"
    )
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--evaluate-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260922)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
