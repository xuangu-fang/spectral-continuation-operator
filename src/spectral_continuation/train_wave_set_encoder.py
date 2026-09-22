"""Self-supervised pretraining of a permutation-invariant wave-system encoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .train_wave_operator_contrastive import anti_collapse_loss, retrieval_metrics
from .wave_set_encoder import (
    GROUP_HIGH,
    GROUP_LOW,
    QueryViewDecoder,
    WaveSetBank,
    WaveSetEncoder,
    apply_independent_phase_rotation,
    sample_rectangular_views,
)


def normalized_view_reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Give each response-token channel comparable influence without dataset labels."""

    channel_energy = target.square().mean(dim=(0, 2, 3)).detach().clamp_min(0.05)
    channel_error = (prediction - target).square().mean(dim=(0, 2, 3))
    return (channel_error / channel_energy).mean()


def dense_system_contrastive_loss(
    first: torch.Tensor, second: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Match the same physical location across views while contrasting systems."""

    first = F.normalize(first.flatten(2), dim=1)
    second = F.normalize(second.flatten(2), dim=1)
    # [position, anchor system, candidate system]
    logits = torch.einsum("bcp,dcp->pbd", first, second) / temperature
    labels = torch.arange(first.shape[0], device=first.device).expand(logits.shape[0], -1)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(1, 2), labels)
    )


@torch.no_grad()
def embed(
    model: WaveSetEncoder,
    bank: WaveSetBank,
    systems: np.ndarray,
    views: tuple[tuple[int, int], ...],
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(systems), batch_size):
        batch = systems[start : start + batch_size]
        value, metadata = bank.set_tensor(batch, views, device)
        output.append(model(value, metadata).cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def evaluate_view_prediction(
    model: WaveSetEncoder,
    decoder: QueryViewDecoder,
    bank: WaveSetBank,
    systems: np.ndarray,
    device: str,
    batch_size: int,
) -> float:
    """Evaluate symmetric prediction across disjoint source-frequency groups."""

    model.eval()
    decoder.eval()
    losses = []
    low_queries = ((0, 0), (2, 2), (5, 3))
    high_queries = ((6, 4), (7, 6), (8, 7))
    for start in range(0, len(systems), batch_size):
        batch = systems[start : start + batch_size]
        low, low_metadata = bank.set_tensor(batch, GROUP_LOW, device)
        high, high_metadata = bank.set_tensor(batch, GROUP_HIGH, device)
        low_features = model.spatial(low, low_metadata)
        high_features = model.spatial(high, high_metadata)
        for low_query, high_query in zip(low_queries, high_queries, strict=True):
            low_target, low_target_metadata = bank.set_tensor(batch, (low_query,), device)
            high_target, high_target_metadata = bank.set_tensor(batch, (high_query,), device)
            losses.append(
                normalized_view_reconstruction_loss(
                    decoder(low_features, high_target_metadata[:, 0]), high_target[:, 0]
                ).item()
            )
            losses.append(
                normalized_view_reconstruction_loss(
                    decoder(high_features, low_target_metadata[:, 0]), low_target[:, 0]
                ).item()
            )
    return float(np.mean(losses))


def _sample_group(
    rng: np.random.Generator,
    *,
    sources: tuple[int, ...],
    frequencies: tuple[int, ...],
    maximum_views: int,
) -> tuple[tuple[int, int], ...]:
    source_count = int(rng.integers(1, len(sources) + 1))
    maximum_frequencies = min(len(frequencies), max(maximum_views // source_count, 1))
    frequency_count = int(rng.integers(1, maximum_frequencies + 1))
    return sample_rectangular_views(
        rng,
        sources=sources,
        frequencies=frequencies,
        source_count=source_count,
        frequency_count=frequency_count,
    )


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.dataset, allow_pickle=True)
    split = data["split"].astype(str)
    train_systems = np.flatnonzero(split == "train")
    validation_systems = np.flatnonzero(split == "validation")
    test_systems = np.flatnonzero(split == "test")
    bank = WaveSetBank(data, args.representation)
    model = WaveSetEncoder(
        bank.channels,
        args.width,
        args.dimension,
        args.aggregation,
        args.slots,
        args.metadata_conditioning,
        args.basis_dimension,
    ).to(device)
    view_decoder = (
        QueryViewDecoder(bank.channels, model.feature_width).to(device)
        if args.view_prediction_weight > 0.0
        else None
    )
    parameters = list(model.parameters())
    if view_decoder is not None:
        parameters.extend(view_decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    best_key = (-float("inf"), -float("inf"))
    best_state = None
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        systems = rng.choice(train_systems, args.batch_size, replace=False)
        views_low = _sample_group(
            rng,
            sources=tuple(range(6)),
            frequencies=tuple(range(4)),
            maximum_views=args.maximum_views,
        )
        views_high = _sample_group(
            rng,
            sources=tuple(range(6, 9)),
            frequencies=tuple(range(4, 8)),
            maximum_views=args.maximum_views,
        )
        low, low_metadata = bank.set_tensor(systems, views_low, device)
        high, high_metadata = bank.set_tensor(systems, views_high, device)
        if args.phase_augmentation:
            if args.representation != "raw_normalized":
                raise ValueError("phase augmentation is only defined for raw_normalized")
            low = apply_independent_phase_rotation(low)
            high = apply_independent_phase_rotation(high)
        model.train()
        if view_decoder is not None:
            view_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        first_spatial = model.spatial(low, low_metadata)
        second_spatial = model.spatial(high, high_metadata)
        first = model.global_embedding(first_spatial)
        second = model.global_embedding(second_spatial)
        logits = first @ second.T / args.temperature
        labels = torch.arange(len(systems), device=device)
        contrastive = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
        regularizer = anti_collapse_loss(first, second)
        spatial_contrastive = dense_system_contrastive_loss(
            first_spatial[-1], second_spatial[-1], args.spatial_temperature
        )
        if args.view_prediction_weight > 0.0:
            if view_decoder is None:
                raise RuntimeError("view decoder is required for view prediction")
            target_low_view = ((int(rng.integers(0, 6)), int(rng.integers(0, 4))),)
            target_high_view = ((int(rng.integers(6, 9)), int(rng.integers(4, 8))),)
            target_low, target_low_metadata = bank.set_tensor(
                systems, target_low_view, device
            )
            target_high, target_high_metadata = bank.set_tensor(
                systems, target_high_view, device
            )
            prediction_high = view_decoder(first_spatial, target_high_metadata[:, 0])
            prediction_low = view_decoder(second_spatial, target_low_metadata[:, 0])
            view_prediction = 0.5 * (
                normalized_view_reconstruction_loss(prediction_high, target_high[:, 0])
                + normalized_view_reconstruction_loss(prediction_low, target_low[:, 0])
            )
        else:
            view_prediction = torch.zeros((), device=device)
        loss = (
            contrastive
            + regularizer
            + args.spatial_weight * spatial_contrastive
            + args.view_prediction_weight * view_prediction
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.evaluate_every == 0 or step == args.steps:
            validation_low = embed(
                model, bank, validation_systems, GROUP_LOW, device, args.eval_batch_size
            )
            validation_high = embed(
                model, bank, validation_systems, GROUP_HIGH, device, args.eval_batch_size
            )
            metrics = retrieval_metrics(validation_low, validation_high)
            validation_view_prediction = (
                evaluate_view_prediction(
                    model,
                    view_decoder,
                    bank,
                    validation_systems,
                    device,
                    args.eval_batch_size,
                )
                if args.view_prediction_weight > 0.0
                else None
            )
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "global_contrastive": float(contrastive.detach()),
                    "spatial_contrastive": float(spatial_contrastive.detach()),
                    "view_prediction": float(view_prediction.detach()),
                    "validation_view_prediction": validation_view_prediction,
                    "low_views": len(views_low),
                    "high_views": len(views_high),
                    **metrics,
                }
            )
            print(
                f"{args.representation} step {step:5d} loss {float(loss.detach()):.4f} "
                f"top1 {metrics['top1']:.3f} margin {metrics['cosine_margin']:.3f}",
                flush=True,
            )
            key = (
                (-validation_view_prediction, metrics["top1"])
                if validation_view_prediction is not None
                else (metrics["top1"], metrics["cosine_margin"])
            )
            if key > best_key:
                best_key = key
                best_state = {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                }
                best_view_decoder_state = {
                    name: value.detach().cpu()
                    for name, value in (
                        view_decoder.state_dict().items() if view_decoder is not None else ()
                    )
                }
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    test_low = embed(model, bank, test_systems, GROUP_LOW, device, args.eval_batch_size)
    test_high = embed(model, bank, test_systems, GROUP_HIGH, device, args.eval_batch_size)
    metrics = retrieval_metrics(test_low, test_high)
    summary = {
        "schema_version": 1,
        "task": "self_supervised_wave_system_set_encoder",
        "representation": args.representation,
        "phase_augmentation": args.phase_augmentation,
        "seed": args.seed,
        "steps": args.steps,
        "width": args.width,
        "feature_width": model.feature_width,
        "dimension": args.dimension,
        "aggregation": args.aggregation,
        "slots": args.slots,
        "metadata_conditioning": args.metadata_conditioning,
        "basis_dimension": args.basis_dimension,
        "spatial_weight": args.spatial_weight,
        "view_prediction_weight": args.view_prediction_weight,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pretraining_head_parameters": (
            sum(parameter.numel() for parameter in view_decoder.parameters())
            if view_decoder is not None
            else 0
        ),
        "test_systems": len(test_systems),
        "chance_top1": 1.0 / len(test_systems),
        **{f"test_{key}": value for key, value in metrics.items()},
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / f"set_encoder_{args.representation}_seed{args.seed}.pt"
    torch.save(
        {
            "encoder": best_state,
            "view_decoder": best_view_decoder_state,
            "summary": summary,
            "args": vars(args),
            "field_scale": bank.scale,
        },
        checkpoint_path,
    )
    summary["checkpoint"] = str(checkpoint_path)
    summary_path = args.output / f"set_encoder_{args.representation}_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=(
            "raw",
            "raw_normalized",
            "amplitude",
            "amplitude_soft_phase_gradient",
            "amplitude_multiscale_phase_gradient",
        ),
        required=True,
    )
    parser.add_argument("--phase-augmentation", action="store_true")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--evaluate-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--maximum-views", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument(
        "--aggregation",
        choices=("moments", "slots", "slot_stack", "coordinate_basis"),
        default="moments",
    )
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--basis-dimension", type=int, default=4)
    parser.add_argument(
        "--metadata-conditioning",
        choices=("deep_add", "multiscale_film"),
        default="deep_add",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--spatial-temperature", type=float, default=0.1)
    parser.add_argument("--spatial-weight", type=float, default=1.0)
    parser.add_argument("--view-prediction-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260922)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
