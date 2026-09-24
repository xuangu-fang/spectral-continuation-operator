"""Train and evaluate compact in-context wave response query models."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from .response_query import (
    MaskedResponseTransformer,
    ResponseNormalizer,
    ResponseQueryCNP,
    SpatialResponseQueryOperator,
    complex_relative_l2,
    frozen_episodes,
    load_response_archive,
    normalized_query_coordinates,
    sample_episode,
)


def _encoded_table(archive, normalizer: ResponseNormalizer) -> np.ndarray:
    systems, sources, frequencies, receivers = archive.responses.shape
    source_index = np.broadcast_to(
        np.arange(sources)[None, :, None], (systems, sources, frequencies)
    )
    frequency_index = np.broadcast_to(
        np.arange(frequencies)[None, None, :], (systems, sources, frequencies)
    )
    encoded = normalizer.encode(archive.responses, source_index, frequency_index)
    side = round(receivers**0.5)
    return encoded.reshape(systems, sources, frequencies, 2, side, side)


def reciprocity_baseline_encoded(archive, normalizer, episode, budget, calibrated=False):
    from .audit_response_query import _reciprocity_prediction

    physical = _reciprocity_prediction(
        archive, normalizer, episode, budget, calibrated=calibrated
    )
    encoded = normalizer.encode(
        physical, np.asarray(episode.target_source), np.asarray(episode.target_frequency)
    )
    side = round(encoded.shape[-1] ** 0.5)
    return encoded.reshape(2, side, side)


class TrainingBatcher:
    def __init__(self, archive, normalizer, encoded: np.ndarray, seed: int):
        self.archive = archive
        self.normalizer = normalizer
        self.encoded = encoded
        self.coordinates = normalized_query_coordinates(archive)
        self.systems = np.flatnonzero(archive.split == "train")
        self.generator = np.random.default_rng(seed)
        self.source_count, self.frequency_count = archive.query_shape

    def sample(
        self,
        batch_size: int,
        budgets: tuple[int, ...],
        regimes: tuple[str, ...],
        reciprocity_residual: bool = False,
        calibrated_reciprocity: bool = False,
    ):
        budget = int(self.generator.choice(budgets))
        context_response = []
        context_query = []
        target_response = []
        target_query = []
        target_indices = []
        baselines = []
        for _ in range(batch_size):
            system = int(self.generator.choice(self.systems))
            regime = str(self.generator.choice(regimes))
            episode = sample_episode(
                self.generator,
                system=system,
                source_count=self.source_count,
                frequency_count=self.frequency_count,
                budget=budget,
                regime=regime,
            )
            context_response.append(
                np.stack(
                    [self.encoded[system, source, frequency] for source, frequency in episode.context]
                )
            )
            context_query.append(
                np.stack([self.coordinates[source, frequency] for source, frequency in episode.context])
            )
            target_response.append(
                self.encoded[system, episode.target_source, episode.target_frequency]
            )
            target_query.append(self.coordinates[episode.target_source, episode.target_frequency])
            target_indices.append((episode.target_source, episode.target_frequency, system))
            baselines.append(
                reciprocity_baseline_encoded(
                    self.archive,
                    self.normalizer,
                    episode,
                    budget,
                    calibrated_reciprocity,
                )
                if reciprocity_residual
                else np.zeros_like(target_response[-1])
            )
        return (
            np.stack(context_response),
            np.stack(context_query),
            np.stack(target_response),
            np.stack(target_query),
            np.asarray(target_indices, dtype=np.int64),
            np.stack(baselines),
        )


@torch.no_grad()
def evaluate(
    model,
    archive,
    normalizer,
    encoded,
    episodes,
    budgets,
    device,
    batch_size=128,
    reciprocity_residual=False,
    calibrated_reciprocity=False,
):
    model.eval()
    coordinates = normalized_query_coordinates(archive)
    rows = []
    for budget in budgets:
        for start in range(0, len(episodes), batch_size):
            batch = episodes[start : start + batch_size]
            context_response = np.stack(
                [
                    np.stack(
                        [encoded[e.system, source, frequency] for source, frequency in e.context[:budget]]
                    )
                    for e in batch
                ]
            )
            context_query = np.stack(
                [
                    np.stack([coordinates[source, frequency] for source, frequency in e.context[:budget]])
                    for e in batch
                ]
            )
            target_query = np.stack(
                [coordinates[e.target_source, e.target_frequency] for e in batch]
            )
            prediction = model(
                torch.from_numpy(context_response).to(device),
                torch.from_numpy(context_query).to(device),
                torch.ones(context_response.shape[:2], dtype=torch.bool, device=device),
                torch.from_numpy(target_query).to(device),
            ).cpu().numpy()
            if reciprocity_residual:
                prediction += np.stack(
                    [
                        reciprocity_baseline_encoded(
                            archive, normalizer, episode, budget, calibrated_reciprocity
                        )
                        for episode in batch
                    ]
                )
            prediction = prediction.reshape(len(batch), 2, -1)
            for index, episode in enumerate(batch):
                decoded = normalizer.decode(
                    prediction[index],
                    np.asarray(episode.target_source),
                    np.asarray(episode.target_frequency),
                )
                truth = archive.responses[
                    episode.system, episode.target_source, episode.target_frequency
                ]
                rows.append(
                    {
                        "system": episode.system,
                        "regime": episode.regime,
                        "budget": budget,
                        "relative_l2": float(complex_relative_l2(decoded, truth)),
                    }
                )
    summary = {}
    for regime in sorted({row["regime"] for row in rows}):
        summary[regime] = {}
        for budget in budgets:
            selected = [
                row["relative_l2"]
                for row in rows
                if row["regime"] == regime and row["budget"] == budget
            ]
            summary[regime][str(budget)] = {
                "mean": float(np.mean(selected)),
                "median": float(np.median(selected)),
            }
    return summary, rows


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    archive = load_response_archive(args.dataset, args.receivers, args.receiver_mode)
    normalizer = ResponseNormalizer(archive)
    encoded = _encoded_table(archive, normalizer)
    batcher = TrainingBatcher(archive, normalizer, encoded, args.seed)
    budgets = tuple(int(value) for value in args.budgets.split(","))
    regimes = tuple(args.regimes.split(","))
    validation = frozen_episodes(
        archive,
        split="validation",
        regimes=regimes,
        episodes_per_system=args.validation_episodes,
        seed=20260924,
    )
    if args.model == "spatial":
        model = SpatialResponseQueryOperator(args.spatial_width).to(device)
    elif args.model == "transformer":
        model = MaskedResponseTransformer(
            args.receivers, args.hidden, args.transformer_layers, args.transformer_heads
        ).to(device)
    else:
        model = ResponseQueryCNP(
            receiver_count=args.receivers,
            hidden=args.hidden,
            attention=args.attention,
        ).to(device)
    if args.initial_checkpoint is not None:
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_error = float("inf")
    start_time = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        arrays = batcher.sample(
            args.batch_size,
            budgets,
            regimes,
            args.reciprocity_residual,
            args.calibrated_reciprocity,
        )
        context_response, context_query, target_response, target_query, target_indices, baseline = arrays
        context_response_t = torch.from_numpy(context_response).to(device)
        context_query_t = torch.from_numpy(context_query).to(device)
        target_response_t = torch.from_numpy(target_response).to(device)
        target_query_t = torch.from_numpy(target_query).to(device)
        baseline_t = torch.from_numpy(baseline).to(device)
        mask = torch.ones(context_response.shape[:2], dtype=torch.bool, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(context_response_t, context_query_t, mask, target_query_t)
            normalized_prediction = prediction + baseline_t
            loss = F.smooth_l1_loss(normalized_prediction, target_response_t)
            if args.relative_loss_weight > 0.0:
                sources = target_indices[:, 0]
                frequencies = target_indices[:, 1]
                systems = target_indices[:, 2]
                mean = normalizer.mean[sources, frequencies]
                scale = normalizer.scale[sources, frequencies]
                target_physical = archive.responses[systems, sources, frequencies]
                mean_t = torch.from_numpy(np.stack((mean.real, mean.imag), axis=1)).to(device)
                scale_t = torch.from_numpy(scale[:, None]).to(device)
                target_physical_t = torch.from_numpy(
                    np.stack((target_physical.real, target_physical.imag), axis=1)
                ).to(device)
                physical_prediction = mean_t + scale_t * normalized_prediction.flatten(2)
                numerator = (physical_prediction - target_physical_t).square().sum((1, 2)).sqrt()
                denominator = target_physical_t.square().sum((1, 2)).sqrt().clamp_min(1e-12)
                loss = loss + args.relative_loss_weight * (numerator / denominator).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step % args.evaluate_every == 0 or step == args.steps:
            summary, _ = evaluate(
                model,
                archive,
                normalizer,
                encoded,
                validation,
                budgets,
                device,
                reciprocity_residual=args.reciprocity_residual,
                calibrated_reciprocity=args.calibrated_reciprocity,
            )
            selection_budget = "8" if 8 in budgets else str(max(budgets))
            q3_error = summary.get("q3", summary[regimes[-1]])[selection_budget]["mean"]
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "q3_k8": q3_error,
                "elapsed_seconds": time.time() - start_time,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if q3_error < best_error:
                best_error = q3_error
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "normalizer_mean": normalizer.mean,
                        "normalizer_scale": normalizer.scale,
                        "validation": summary,
                    },
                    output / "best.pt",
                )
            model.train()

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    summary, rows = evaluate(
        model,
        archive,
        normalizer,
        encoded,
        validation,
        budgets,
        device,
        reciprocity_residual=args.reciprocity_residual,
        calibrated_reciprocity=args.calibrated_reciprocity,
    )
    result = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_q3_k8": best_error,
        "validation": summary,
        "history": history,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    with (output / "validation_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.plot([item["step"] for item in history], [item["q3_k8"] for item in history], marker="o")
    axis.axhline(0.939, color="black", linestyle="--", linewidth=1, label="RQ-0 baseline")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Q3 K=8 relative L2")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "validation_curve.png", dpi=180)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receivers", type=int, default=64)
    parser.add_argument("--receiver-mode", choices=("cartesian", "source_array"), default="cartesian")
    parser.add_argument("--reciprocity-residual", action="store_true")
    parser.add_argument("--calibrated-reciprocity", action="store_true")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--relative-loss-weight", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--model", choices=("cnp", "spatial", "transformer"), default="cnp")
    parser.add_argument("--spatial-width", type=int, default=64)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--evaluate-every", type=int, default=1000)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--budgets", default="1,2,4,8,16")
    parser.add_argument("--regimes", default="q1,q2,q3")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(args)
    print(json.dumps({"best_q3_k8": result["best_q3_k8"], "parameters": result["parameters"]}))


if __name__ == "__main__":
    main()
