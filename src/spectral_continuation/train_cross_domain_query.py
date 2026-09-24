"""Train a receiver-preserving residual operator for held-out source queries."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cross_domain_query_audit import _fit_subspaces
from .cross_domain_response_data import load_cross_domain_archive
from .response_query import fourier_query_features


class QueryNormalizer:
    def __init__(self, response: np.ndarray, train: np.ndarray, floor_fraction: float = 0.02):
        values = response[train]
        self.mean = values.mean(axis=0).astype(np.complex64)
        scale = np.sqrt(np.mean(np.abs(values - self.mean[None]) ** 2, axis=0))
        floor = floor_fraction * float(np.median(scale))
        self.scale = np.maximum(scale, floor).astype(np.float32)

    def encode(self, value: np.ndarray, source: np.ndarray, frequency: np.ndarray) -> np.ndarray:
        normalized = (value - self.mean[source, frequency]) / self.scale[source, frequency]
        return np.stack((normalized.real, normalized.imag), axis=1).astype(np.float32)


class TorchSubspacePredictor:
    def __init__(self, subspaces, receiver_count: int, device: torch.device):
        self.mean = torch.stack(
            [torch.from_numpy(item.mean) for item in subspaces]
        ).to(device)
        self.basis = torch.stack(
            [torch.from_numpy(item.basis) for item in subspaces]
        ).to(device)
        self.receiver_count = receiver_count

    @torch.no_grad()
    def predict(
        self,
        response: torch.Tensor,
        context: torch.Tensor,
        target: torch.Tensor,
        frequency: torch.Tensor,
        rank: int,
        ridge: float,
    ) -> torch.Tensor:
        batch, budget, receivers = response.shape
        basis = self.basis[frequency, :, :rank]
        mean = self.mean[frequency]
        receiver = torch.arange(receivers, device=response.device)
        observed = (context[:, :, None] * receivers + receiver[None, None]).reshape(batch, -1)
        observed_basis = torch.gather(
            basis, 1, observed[:, :, None].expand(-1, -1, rank)
        )
        observed_mean = torch.gather(mean, 1, observed)
        centered = response.reshape(batch, budget * receivers) - observed_mean
        adjoint = observed_basis.conj().transpose(1, 2)
        gram = adjoint @ observed_basis
        identity = torch.eye(rank, device=response.device, dtype=gram.dtype)[None]
        coefficient = torch.linalg.solve(
            gram + ridge * identity, adjoint @ centered[:, :, None]
        )
        target_index = target[:, None] * receivers + receiver[None]
        target_basis = torch.gather(
            basis, 1, target_index[:, :, None].expand(-1, -1, rank)
        )
        target_mean = torch.gather(mean, 1, target_index)
        return target_mean + (target_basis @ coefficient)[:, :, 0]


class ReceiverQueryOperator(nn.Module):
    """Keep receiver-local phase relations while aggregating a probe set."""

    def __init__(self, receiver_coordinates: np.ndarray, width: int, periodic: bool):
        super().__init__()
        padding_mode = "circular" if periodic else "zeros"
        self.width = width
        self.response_encoder = nn.Sequential(
            nn.Conv1d(2, width, 5, padding=2, padding_mode=padding_mode),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=2, padding_mode=padding_mode),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.baseline_encoder = nn.Sequential(
            nn.Conv1d(2, width, 5, padding=2, padding_mode=padding_mode),
            nn.SiLU(),
        )
        self.context_metadata = nn.Sequential(
            nn.Linear(21, width), nn.SiLU(), nn.Linear(width, 2 * width)
        )
        self.target_metadata = nn.Sequential(
            nn.Linear(21, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.receiver_metadata = nn.Sequential(
            nn.Linear(14, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.key = nn.Linear(width, width)
        self.decoder = nn.Sequential(
            nn.Conv1d(5 * width, 2 * width, 5, padding=2, padding_mode=padding_mode),
            nn.GroupNorm(8, 2 * width),
            nn.SiLU(),
            nn.Conv1d(2 * width, width, 5, padding=2, padding_mode=padding_mode),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv1d(width, 2, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        coordinate = np.asarray(receiver_coordinates, dtype=np.float32)
        low = coordinate.min(axis=0)
        span = np.maximum(coordinate.max(axis=0) - low, 1e-6)
        coordinate = 2.0 * (coordinate - low) / span - 1.0
        self.register_buffer("receiver_coordinates", torch.from_numpy(coordinate))

    def forward(
        self,
        context_response: torch.Tensor,
        context_query: torch.Tensor,
        target_query: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, _, receivers = context_response.shape
        feature = self.response_encoder(context_response.reshape(batch * count, 2, receivers))
        feature = feature.reshape(batch, count, self.width, receivers)
        scale, shift = self.context_metadata(
            fourier_query_features(context_query).reshape(batch * count, -1)
        ).chunk(2, dim=-1)
        feature = feature * (1.0 + 0.1 * torch.tanh(scale.reshape(batch, count, -1, 1)))
        feature = feature + shift.reshape(batch, count, -1, 1)

        mean = feature.mean(dim=1)
        standard_deviation = torch.sqrt(
            (feature.square().mean(dim=1) - mean.square()).clamp_min(0.0) + 1e-5
        )
        target = self.target_metadata(fourier_query_features(target_query))
        pooled = feature.mean(dim=-1)
        score = torch.einsum("bkh,bh->bk", self.key(pooled), target) / self.width**0.5
        attention = torch.softmax(score, dim=1)
        attended = torch.einsum("bk,bkhr->bhr", attention, feature)
        baseline_feature = self.baseline_encoder(baseline)
        receiver_feature = self.receiver_metadata(
            fourier_query_features(self.receiver_coordinates)
        ).T[None].expand(batch, -1, -1)
        conditioned = receiver_feature + target[:, :, None]
        residual = self.decoder(
            torch.cat(
                (mean, standard_deviation, attended, baseline_feature, conditioned), dim=1
            )
        )
        return baseline + residual


class EpisodeBatcher:
    def __init__(self, archive, normalizer: QueryNormalizer, seed: int):
        self.archive = archive
        self.normalizer = normalizer
        self.generator = np.random.default_rng(seed)
        self.systems = np.flatnonzero(archive.split == "train")
        source = 2.0 * archive.sources - 1.0
        log_frequency = np.log(archive.frequencies)
        log_frequency = 2.0 * (log_frequency - log_frequency.min()) / (
            log_frequency.max() - log_frequency.min()
        ) - 1.0
        self.query = np.concatenate(
            (
                np.repeat(source[:, None], len(log_frequency), axis=1),
                np.repeat(log_frequency[None, :, None], len(source), axis=0),
            ),
            axis=-1,
        ).astype(np.float32)

    def sample(self, batch_size: int, budget: int):
        source_count = self.archive.responses.shape[1]
        frequency_count = self.archive.responses.shape[2]
        system = self.generator.choice(self.systems, size=batch_size)
        target = self.generator.integers(source_count, size=batch_size)
        frequency = self.generator.integers(frequency_count, size=batch_size)
        context = np.empty((batch_size, budget), dtype=np.int64)
        for index, target_source in enumerate(target):
            candidates = np.delete(np.arange(source_count), target_source)
            context[index] = self.generator.choice(candidates, size=budget, replace=False)
        raw_context = self.archive.responses[
            system[:, None], context, frequency[:, None]
        ]
        context_encoded = np.stack(
            [
                self.normalizer.encode(raw_context[index], context[index], frequency[index])
                for index in range(batch_size)
            ]
        )
        target_raw = self.archive.responses[system, target, frequency]
        target_encoded = self.normalizer.encode(target_raw, target, frequency)
        context_query = self.query[context, frequency[:, None]]
        target_query = self.query[target, frequency]
        return {
            "system": system,
            "target": target,
            "frequency": frequency,
            "context": context,
            "context_raw": raw_context,
            "context_encoded": context_encoded,
            "target_raw": target_raw,
            "target_encoded": target_encoded,
            "context_query": context_query,
            "target_query": target_query,
        }


def _parse_rank_map(text: str) -> dict[int, int]:
    return {int(pair.split(":")[0]): int(pair.split(":")[1]) for pair in text.split(",")}


def _complex_to_channels(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((value.real, value.imag), dim=1)


def _encode_baseline(
    baseline: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    frequency: torch.Tensor,
) -> torch.Tensor:
    normalized = (baseline - mean[target, frequency]) / scale[target, frequency]
    return _complex_to_channels(normalized)


def _decode(
    encoded: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    frequency: torch.Tensor,
) -> torch.Tensor:
    normalized = torch.complex(encoded[:, 0], encoded[:, 1])
    return mean[target, frequency] + scale[target, frequency] * normalized


@torch.no_grad()
def evaluate(
    model,
    archive,
    normalizer,
    predictor,
    rank_by_budget,
    split,
    budgets,
    repeats,
    seed,
    device,
    max_systems=None,
):
    model.eval()
    systems = np.flatnonzero(archive.split == split)
    if max_systems is not None:
        systems = systems[:max_systems]
    generator = np.random.default_rng(seed)
    mean = torch.from_numpy(normalizer.mean).to(device)
    scale = torch.from_numpy(normalizer.scale).to(device)
    source = 2.0 * archive.sources - 1.0
    log_frequency = np.log(archive.frequencies)
    log_frequency = 2.0 * (log_frequency - log_frequency.min()) / (
        log_frequency.max() - log_frequency.min()
    ) - 1.0
    query = np.concatenate(
        (
            np.repeat(source[:, None], len(log_frequency), axis=1),
            np.repeat(log_frequency[None, :, None], len(source), axis=0),
        ),
        axis=-1,
    ).astype(np.float32)
    frozen_orders = {
        target: [
            generator.permutation(np.delete(np.arange(len(source)), target))
            for _ in range(repeats)
        ]
        for target in range(len(source))
    }
    rows = []
    for budget in budgets:
        model_total = np.zeros(len(systems), dtype=np.float64)
        baseline_total = np.zeros(len(systems), dtype=np.float64)
        count = 0
        for target in range(len(source)):
            for order in frozen_orders[target]:
                context = order[:budget]
                for frequency in range(len(log_frequency)):
                    for start in range(0, len(systems), 128):
                        selected = systems[start : start + 128]
                        batch = len(selected)
                        context_batch = np.broadcast_to(context, (batch, budget)).copy()
                        target_batch = np.full(batch, target, dtype=np.int64)
                        frequency_batch = np.full(batch, frequency, dtype=np.int64)
                        raw_context = archive.responses[
                            selected[:, None], context_batch, frequency_batch[:, None]
                        ]
                        encoded_context = np.stack(
                            [
                                normalizer.encode(
                                    raw_context[index], context_batch[index], frequency
                                )
                                for index in range(batch)
                            ]
                        )
                        context_t = torch.from_numpy(context_batch).to(device)
                        target_t = torch.from_numpy(target_batch).to(device)
                        frequency_t = torch.from_numpy(frequency_batch).to(device)
                        raw_context_t = torch.from_numpy(raw_context).to(device)
                        baseline = predictor.predict(
                            raw_context_t,
                            context_t,
                            target_t,
                            frequency_t,
                            rank_by_budget[budget],
                            0.01,
                        )
                        baseline_encoded = _encode_baseline(
                            baseline, mean, scale, target_t, frequency_t
                        )
                        prediction_encoded = model(
                            torch.from_numpy(encoded_context).to(device),
                            torch.from_numpy(
                                query[context_batch, frequency_batch[:, None]]
                            ).to(device),
                            torch.from_numpy(query[target_batch, frequency_batch]).to(device),
                            baseline_encoded,
                        )
                        prediction = _decode(
                            prediction_encoded, mean, scale, target_t, frequency_t
                        )
                        truth = torch.from_numpy(
                            archive.responses[selected, target, frequency]
                        ).to(device)
                        denominator = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)
                        model_error = (
                            torch.linalg.vector_norm(prediction - truth, dim=-1) / denominator
                        ).cpu().numpy()
                        baseline_error = (
                            torch.linalg.vector_norm(baseline - truth, dim=-1) / denominator
                        ).cpu().numpy()
                        model_total[start : start + batch] += model_error
                        baseline_total[start : start + batch] += baseline_error
                    count += 1
        model_total /= count
        baseline_total /= count
        for system_id, model_error, baseline_error in zip(
            archive.system_ids[systems], model_total, baseline_total, strict=True
        ):
            rows.append(
                {
                    "system_id": str(system_id),
                    "split": split,
                    "budget": budget,
                    "model_rel_l2": float(model_error),
                    "subspace_rel_l2": float(baseline_error),
                }
            )
    return rows


def _summary(rows):
    result = {}
    for budget in sorted({row["budget"] for row in rows}):
        chosen = [row for row in rows if row["budget"] == budget]
        model = np.asarray([row["model_rel_l2"] for row in chosen])
        baseline = np.asarray([row["subspace_rel_l2"] for row in chosen])
        result[str(budget)] = {
            "model_mean": float(model.mean()),
            "subspace_mean": float(baseline.mean()),
            "model_vs_subspace_relative_improvement": float(
                np.mean((baseline - model) / np.maximum(baseline, 1e-12))
            ),
            "positive_system_fraction": float(np.mean(model < baseline)),
        }
    return result


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    archive = load_cross_domain_archive(args.dataset)
    train_systems = np.flatnonzero(archive.split == "train")
    normalizer = QueryNormalizer(archive.responses, train_systems)
    subspaces = _fit_subspaces(archive, max_rank=32)
    predictor = TorchSubspacePredictor(subspaces, archive.responses.shape[-1], device)
    rank_by_budget = _parse_rank_map(args.rank_by_budget)
    budgets = tuple(sorted(rank_by_budget))
    model = ReceiverQueryOperator(
        archive.receivers, args.width, periodic=args.receiver_topology == "periodic"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.steps, eta_min=args.learning_rate * 0.05
    )
    batcher = EpisodeBatcher(archive, normalizer, args.seed)
    mean = torch.from_numpy(normalizer.mean).to(device)
    scale = torch.from_numpy(normalizer.scale).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    initial_validation = evaluate(
        model,
        archive,
        normalizer,
        predictor,
        rank_by_budget,
        "validation",
        (args.selection_budget,),
        1,
        20260924,
        device,
        max_systems=args.validation_systems,
    )
    best_error = _summary(initial_validation)[str(args.selection_budget)]["model_mean"]
    history = [
        {
            "step": 0,
            "train_loss": None,
            "train_rel_l2": None,
            "validation_rel_l2": best_error,
        }
    ]
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "rank_by_budget": rank_by_budget,
            "step": 0,
            "validation_rel_l2": best_error,
        },
        args.output / "best.pt",
    )
    start_time = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        budget = int(np.random.choice(budgets))
        batch = batcher.sample(args.batch_size, budget)
        context = torch.from_numpy(batch["context"]).to(device)
        target = torch.from_numpy(batch["target"]).to(device)
        frequency = torch.from_numpy(batch["frequency"]).to(device)
        raw_context = torch.from_numpy(batch["context_raw"]).to(device)
        baseline = predictor.predict(
            raw_context,
            context,
            target,
            frequency,
            rank_by_budget[budget],
            0.01,
        )
        baseline_encoded = _encode_baseline(baseline, mean, scale, target, frequency)
        prediction_encoded = model(
            torch.from_numpy(batch["context_encoded"]).to(device),
            torch.from_numpy(batch["context_query"]).to(device),
            torch.from_numpy(batch["target_query"]).to(device),
            baseline_encoded,
        )
        target_encoded = torch.from_numpy(batch["target_encoded"]).to(device)
        prediction = _decode(prediction_encoded, mean, scale, target, frequency)
        truth = torch.from_numpy(batch["target_raw"]).to(device)
        relative = torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
            truth, dim=-1
        ).clamp_min(1e-12)
        loss = F.smooth_l1_loss(prediction_encoded, target_encoded) + 0.2 * relative.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.validate_every == 0 or step == args.steps:
            validation = evaluate(
                model,
                archive,
                normalizer,
                predictor,
                rank_by_budget,
                "validation",
                (args.selection_budget,),
                1,
                20260924,
                device,
                max_systems=args.validation_systems,
            )
            validation_error = _summary(validation)[str(args.selection_budget)]["model_mean"]
            record = {
                "step": step,
                "train_loss": float(loss.detach()),
                "train_rel_l2": float(relative.mean().detach()),
                "validation_rel_l2": validation_error,
            }
            history.append(record)
            if validation_error < best_error:
                best_error = validation_error
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "rank_by_budget": rank_by_budget,
                        "step": step,
                        "validation_rel_l2": best_error,
                    },
                    args.output / "best.pt",
                )
            print(json.dumps(record), flush=True)
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_rows = evaluate(
        model,
        archive,
        normalizer,
        predictor,
        rank_by_budget,
        "test",
        budgets,
        3,
        20260924,
        device,
    )
    with (args.output / "test_system_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=test_rows[0].keys())
        writer.writeheader()
        writer.writerows(test_rows)
    summary = {
        "dataset": str(args.dataset.resolve()),
        "seed": args.seed,
        "best_step": checkpoint["step"],
        "best_validation_rel_l2": checkpoint["validation_rel_l2"],
        "elapsed_seconds": time.time() - start_time,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_evaluation_protocol": (
            "three frozen nested context orders per target; system macro-average"
        ),
        "test": _summary(test_rows),
        "history": history,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank-by-budget", required=True)
    parser.add_argument("--selection-budget", type=int, required=True)
    parser.add_argument("--receiver-topology", choices=("line", "periodic"), required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-systems", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
