"""Train a matched generic conditional neural process for strict Q3 queries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import _q3_events
from .train_cross_domain_query import QueryNormalizer, ReceiverQueryOperator


def query_coordinates(archive) -> np.ndarray:
    source = np.asarray(archive.sources, dtype=np.float32)
    if source.ndim == 1:
        source = source[:, None]
    source = np.pad(source, ((0, 0), (0, max(0, 2 - source.shape[1]))))[:, :2]
    low = source.min(axis=0)
    span = np.maximum(source.max(axis=0) - low, 1e-12)
    source = 2.0 * (source - low) / span - 1.0
    frequency = np.log(np.asarray(archive.frequencies, dtype=np.float32))
    frequency = 2.0 * (frequency - frequency.min()) / max(
        float(np.ptp(frequency)), 1e-12
    ) - 1.0
    return np.concatenate(
        (
            np.repeat(source[:, None], len(frequency), axis=1),
            np.repeat(frequency[None, :, None], len(source), axis=0),
        ),
        axis=-1,
    ).astype(np.float32)


def _encode(
    response: np.ndarray,
    source: np.ndarray,
    frequency: np.ndarray,
    normalizer: QueryNormalizer,
) -> np.ndarray:
    normalized = (
        response - normalizer.mean[source, frequency]
    ) / normalizer.scale[source, frequency]
    return np.stack((normalized.real, normalized.imag), axis=-2).astype(np.float32)


def _decode(
    encoded: torch.Tensor,
    source: int,
    frequency: int,
    normalizer: QueryNormalizer,
    device: torch.device,
) -> torch.Tensor:
    mean = torch.as_tensor(normalizer.mean[source, frequency], device=device)
    scale = torch.as_tensor(normalizer.scale[source, frequency], device=device)
    return mean[None] + scale[None] * torch.complex(encoded[:, 0], encoded[:, 1])


@torch.inference_mode()
def evaluate(
    model: ReceiverQueryOperator,
    archive,
    normalizer: QueryNormalizer,
    coordinates: np.ndarray,
    split: str,
    budgets: tuple[int, ...],
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    model.eval()
    responses = archive.responses[archive.split == split]
    source_count, frequency_count, receiver_count = archive.responses.shape[1:]
    result = {}
    for budget in budgets:
        contexts, targets = _q3_events(
            source_count, frequency_count, budget, repeats, seed
        )
        total = torch.zeros(len(responses), device=device)
        for context, target in zip(contexts, targets, strict=True):
            context_raw = responses[:, context[:, 0], context[:, 1]]
            context_encoded = torch.as_tensor(
                _encode(
                    context_raw,
                    context[:, 0][None],
                    context[:, 1][None],
                    normalizer,
                ),
                device=device,
            )
            context_query = torch.as_tensor(
                coordinates[context[:, 0], context[:, 1]], device=device
            )[None].expand(len(responses), -1, -1)
            target_query = torch.as_tensor(
                coordinates[target[0], target[1]], device=device
            )[None].expand(len(responses), -1)
            prediction_encoded = model(
                context_encoded,
                context_query,
                target_query,
                torch.zeros(
                    len(responses), 2, receiver_count, device=device
                ),
            )
            prediction = _decode(
                prediction_encoded,
                int(target[0]),
                int(target[1]),
                normalizer,
                device,
            )
            truth = torch.as_tensor(
                responses[:, target[0], target[1]], device=device
            )
            total += torch.linalg.vector_norm(
                prediction - truth, dim=-1
            ) / torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)
        result[budget] = (total / len(contexts)).cpu().numpy().astype(np.float64)
    return result


def train_domain(
    domain: str,
    config: dict,
    output: Path,
    device_name: str,
    resume: Path | None = None,
) -> dict:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = np.random.default_rng(seed)
    device = torch.device(device_name)
    setting = config["domains"][domain]
    archive = load_cross_domain_archive(Path(setting["dataset"]))
    normalizer = QueryNormalizer(archive.responses, archive.split == "train")
    coordinates = query_coordinates(archive)
    model = ReceiverQueryOperator(
        archive.receivers,
        int(config["width"]),
        bool(setting.get("periodic_receiver", False)),
    ).to(device)
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    budgets = tuple(int(value) for value in config["budgets"])
    train_systems = np.flatnonzero(archive.split == "train")
    source_count, frequency_count, receiver_count = archive.responses.shape[1:]
    batch_size = int(config["batch_size"])
    validation_initial = evaluate(
        model,
        archive,
        normalizer,
        coordinates,
        "validation",
        budgets,
        int(config["validation_repeats"]),
        seed,
        device,
    )
    initial_mean = {
        str(budget): float(values.mean())
        for budget, values in validation_initial.items()
    }
    best_score = 1.0
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    history = [{"step": 0, "score": 1.0, **initial_mean}]
    started = time.perf_counter()
    for step in range(1, int(config["steps"]) + 1):
        model.train()
        budget = int(generator.choice(budgets))
        systems = generator.choice(train_systems, size=batch_size)
        target_source = generator.integers(source_count, size=batch_size)
        target_frequency = generator.integers(frequency_count, size=batch_size)
        context_source = np.empty((batch_size, budget), dtype=np.int64)
        context_frequency = np.empty_like(context_source)
        for index in range(batch_size):
            candidates = np.asarray(
                [
                    (source, frequency)
                    for source in range(source_count)
                    for frequency in range(frequency_count)
                    if source != target_source[index]
                    and frequency != target_frequency[index]
                ],
                dtype=np.int64,
            )
            selected = candidates[
                generator.permutation(len(candidates))[:budget]
            ]
            context_source[index] = selected[:, 0]
            context_frequency[index] = selected[:, 1]
        context_raw = archive.responses[
            systems[:, None], context_source, context_frequency
        ]
        context_encoded = torch.as_tensor(
            _encode(
                context_raw, context_source, context_frequency, normalizer
            ),
            device=device,
        )
        context_query = torch.as_tensor(
            coordinates[context_source, context_frequency], device=device
        )
        target_query = torch.as_tensor(
            coordinates[target_source, target_frequency], device=device
        )
        prediction_encoded = model(
            context_encoded,
            context_query,
            target_query,
            torch.zeros(batch_size, 2, receiver_count, device=device),
        )
        mean = torch.as_tensor(
            normalizer.mean[target_source, target_frequency], device=device
        )
        scale = torch.as_tensor(
            normalizer.scale[target_source, target_frequency], device=device
        )
        prediction = mean + scale * torch.complex(
            prediction_encoded[:, 0], prediction_encoded[:, 1]
        )
        truth = torch.as_tensor(
            archive.responses[systems, target_source, target_frequency],
            device=device,
        )
        loss = (
            torch.linalg.vector_norm(prediction - truth, dim=-1)
            / torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0:
            values = evaluate(
                model,
                archive,
                normalizer,
                coordinates,
                "validation",
                budgets,
                int(config["validation_repeats"]),
                seed,
                device,
            )
            means = {
                str(budget_value): float(metric.mean())
                for budget_value, metric in values.items()
            }
            score = float(
                np.mean(
                    [means[str(value)] / initial_mean[str(value)] for value in budgets]
                )
            )
            history.append(
                {"step": step, "score": score, "loss": float(loss.detach()), **means}
            )
            if score < best_score:
                best_score = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    metrics = {}
    output.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test", "ood"):
        if not np.any(archive.split == split):
            continue
        values = evaluate(
            model,
            archive,
            normalizer,
            coordinates,
            split,
            budgets,
            int(
                config["validation_repeats"]
                if split == "validation"
                else config["test_repeats"]
            ),
            seed,
            device,
        )
        metrics[split] = {
            str(budget): float(value.mean()) for budget, value in values.items()
        }
        np.savez_compressed(
            output / f"{split}_system_metrics.npz",
            **{f"k{budget}": value for budget, value in values.items()},
        )
    torch.save(
        {"model": model.state_dict(), "config": config, "domain": domain},
        output / "best.pt",
    )
    summary = {
        "domain": domain,
        "task": "strict Q3 generic CNP baseline",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_ratio": best_score,
        "metrics": metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    print(
        json.dumps(
            train_domain(
                args.domain, config, args.output, args.device, args.resume
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
