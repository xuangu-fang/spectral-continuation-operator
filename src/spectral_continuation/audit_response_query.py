"""Information and non-neural baseline audit for wave response queries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .response_query import (
    QueryEpisode,
    ResponseNormalizer,
    complex_relative_l2,
    frozen_episodes,
    load_response_archive,
    normalized_query_coordinates,
    validate_episode,
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
    return encoded.reshape(systems, sources, frequencies, 2, receivers)


class PCATableCompletion:
    def __init__(self, encoded: np.ndarray, train_systems: np.ndarray, rank: int, ridge: float):
        flat = encoded[train_systems].reshape(len(train_systems), -1)
        self.mean = flat.mean(axis=0)
        _, _, right = np.linalg.svd(flat - self.mean, full_matrices=False)
        self.basis = right[:rank].T
        self.ridge = ridge
        self.query_width = encoded.shape[-2] * encoded.shape[-1]
        self.frequency_count = encoded.shape[2]
        self.table_shape = encoded.shape[1:]

    def predict(self, encoded_system: np.ndarray, episode: QueryEpisode, budget: int) -> np.ndarray:
        indices = []
        for source, frequency in episode.context[:budget]:
            start = (source * self.frequency_count + frequency) * self.query_width
            indices.extend(range(start, start + self.query_width))
        indices = np.asarray(indices)
        observed = encoded_system.reshape(-1)[indices]
        design = self.basis[indices]
        gram = design.T @ design + self.ridge * np.eye(self.basis.shape[1])
        coefficient = np.linalg.solve(gram, design.T @ (observed - self.mean[indices]))
        table = (self.mean + self.basis @ coefficient).reshape(self.table_shape)
        return table[episode.target_source, episode.target_frequency]


def _target(archive, episode: QueryEpisode) -> np.ndarray:
    return archive.responses[
        episode.system, episode.target_source, episode.target_frequency
    ]


def _nearest_prediction(archive, coordinates, episode: QueryEpisode, budget: int) -> np.ndarray:
    target_coordinate = coordinates[episode.target_source, episode.target_frequency]
    context = episode.context[:budget]
    distance = np.asarray(
        [np.sum((coordinates[source, frequency] - target_coordinate) ** 2) for source, frequency in context]
    )
    source, frequency = context[int(np.argmin(distance))]
    return archive.responses[episode.system, source, frequency]


def _kernel_prediction(
    archive, coordinates, episode: QueryEpisode, budget: int, bandwidth: float
) -> np.ndarray:
    target_coordinate = coordinates[episode.target_source, episode.target_frequency]
    context = episode.context[:budget]
    distance = np.asarray(
        [np.sum((coordinates[source, frequency] - target_coordinate) ** 2) for source, frequency in context]
    )
    weight = np.exp(-distance / max(2.0 * bandwidth**2, 1e-8))
    weight /= weight.sum().clip(min=1e-12)
    values = np.stack(
        [archive.responses[episode.system, source, frequency] for source, frequency in context]
    )
    return np.einsum("k,kr->r", weight, values)


def _retrieval_prediction(
    archive,
    encoded: np.ndarray,
    train_systems: np.ndarray,
    episode: QueryEpisode,
    budget: int,
) -> tuple[np.ndarray, float]:
    context = episode.context[:budget]
    observed = np.stack(
        [encoded[episode.system, source, frequency].reshape(-1) for source, frequency in context]
    )
    bank = np.stack(
        [
            np.stack([encoded[system, source, frequency].reshape(-1) for source, frequency in context])
            for system in train_systems
        ]
    )
    distance = np.mean((bank - observed[None]) ** 2, axis=(1, 2))
    retrieved = int(train_systems[int(np.argmin(distance))])
    prediction = archive.responses[
        retrieved, episode.target_source, episode.target_frequency
    ]
    return prediction, float(np.min(distance))


def _reciprocity_prediction(
    archive, normalizer, episode: QueryEpisode, budget: int, calibrated: bool = False
):
    """Fill co-located source/receiver entries using Green-function reciprocity."""

    prediction = normalizer.mean[episode.target_source, episode.target_frequency].copy()
    matching = sorted(
        {
            source
            for source, frequency in episode.context[:budget]
            if frequency == episode.target_frequency and source != episode.target_source
        }
    )
    weights = np.ones(len(matching))
    if calibrated and len(matching) > 1:
        design = []
        target = []
        for first_index, first in enumerate(matching):
            for second_index in range(first_index + 1, len(matching)):
                second = matching[second_index]
                row = np.zeros(len(matching))
                row[first_index], row[second_index] = 1.0, -1.0
                first_to_second = archive.responses[
                    episode.system, first, episode.target_frequency, second
                ]
                second_to_first = archive.responses[
                    episode.system, second, episode.target_frequency, first
                ]
                design.append(row)
                target.append(
                    -np.log((abs(first_to_second) + 1e-12) / (abs(second_to_first) + 1e-12))
                )
        design.append(np.ones(len(matching)))
        target.append(0.0)
        log_weight = np.linalg.lstsq(np.stack(design), np.asarray(target), rcond=None)[0]
        weights = np.exp(log_weight)
        weights /= np.median(weights)
    for source, weight in zip(matching, weights, strict=True):
        frequency = episode.target_frequency
        if frequency == episode.target_frequency and source != episode.target_source:
            prediction[source] = weight * archive.responses[
                episode.system, source, frequency, episode.target_source
            ]
    return prediction


def run_audit(
    dataset: Path,
    output: Path,
    receiver_count: int,
    receiver_mode: str,
    episodes_per_system: int,
    seed: int,
) -> dict:
    archive = load_response_archive(dataset, receiver_count, receiver_mode)
    normalizer = ResponseNormalizer(archive)
    encoded = _encoded_table(archive, normalizer)
    coordinates = normalized_query_coordinates(archive)
    train_systems = np.flatnonzero(archive.split == "train")
    validation = frozen_episodes(
        archive,
        split="validation",
        episodes_per_system=episodes_per_system,
        seed=seed,
    )
    for episode in validation:
        validate_episode(episode)

    pca_models = {
        (rank, ridge): PCATableCompletion(encoded, train_systems, rank, ridge)
        for rank in (2, 4, 8, 16)
        for ridge in (0.01, 0.1, 1.0)
    }
    budgets = (1, 2, 4, 8, 16)
    rows = []
    collision_rows = []
    for episode in validation:
        truth = _target(archive, episode)
        source = np.asarray(episode.target_source)
        frequency = np.asarray(episode.target_frequency)
        for budget in budgets:
            predictions = {
                "train_mean": normalizer.mean[episode.target_source, episode.target_frequency],
                "nearest_query": _nearest_prediction(archive, coordinates, episode, budget),
            }
            for bandwidth in (0.25, 0.5, 1.0):
                predictions[f"kernel_h{bandwidth:g}"] = _kernel_prediction(
                    archive, coordinates, episode, budget, bandwidth
                )
            retrieval, context_distance = _retrieval_prediction(
                archive, encoded, train_systems, episode, budget
            )
            predictions["system_retrieval"] = retrieval
            if receiver_mode == "source_array":
                predictions["reciprocity_fill"] = _reciprocity_prediction(
                    archive, normalizer, episode, budget
                )
                predictions["calibrated_reciprocity"] = _reciprocity_prediction(
                    archive, normalizer, episode, budget, calibrated=True
                )
            collision_rows.append(
                {
                    "system": episode.system,
                    "regime": episode.regime,
                    "budget": budget,
                    "context_distance": context_distance,
                    "retrieval_target_error": float(complex_relative_l2(retrieval, truth)),
                }
            )
            for (rank, ridge), model in pca_models.items():
                normalized_prediction = model.predict(encoded[episode.system], episode, budget)
                decoded = normalizer.decode(normalized_prediction, source, frequency)
                predictions[f"pca_r{rank}_l{ridge:g}"] = decoded
            target_norm = float(np.linalg.norm(truth))
            for method, prediction in predictions.items():
                rows.append(
                    {
                        "system": episode.system,
                        "regime": episode.regime,
                        "budget": budget,
                        "method": method,
                        "relative_l2": float(complex_relative_l2(prediction, truth)),
                        "target_norm": target_norm,
                    }
                )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation_episode_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "conditional_collisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=collision_rows[0].keys())
        writer.writeheader()
        writer.writerows(collision_rows)

    summary = {}
    for regime in ("q1", "q2", "q3"):
        summary[regime] = {}
        for budget in budgets:
            selected = [row for row in rows if row["regime"] == regime and row["budget"] == budget]
            methods = sorted({row["method"] for row in selected})
            summary[regime][str(budget)] = {
                method: {
                    "mean": float(np.mean([row["relative_l2"] for row in selected if row["method"] == method])),
                    "median": float(np.median([row["relative_l2"] for row in selected if row["method"] == method])),
                }
                for method in methods
            }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    q3_methods = sorted(summary["q3"]["8"], key=lambda name: summary["q3"]["8"][name]["mean"])[:6]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for method in q3_methods:
        axis.plot(
            budgets,
            [summary["q3"][str(budget)][method]["mean"] for budget in budgets],
            marker="o",
            label=method,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, labels=[str(value) for value in budgets])
    axis.set_xlabel("Context probes K")
    axis.set_ylabel("Raw complex relative L2")
    axis.set_title("Q3 joint-held-out validation audit")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "q3_baseline_scaling.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receivers", type=int, default=64)
    parser.add_argument("--receiver-mode", choices=("cartesian", "source_array"), default="cartesian")
    parser.add_argument("--episodes-per-system", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    summary = run_audit(
        args.dataset,
        args.output,
        args.receivers,
        args.receiver_mode,
        args.episodes_per_system,
        args.seed,
    )
    print(json.dumps({regime: summary[regime]["8"] for regime in summary}, indent=2))


if __name__ == "__main__":
    main()
