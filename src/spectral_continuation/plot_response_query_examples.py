"""Paper-style qualitative examples for the multistatic response-query model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .audit_response_query import _reciprocity_prediction
from .response_query import (
    MaskedResponseTransformer,
    ResponseNormalizer,
    complex_relative_l2,
    frozen_episodes,
    load_response_archive,
    normalized_query_coordinates,
)
from .train_response_query import _encoded_table, reciprocity_baseline_encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    archive = load_response_archive(args.dataset, 9, "source_array")
    normalizer = ResponseNormalizer(archive)
    encoded = _encoded_table(archive, normalizer)
    coordinates = normalized_query_coordinates(archive)
    episodes = frozen_episodes(
        archive, split="test", regimes=("q2",), episodes_per_system=8, seed=20260924
    )
    model = MaskedResponseTransformer(
        9,
        int(config["hidden"]),
        int(config["transformer_layers"]),
        int(config["transformer_heads"]),
    ).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for episode in episodes:
            context = episode.context[: args.budget]
            response = np.stack(
                [encoded[episode.system, source, frequency] for source, frequency in context]
            )[None]
            query = np.stack([coordinates[source, frequency] for source, frequency in context])[None]
            target_query = coordinates[episode.target_source, episode.target_frequency][None]
            correction = model(
                torch.from_numpy(response).cuda(),
                torch.from_numpy(query).cuda(),
                torch.ones(1, len(context), dtype=torch.bool, device="cuda"),
                torch.from_numpy(target_query).cuda(),
            ).cpu().numpy()[0]
            baseline_encoded = reciprocity_baseline_encoded(
                archive, normalizer, episode, args.budget, calibrated=True
            )
            prediction = normalizer.decode(
                (baseline_encoded + correction).reshape(2, -1),
                np.asarray(episode.target_source),
                np.asarray(episode.target_frequency),
            )
            baseline = _reciprocity_prediction(
                archive, normalizer, episode, args.budget, calibrated=True
            )
            truth = archive.responses[
                episode.system, episode.target_source, episode.target_frequency
            ]
            predictions.append(
                (
                    episode,
                    prediction,
                    baseline,
                    truth,
                    float(complex_relative_l2(prediction, truth)),
                    float(complex_relative_l2(baseline, truth)),
                )
            )
    improvement = np.asarray([base - model_error for *_, model_error, base in predictions])
    model_error = np.asarray([value[-2] for value in predictions])
    baseline_error = np.asarray([value[-1] for value in predictions])
    eligible = np.flatnonzero((baseline_error < 1.5) & (model_error < 1.0))
    eligible = eligible[np.argsort(improvement[eligible])]
    choices = [int(eligible[-1]), int(eligible[len(eligible) // 2])]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.5), sharex=True)
    receiver = np.arange(1, 10)
    for row, choice in enumerate(choices):
        episode, prediction, baseline, truth, model_error, baseline_error = predictions[choice]
        axes[row, 0].plot(receiver, np.abs(truth), "o-", color="black", label="Truth")
        axes[row, 0].plot(receiver, np.abs(baseline), "s--", color="#6b7280", label="Reciprocity")
        axes[row, 0].plot(receiver, np.abs(prediction), "o-", color="#0f766e", label="Ours")
        axes[row, 1].plot(receiver, np.angle(truth), "o-", color="black")
        axes[row, 1].plot(receiver, np.angle(baseline), "s--", color="#6b7280")
        axes[row, 1].plot(receiver, np.angle(prediction), "o-", color="#0f766e")
        axes[row, 0].set_ylabel(f"System {episode.system}\nMagnitude")
        axes[row, 1].set_ylabel("Wrapped phase [rad]")
        axes[row, 0].text(
            0.02,
            0.94,
            f"RelL2 {baseline_error:.2f} → {model_error:.2f}",
            transform=axes[row, 0].transAxes,
            va="top",
        )
    axes[0, 0].set_title("Response magnitude")
    axes[0, 1].set_title("Response phase")
    axes[1, 0].set_xlabel("Receiver / transducer index")
    axes[1, 1].set_xlabel("Receiver / transducer index")
    for axis in axes.ravel():
        axis.grid(alpha=0.22)
        axis.set_xticks(receiver)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle("Held-out source query at the observed frequency (K=8)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
