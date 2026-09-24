"""Complete-table oracle audit for active source selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .audit_response_query import _reciprocity_prediction
from .response_query import (
    QueryEpisode,
    ResponseNormalizer,
    complex_relative_l2,
    load_response_archive,
)


def evaluate_order(archive, normalizer, system, target_source, target_frequency, order):
    truth = archive.responses[system, target_source, target_frequency]
    errors = []
    for budget in range(1, len(order) + 1):
        episode = QueryEpisode(
            system,
            target_source,
            target_frequency,
            tuple((source, target_frequency) for source in order[:budget]),
            "q2",
        )
        prediction = _reciprocity_prediction(
            archive, normalizer, episode, budget, calibrated=True
        )
        errors.append(float(complex_relative_l2(prediction, truth)))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-system", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    archive = load_response_archive(args.dataset, 9, "source_array")
    normalizer = ResponseNormalizer(archive)
    systems = np.flatnonzero(archive.split == "test")
    generator = np.random.default_rng(args.seed)
    curves = {"random": [], "nearest_first": [], "oracle": []}
    for system in systems:
        for _ in range(args.episodes_per_system):
            target_source = int(generator.integers(archive.query_shape[0]))
            target_frequency = int(generator.integers(archive.query_shape[1]))
            candidates = [source for source in range(archive.query_shape[0]) if source != target_source]
            generator.shuffle(candidates)
            curves["random"].append(
                evaluate_order(
                    archive,
                    normalizer,
                    int(system),
                    target_source,
                    target_frequency,
                    candidates,
                )
            )
            nearest = sorted(
                candidates,
                key=lambda source: np.sum(
                    (archive.sources[source] - archive.sources[target_source]) ** 2
                ),
            )
            curves["nearest_first"].append(
                evaluate_order(
                    archive,
                    normalizer,
                    int(system),
                    target_source,
                    target_frequency,
                    nearest,
                )
            )
            selected = []
            remaining = set(candidates)
            oracle_curve = []
            for _ in candidates:
                choices = []
                for source in remaining:
                    order = selected + [source]
                    error = evaluate_order(
                        archive,
                        normalizer,
                        int(system),
                        target_source,
                        target_frequency,
                        order,
                    )[-1]
                    choices.append((error, source))
                error, source = min(choices)
                selected.append(source)
                remaining.remove(source)
                oracle_curve.append(error)
            curves["oracle"].append(oracle_curve)

    mean_curves = {name: np.mean(value, axis=0) for name, value in curves.items()}
    auc = {name: float(np.mean(value)) for name, value in mean_curves.items()}
    best_fixed = min(("random", "nearest_first"), key=lambda name: auc[name])
    improvement = 1.0 - auc["oracle"] / auc[best_fixed]
    result = {
        "episodes": int(len(systems) * args.episodes_per_system),
        "mean_error": {name: value.tolist() for name, value in mean_curves.items()},
        "error_auc": auc,
        "best_fixed": best_fixed,
        "oracle_relative_auc_improvement": float(improvement),
        "active_gate_15_percent": bool(improvement >= 0.15),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    style = {"random": "--", "nearest_first": "-.", "oracle": "-"}
    for name, value in mean_curves.items():
        axis.plot(range(1, 9), value, style[name], marker="o", label=name)
    axis.set_xlabel("Acquired source probes")
    axis.set_ylabel("Calibrated-reciprocity RelL2")
    axis.set_title("Best-next-source oracle headroom")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "active_source_oracle.png", dpi=200)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
