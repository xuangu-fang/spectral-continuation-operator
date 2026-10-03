"""Empirical real-generator Gaussian prior: a strong classical learned control.

Training medium labels fit the prior, exactly as for the coefficient-pretrained
neural encoder. Development/query truths are never used to infer coefficients.
This is not a restart of PCA on oscillatory fields: the coordinates are real
physical generators, constrained by context-derived PDE normal equations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .learned_waveguide_generator import error, evaluate, prepare


def fit_prior(targets, rank):
    values = np.asarray(targets, dtype=float)
    mean = values.mean(axis=0)
    _, singular, right = np.linalg.svd(
        (values - mean).reshape(len(values), -1), full_matrices=False
    )
    rank = min(rank, len(values) - 1, len(singular))
    factor = (right[:rank].T * singular[:rank] / np.sqrt(len(values) - 1)).reshape(
        *mean.shape, rank
    )
    return mean, factor


def unpack_statistics(features, terms):
    """Undo the documented unnormalized feature packing, without target labels."""
    upper = np.triu_indices(terms)
    count = len(upper[0])
    values = np.asarray(features, dtype=float).T
    gram = np.zeros((len(values), terms, terms))
    gram[:, upper[0], upper[1]] = values[:, :count]
    gram[:, upper[1], upper[0]] = values[:, :count]
    rhs = values[:, count : count + terms] * 0.02
    return gram, rhs


def conditional_mean(gram, rhs, mean, factor, ridge):
    if ridge <= 0:
        raise ValueError("strictly positive prior precision required")
    matrix = np.einsum("rdk,rde,rel->kl", factor, gram, factor, optimize=True)
    vector = np.einsum(
        "rdk,rd->k", factor, rhs - np.einsum("rde,re->rd", gram, mean), optimize=True
    )
    weights = np.linalg.solve(matrix + ridge * np.eye(matrix.shape[0]), vector)
    return mean + np.einsum("rdk,k->rd", factor, weights)


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads((Path(config["training_run"]) / "config.json").read_text())
    if set(base["train_systems"]) & set(base["validation_systems"]):
        raise ValueError("training/development overlap")
    (output / "config.json").write_text(json.dumps({**config, "base": base}, indent=2))
    with np.load(Path(config["training_run"]) / "training_statistics.npz") as data:
        target = data["target"].astype(float)
    val, metadata = prepare(base, base["validation_systems"], False)
    statistics = [unpack_statistics(x, base["depth_terms"]) for x in val["features"]]
    start = time.perf_counter()
    mean, full_factor = fit_prior(target, max(config["ranks"]))
    fit_seconds = time.perf_counter() - start
    history, results, best = [], [], float("inf")
    for rank in config["ranks"]:
        factor = full_factor[..., :rank]
        for ridge in config["prior_precisions"]:
            start = time.perf_counter()
            beta = np.array([conditional_mean(g, h, mean, factor, ridge) for g, h in statistics])
            inference = time.perf_counter() - start
            rows = evaluate(beta, metadata, base)
            options = {"rank": rank, "prior_precision": ridge}
            summary = {
                **options,
                "macro_error": error(rows),
                "conditional_inference_seconds_all_systems": inference,
            }
            history.append(summary)
            results.extend({**row, **options} for row in rows)
            print(json.dumps(summary), flush=True)
            if error(rows) < best:
                best = error(rows)
                np.savez_compressed(output / "best_coefficients.npz", coefficients=beta)
                (output / "best_results.json").write_text(
                    json.dumps({"options": options, "results": rows}, indent=2)
                )
            (output / "history.json").write_text(json.dumps(history, indent=2))
            (output / "results.json").write_text(json.dumps(results, indent=2))
    (output / "summary.json").write_text(
        json.dumps(
            {
                "best_development_error": best,
                "training_prior_fit_seconds": fit_seconds,
                "selection": "bounded development query-error selection, not formal test",
                "inference_information": "context normal equations + training medium prior",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
