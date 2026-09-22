"""Non-learned audits for response-conditioned wave-operator representations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm((prediction - target).reshape(len(target), -1), axis=1)
    denominator = np.linalg.norm(target.reshape(len(target), -1), axis=1)
    return numerator / np.maximum(denominator, 1e-30)


def _rbf_weights(
    context: np.ndarray, query: np.ndarray, length: float, ridge: float
) -> np.ndarray:
    distance = ((context[:, None] - context[None]) ** 2).sum(axis=-1)
    kernel = np.exp(-0.5 * distance / length**2) + ridge * np.eye(len(context))
    query_distance = ((query[:, None] - context[None]) ** 2).sum(axis=-1)
    return np.exp(-0.5 * query_distance / length**2) @ np.linalg.inv(kernel)


def _estimate_reciprocity_metric(
    field: np.ndarray, source_pixels: np.ndarray, low_frequency_indices: list[int]
) -> np.ndarray:
    """Estimate positive source weights that symmetrize cross responses.

    If ``u(x;s) = G(x,s) / w(s)`` for a symmetric Green kernel, then
    ``log w(j)-log w(i) = log|u(j;i)|-log|u(i;j)|``.  The overdetermined
    graph system fixes the weights up to a harmless common scale.
    """

    source_count = field.shape[0]
    rows, values = [], []
    for frequency_index in low_frequency_indices:
        for first in range(source_count):
            for second in range(first + 1, source_count):
                first_pixel = source_pixels[first]
                second_pixel = source_pixels[second]
                forward = field[
                    first, frequency_index, second_pixel[0], second_pixel[1]
                ]
                backward = field[
                    second, frequency_index, first_pixel[0], first_pixel[1]
                ]
                if min(abs(forward), abs(backward)) <= 1e-12:
                    continue
                row = np.zeros(source_count)
                row[second] = 1.0
                row[first] = -1.0
                rows.append(row)
                values.append(np.log(abs(forward)) - np.log(abs(backward)))
    rows.append(np.ones(source_count))
    values.append(0.0)
    log_weight = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)[0]
    return np.exp(log_weight)


def _nystrom_prediction(
    fields: np.ndarray,
    frequency_index: int,
    context_sources: np.ndarray,
    query_sources: np.ndarray,
    source_pixels: np.ndarray,
    metric: np.ndarray,
    ridge: float,
) -> np.ndarray:
    columns = fields[context_sources, frequency_index] * metric[
        context_sources, None, None
    ]
    core = np.asarray(
        [
            [
                columns[column, source_pixels[row, 0], source_pixels[row, 1]]
                for column in range(len(columns))
            ]
            for row in context_sources
        ]
    )
    cross = np.asarray(
        [
            [
                columns[column, source_pixels[query, 0], source_pixels[query, 1]]
                for query in query_sources
            ]
            for column in range(len(columns))
        ]
    )
    left, singular, right = np.linalg.svd(core, full_matrices=False)
    inverse = singular / (singular**2 + (ridge * singular[0]) ** 2)
    pseudo_inverse = (right.conj().T * inverse) @ left.conj().T
    symmetric_prediction = columns.reshape(len(columns), -1).T @ pseudo_inverse @ cross
    prediction = symmetric_prediction.T / metric[query_sources, None]
    return prediction.reshape(len(query_sources), *fields.shape[-2:])


def run_audit(dataset: Path, output: Path) -> dict:
    data = np.load(dataset, allow_pickle=True)
    field = data["field"]
    sources = data["sources"]
    frequencies = data["frequencies"]
    split = data["split"].astype(str)
    grid = field.shape[-1]
    source_pixels = np.rint(sources * (grid - 1)).astype(int)
    context_sources = np.arange(6)
    query_sources = np.arange(6, 9)
    low_frequencies = np.arange(4)
    high_frequencies = np.arange(4, 8)

    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    source_grid = [
        (length, ridge)
        for length in (0.08, 0.12, 0.18, 0.25, 0.4, 0.7)
        for ridge in (1e-4, 1e-2, 0.1, 0.5)
    ]
    frequency_grid = [
        (length, ridge)
        for length in (0.08, 0.12, 0.18, 0.25, 0.4, 0.7, 1.2)
        for ridge in (1e-4, 1e-2, 0.1, 0.5)
    ]
    normalized_frequency = (frequencies - frequencies.min()) / (
        frequencies.max() - frequencies.min()
    )

    def evaluate(indices: np.ndarray, method: str, hyper: tuple[float, float]):
        errors = []
        for system in indices:
            if method == "source_rbf":
                weights = _rbf_weights(
                    sources[context_sources], sources[query_sources], *hyper
                )
                for frequency_index in high_frequencies:
                    prediction = np.tensordot(
                        weights, field[system, context_sources, frequency_index], axes=(1, 0)
                    )
                    errors.extend(
                        relative_l2(
                            prediction, field[system, query_sources, frequency_index]
                        )
                    )
            elif method == "frequency_rbf":
                weights = _rbf_weights(
                    normalized_frequency[low_frequencies, None],
                    normalized_frequency[high_frequencies, None],
                    *hyper,
                )
                for source_index in query_sources:
                    prediction = np.tensordot(
                        weights, field[system, source_index, low_frequencies], axes=(1, 0)
                    )
                    errors.extend(
                        relative_l2(
                            prediction, field[system, source_index, high_frequencies]
                        )
                    )
            else:
                raise ValueError(method)
        return np.asarray(errors)

    records = []
    selected = {}
    for method, grid_search in (
        ("source_rbf", source_grid),
        ("frequency_rbf", frequency_grid),
    ):
        candidates = [
            (float(evaluate(validation, method, hyper).mean()), hyper)
            for hyper in grid_search
        ]
        _, hyper = min(candidates)
        selected[method] = {"length": hyper[0], "ridge": hyper[1]}
        errors = evaluate(test, method, hyper)
        records.append(
            {
                "method": method,
                "mean_complex_rel_l2": float(errors.mean()),
                "median_complex_rel_l2": float(np.median(errors)),
            }
        )

    nystrom_candidates = []
    for ridge in (1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0):
        errors = []
        for system in validation:
            metric = _estimate_reciprocity_metric(
                field[system], source_pixels, low_frequencies.tolist()
            )
            for frequency_index in high_frequencies:
                prediction = _nystrom_prediction(
                    field[system], frequency_index, context_sources, query_sources,
                    source_pixels, metric, ridge,
                )
                errors.extend(
                    relative_l2(prediction, field[system, query_sources, frequency_index])
                )
        nystrom_candidates.append((float(np.mean(errors)), ridge))
    _, nystrom_ridge = min(nystrom_candidates)
    selected["metric_nystrom"] = {"ridge": nystrom_ridge}
    errors = []
    metric_symmetry = []
    for system in test:
        metric = _estimate_reciprocity_metric(
            field[system], source_pixels, low_frequencies.tolist()
        )
        weighted_forward, weighted_backward = [], []
        for frequency_index in low_frequencies:
            for first in range(len(sources)):
                for second in range(first + 1, len(sources)):
                    weighted_forward.append(
                        field[
                            system,
                            first,
                            frequency_index,
                            source_pixels[second, 0],
                            source_pixels[second, 1],
                        ]
                        * metric[first]
                    )
                    weighted_backward.append(
                        field[
                            system,
                            second,
                            frequency_index,
                            source_pixels[first, 0],
                            source_pixels[first, 1],
                        ]
                        * metric[second]
                    )
        weighted_forward = np.asarray(weighted_forward)
        weighted_backward = np.asarray(weighted_backward)
        metric_symmetry.append(
            np.linalg.norm(weighted_forward - weighted_backward)
            / np.linalg.norm(weighted_backward)
        )
        for frequency_index in high_frequencies:
            prediction = _nystrom_prediction(
                field[system], frequency_index, context_sources, query_sources,
                source_pixels, metric, nystrom_ridge,
            )
            errors.extend(
                relative_l2(prediction, field[system, query_sources, frequency_index])
            )
    records.append(
        {
            "method": "metric_nystrom",
            "mean_complex_rel_l2": float(np.mean(errors)),
            "median_complex_rel_l2": float(np.median(errors)),
        }
    )

    # Descriptive view-space rank: this is not a trainable result, but tells us
    # whether a single global linear wavefield code is plausible.
    ranks = []
    for system in test:
        matrix = field[system].reshape(72, -1)
        singular = np.linalg.svd(matrix, compute_uv=False)
        energy = np.cumsum(singular**2) / np.sum(singular**2)
        ranks.append(
            {
                "rank90": int(np.searchsorted(energy, 0.90) + 1),
                "rank95": int(np.searchsorted(energy, 0.95) + 1),
                "rank99": int(np.searchsorted(energy, 0.99) + 1),
            }
        )
    summary = {
        "dataset": str(dataset),
        "protocol": "low-frequency all-source plus high-frequency context-source -> missing high-frequency query-source block",
        "selected_on_validation": selected,
        "test_records": records,
        "weighted_reciprocity_rel_l2_mean": float(np.mean(metric_symmetry)),
        "view_rank_mean": {
            key: float(np.mean([row[key] for row in ranks]))
            for key in ("rank90", "rank95", "rank99")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "baselines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.dataset, args.output)


if __name__ == "__main__":
    main()
