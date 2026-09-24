"""Audits and lightweight learned completion for cross-domain response queries.

The task is Q2: at one frequency, observe complete receiver traces for K
sources and predict the complete trace for a held-out source.  Evaluation is
always macro-averaged over physical systems before confidence intervals are
computed.  The module deliberately starts with linear response subspaces: if
they cannot beat interpolation and reciprocity, a larger neural query model is
not yet justified.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cross_domain_response_data import (
    CrossDomainResponseArchive,
    load_cross_domain_archive,
)


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(prediction - target, axis=-1)
    denominator = np.maximum(np.linalg.norm(target, axis=-1), 1e-12)
    return numerator / denominator


def reciprocity_error(archive: CrossDomainResponseArchive) -> np.ndarray:
    """Per-system reciprocity error on the co-located source sub-array."""

    valid_source = np.flatnonzero(archive.source_receiver_index >= 0)
    if not len(valid_source):
        return np.full(len(archive.responses), np.nan)
    receiver_index = archive.source_receiver_index[valid_source]
    matched = archive.responses[:, valid_source, :, :][..., receiver_index]
    reversed_response = np.swapaxes(matched, 1, 3)
    numerator = np.linalg.norm((matched - reversed_response).reshape(len(matched), -1), axis=-1)
    denominator = np.maximum(
        np.linalg.norm(matched.reshape(len(matched), -1), axis=-1), 1e-12
    )
    return numerator / denominator


def source_orders(source_count: int, repeats: int, seed: int) -> dict[int, list[np.ndarray]]:
    """Frozen nested random context orders for every target source."""

    generator = np.random.default_rng(seed)
    result: dict[int, list[np.ndarray]] = {}
    for target in range(source_count):
        candidates = np.asarray([index for index in range(source_count) if index != target])
        result[target] = []
        for _ in range(repeats):
            result[target].append(generator.permutation(candidates))
    return result


@dataclass(frozen=True)
class FrequencySubspace:
    mean: np.ndarray
    basis: np.ndarray


def _fit_subspaces(archive: CrossDomainResponseArchive, max_rank: int) -> list[FrequencySubspace]:
    train = archive.responses[archive.split == "train"]
    subspaces = []
    for frequency in range(train.shape[2]):
        table = train[:, :, frequency].reshape(len(train), -1)
        mean = table.mean(axis=0)
        _, _, right = np.linalg.svd(table - mean, full_matrices=False)
        # Rows reconstruct as coefficient @ Vh, hence the synthesis matrix is
        # Vh.T (not Vh.conj().T) for complex-valued response tables.
        basis = right[: min(max_rank, len(right))].T
        subspaces.append(FrequencySubspace(mean.astype(np.complex64), basis.astype(np.complex64)))
    return subspaces


def _subspace_prediction(
    subspace: FrequencySubspace,
    response: np.ndarray,
    context: np.ndarray,
    target: int,
    receiver_count: int,
    rank: int,
    ridge: float,
) -> np.ndarray:
    """Complete a source-response table from observed source rows."""

    observed = np.concatenate(
        [np.arange(source * receiver_count, (source + 1) * receiver_count) for source in context]
    )
    target_slice = slice(target * receiver_count, (target + 1) * receiver_count)
    basis = subspace.basis[:, :rank]
    design = basis[observed]
    gram = design.conj().T @ design + ridge * np.eye(rank, dtype=np.complex64)
    projector = np.linalg.solve(gram, design.conj().T)
    values = response[:, context].reshape(len(response), -1)
    coefficient = (values - subspace.mean[observed]) @ projector.T
    return subspace.mean[target_slice] + coefficient @ basis[target_slice].T


def _idw_reciprocity_prediction(
    archive: CrossDomainResponseArchive,
    system_response: np.ndarray,
    context: np.ndarray,
    target: int,
    frequency: int,
    power: float,
) -> np.ndarray:
    """Interpolate exact reciprocal samples of the held-out response."""

    anchor_index = archive.source_receiver_index[context]
    target_receiver = archive.source_receiver_index[target]
    anchor_value = system_response[:, context, frequency, target_receiver]
    anchor_xy = archive.receivers[anchor_index]
    distance = np.linalg.norm(
        archive.receivers[:, None, :] - anchor_xy[None, :, :], axis=-1
    )
    weight = 1.0 / np.maximum(distance, 1e-4) ** power
    weight /= weight.sum(axis=1, keepdims=True)
    prediction = anchor_value @ weight.T
    prediction[:, anchor_index] = anchor_value
    return prediction


def _kernel_probe_prediction(
    archive: CrossDomainResponseArchive,
    system_response: np.ndarray,
    context: np.ndarray,
    target: int,
    frequency: int,
    bandwidth: float,
) -> np.ndarray:
    distance = np.linalg.norm(archive.sources[context] - archive.sources[target], axis=-1)
    weight = np.exp(-(distance**2) / (2.0 * bandwidth**2))
    weight /= np.maximum(weight.sum(), 1e-12)
    return np.einsum("k,nkr->nr", weight, system_response[:, context, frequency])


def _accumulate(
    totals: dict[tuple[str, int], np.ndarray],
    counts: dict[tuple[str, int], int],
    method: str,
    budget: int,
    prediction: np.ndarray,
    truth: np.ndarray,
) -> None:
    key = (method, budget)
    if key not in totals:
        totals[key] = np.zeros(len(truth), dtype=np.float64)
        counts[key] = 0
    totals[key] += relative_l2(prediction, truth)
    counts[key] += 1


def evaluate_candidates(
    archive: CrossDomainResponseArchive,
    split: str,
    budgets: tuple[int, ...],
    subspaces: list[FrequencySubspace],
    ranks: tuple[int, ...],
    ridges: tuple[float, ...],
    repeats: int,
    seed: int,
    learned_choices: dict[int, tuple[int, float]] | None = None,
) -> dict[tuple[str, int], np.ndarray]:
    systems = np.flatnonzero(archive.split == split)
    if not len(systems):
        raise ValueError(f"archive has no {split!r} systems")
    response = archive.responses[systems]
    train_mean = archive.responses[archive.split == "train"].mean(axis=0)
    source_count, frequency_count, receiver_count = response.shape[1:]
    orders = source_orders(source_count, repeats, seed)
    totals: dict[tuple[str, int], np.ndarray] = {}
    counts: dict[tuple[str, int], int] = {}
    for target in range(source_count):
        for order in orders[target]:
            for frequency in range(frequency_count):
                truth = response[:, target, frequency]
                for budget in budgets:
                    context = order[:budget]
                    mean_prediction = np.broadcast_to(train_mean[target, frequency], truth.shape)
                    _accumulate(totals, counts, "train_mean", budget, mean_prediction, truth)

                    distance = np.linalg.norm(
                        archive.sources[context] - archive.sources[target], axis=-1
                    )
                    nearest = int(context[np.argmin(distance)])
                    _accumulate(
                        totals,
                        counts,
                        "nearest_probe",
                        budget,
                        response[:, nearest, frequency],
                        truth,
                    )
                    for bandwidth in (0.15, 0.3, 0.6):
                        prediction = _kernel_probe_prediction(
                            archive, response, context, target, frequency, bandwidth
                        )
                        _accumulate(
                            totals,
                            counts,
                            f"kernel_probe_h{bandwidth:g}",
                            budget,
                            prediction,
                            truth,
                        )
                    for power in (1.0, 2.0, 4.0):
                        prediction = _idw_reciprocity_prediction(
                            archive, response, context, target, frequency, power
                        )
                        _accumulate(
                            totals,
                            counts,
                            f"reciprocity_idw_p{power:g}",
                            budget,
                            prediction,
                            truth,
                        )
                    table = response[:, :, frequency]
                    choices = (
                        [learned_choices[budget]]
                        if learned_choices is not None
                        else [(rank, ridge) for rank in ranks for ridge in ridges]
                    )
                    for rank, ridge in choices:
                        if rank <= subspaces[frequency].basis.shape[1]:
                            prediction = _subspace_prediction(
                                subspaces[frequency],
                                table,
                                context,
                                target,
                                receiver_count,
                                rank,
                                ridge,
                            )
                            _accumulate(
                                totals,
                                counts,
                                f"response_subspace_r{rank}_l{ridge:g}",
                                budget,
                                prediction,
                                truth,
                            )
    return {key: value / counts[key] for key, value in totals.items()}


def _bootstrap_difference(
    baseline: np.ndarray, candidate: np.ndarray, seed: int, draws: int = 20_000
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    improvement = (baseline - candidate) / np.maximum(baseline, 1e-12)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        count = min(1000, draws - start)
        indices = generator.integers(0, len(improvement), size=(count, len(improvement)))
        samples[start : start + count] = improvement[indices].mean(axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "mean_relative_improvement": float(improvement.mean()),
        "bootstrap_95_low": float(lower),
        "bootstrap_95_high": float(upper),
        "positive_system_fraction": float(np.mean(candidate < baseline)),
    }


def _method_family(method: str) -> str:
    return "learned" if method.startswith("response_subspace") else "baseline"


def select_on_validation(
    validation: dict[tuple[str, int], np.ndarray], budgets: tuple[int, ...]
) -> dict[int, dict[str, str]]:
    selection = {}
    for budget in budgets:
        methods = [method for method, value in validation if value == budget]
        baseline = min(
            (method for method in methods if _method_family(method) == "baseline"),
            key=lambda method: float(validation[(method, budget)].mean()),
        )
        learned = min(
            (method for method in methods if _method_family(method) == "learned"),
            key=lambda method: float(validation[(method, budget)].mean()),
        )
        selection[budget] = {"baseline": baseline, "learned": learned}
    return selection


def _parse_learned_selection(selection: dict[int, dict[str, str]]) -> dict[int, tuple[int, float]]:
    parsed = {}
    for budget, methods in selection.items():
        name = methods["learned"]
        rank_text, ridge_text = name.removeprefix("response_subspace_r").split("_l")
        parsed[budget] = (int(rank_text), float(ridge_text))
    return parsed


def _write_rows(
    output: Path,
    domain: str,
    split: str,
    errors: dict[tuple[str, int], np.ndarray],
    system_ids: np.ndarray,
) -> None:
    rows = []
    for (method, budget), values in sorted(errors.items()):
        for system, value in zip(system_ids, values, strict=True):
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "system_id": str(system),
                    "budget": budget,
                    "method": method,
                    "raw_complex_relative_l2": float(value),
                }
            )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_audit(
    dataset: Path,
    output: Path,
    domain: str,
    budgets: tuple[int, ...],
    repeats: int = 3,
    seed: int = 20260924,
) -> dict:
    archive = load_cross_domain_archive(dataset)
    output.mkdir(parents=True, exist_ok=True)
    symmetry = reciprocity_error(archive)
    finite_symmetry = symmetry[np.isfinite(symmetry)]
    source_unique = len(np.unique(archive.sources, axis=0))
    audit = {
        "domain": domain,
        "archive": str(dataset.resolve()),
        "shape": list(archive.shape),
        "unique_source_coordinates": source_unique,
        "reciprocity": {
            "available": bool(len(finite_symmetry)),
            "mean": float(finite_symmetry.mean()) if len(finite_symmetry) else None,
            "median": float(np.median(finite_symmetry)) if len(finite_symmetry) else None,
            "max": float(finite_symmetry.max()) if len(finite_symmetry) else None,
        },
    }
    if not np.any(archive.split == "train"):
        audit["status"] = "semantic_sanity_only_no_train_systems"
        (output / "audit_summary.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        return audit

    max_rank = 64
    ranks = (4, 16, 32)
    ridges = (1e-4, 1e-2)
    subspaces = _fit_subspaces(archive, max_rank)
    validation = evaluate_candidates(
        archive, "validation", budgets, subspaces, ranks, ridges, 1, seed
    )
    selection = select_on_validation(validation, budgets)
    learned_choices = _parse_learned_selection(selection)
    test = evaluate_candidates(
        archive,
        "test",
        budgets,
        subspaces,
        ranks,
        ridges,
        repeats,
        seed,
        learned_choices,
    )
    validation_ids = archive.system_ids[archive.split == "validation"]
    test_ids = archive.system_ids[archive.split == "test"]
    _write_rows(output / "validation_system_metrics.csv", domain, "validation", validation, validation_ids)
    _write_rows(output / "test_system_metrics.csv", domain, "test", test, test_ids)

    results = {}
    for budget in budgets:
        baseline_name = selection[budget]["baseline"]
        learned_name = selection[budget]["learned"]
        baseline = test[(baseline_name, budget)]
        learned = test[(learned_name, budget)]
        results[str(budget)] = {
            "selected_on_validation": selection[budget],
            "baseline_test_mean": float(baseline.mean()),
            "learned_test_mean": float(learned.mean()),
            "paired": _bootstrap_difference(baseline, learned, seed + budget),
        }
    full_budget = max(budgets)
    half_budget = min(budgets, key=lambda value: abs(value - full_budget / 2))
    full_baseline_name = selection[full_budget]["baseline"]
    half_learned_name = selection[half_budget]["learned"]
    full_baseline = test[(full_baseline_name, full_budget)]
    half_learned = test[(half_learned_name, half_budget)]
    audit.update(
        {
            "status": "complete",
            "repeats_per_target": repeats,
            "selection": {str(key): value for key, value in selection.items()},
            "test": results,
            "measurement_efficiency": {
                "half_budget": half_budget,
                "full_budget": full_budget,
                "half_learned_method": half_learned_name,
                "full_baseline_method": full_baseline_name,
                "half_learned_mean": float(half_learned.mean()),
                "full_baseline_mean": float(full_baseline.mean()),
                "ratio": float(half_learned.mean() / full_baseline.mean()),
            },
        }
    )
    if np.any(archive.split == "ood"):
        ood = evaluate_candidates(
            archive,
            "ood",
            budgets,
            subspaces,
            ranks,
            ridges,
            repeats,
            seed,
            learned_choices,
        )
        ood_ids = archive.system_ids[archive.split == "ood"]
        _write_rows(output / "ood_system_metrics.csv", domain, "ood", ood, ood_ids)
        audit["ood"] = {
            str(budget): {
                "baseline_mean": float(ood[(selection[budget]["baseline"], budget)].mean()),
                "learned_mean": float(ood[(selection[budget]["learned"], budget)].mean()),
            }
            for budget in budgets
        }

    (output / "audit_summary.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    x = np.asarray(budgets)
    baseline_y = [results[str(value)]["baseline_test_mean"] for value in budgets]
    learned_y = [results[str(value)]["learned_test_mean"] for value in budgets]
    axis.plot(x, baseline_y, "o-", label="best validation-selected non-neural baseline")
    axis.plot(x, learned_y, "o-", label="response subspace query model")
    axis.set_xlabel("Context source budget K")
    axis.set_ylabel("Raw complex relative L2")
    axis.set_title(f"{domain}: held-out source response query")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "budget_curve.png", dpi=200)
    plt.close(figure)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    summary = run_audit(
        args.dataset,
        args.output,
        args.domain,
        tuple(int(value) for value in args.budgets.split(",")),
        args.repeats,
        args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
