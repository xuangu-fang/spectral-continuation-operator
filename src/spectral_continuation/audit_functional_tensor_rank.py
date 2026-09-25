"""Validation-only functional-rank audit for Wave-as-Prompt response bases."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cross_domain_query_audit import FrequencySubspace, _fit_subspaces, source_orders
from .cross_domain_response_data import CrossDomainResponseArchive, load_cross_domain_archive


@dataclass(frozen=True)
class DomainSpec:
    name: str
    dataset: Path
    budget: int
    latent_rank: int
    ridge: float


def reciprocal_projection(table: np.ndarray, receiver_index: np.ndarray) -> np.ndarray:
    """Project co-located entries onto complex symmetry, without conjugation."""

    result = np.array(table, copy=True)
    valid = np.flatnonzero(receiver_index >= 0)
    if not len(valid):
        return result
    matched_receiver = receiver_index[valid]
    block = result[..., valid, :][..., matched_receiver]
    symmetric = 0.5 * (block + np.swapaxes(block, -1, -2))
    for local_source, source in enumerate(valid):
        result[..., source, matched_receiver] = symmetric[..., local_source, :]
    return result


def matrix_rank_approximation(table: np.ndarray, rank: int) -> np.ndarray:
    """Best complex matrix-rank approximation under Frobenius norm."""

    left, singular, right = np.linalg.svd(table, full_matrices=False)
    kept = min(rank, len(singular))
    return (left[:, :kept] * singular[:kept]) @ right[:kept]


def factorize_subspace(
    subspace: FrequencySubspace,
    source_count: int,
    receiver_count: int,
    latent_rank: int,
    functional_rank: int,
    receiver_index: np.ndarray,
    enforce_reciprocity: bool,
) -> FrequencySubspace:
    """Factor every environment basis table along source and receiver axes."""

    entries = [subspace.mean]
    entries.extend(subspace.basis[:, index] for index in range(latent_rank))
    approximated = []
    for entry in entries:
        table = entry.reshape(source_count, receiver_count)
        low_rank = matrix_rank_approximation(table, functional_rank)
        if enforce_reciprocity:
            low_rank = reciprocal_projection(low_rank, receiver_index)
        approximated.append(low_rank.reshape(-1))
    mean = approximated[0].astype(np.complex64)
    basis = np.stack(approximated[1:], axis=-1).astype(np.complex64)
    return FrequencySubspace(mean=mean, basis=basis)


def representation_storage(
    source_count: int,
    receiver_count: int,
    latent_rank: int,
    functional_rank: int,
) -> tuple[int, int, float]:
    """Complex scalar storage, excluding future coordinate-network parameters."""

    dense = (latent_rank + 1) * source_count * receiver_count
    factorized = (latent_rank + 1) * functional_rank * (source_count + receiver_count)
    return dense, factorized, factorized / dense


def shared_tucker_subspace(
    subspace: FrequencySubspace,
    source_count: int,
    receiver_count: int,
    latent_rank: int,
    source_rank: int,
    receiver_rank: int,
    receiver_index: np.ndarray,
    enforce_reciprocity: bool,
) -> FrequencySubspace:
    """Fit shared source/receiver spaces with a basis-specific Tucker core."""

    entries = np.concatenate(
        (subspace.mean[:, None], subspace.basis[:, :latent_rank]), axis=1
    ).T.reshape(latent_rank + 1, source_count, receiver_count)
    source_unfolding = np.transpose(entries, (1, 0, 2)).reshape(source_count, -1)
    receiver_unfolding = np.transpose(entries, (2, 0, 1)).reshape(receiver_count, -1)
    source_functions = np.linalg.svd(source_unfolding, full_matrices=False)[0][
        :, : min(source_rank, source_count)
    ]
    receiver_functions = np.linalg.svd(receiver_unfolding, full_matrices=False)[0][
        :, : min(receiver_rank, receiver_count)
    ]
    core = np.einsum(
        "sp,ksr,rq->kpq",
        source_functions.conj(),
        entries,
        receiver_functions,
        optimize=True,
    )
    approximation = np.einsum(
        "sp,kpq,rq->ksr",
        source_functions,
        core,
        receiver_functions.conj(),
        optimize=True,
    )
    if enforce_reciprocity:
        approximation = reciprocal_projection(approximation, receiver_index)
    flattened = approximation.reshape(latent_rank + 1, -1)
    return FrequencySubspace(
        mean=flattened[0].astype(np.complex64),
        basis=flattened[1:].T.astype(np.complex64),
    )


def tucker_storage(
    source_count: int,
    receiver_count: int,
    latent_rank: int,
    source_rank: int,
    receiver_rank: int,
) -> tuple[int, int, float]:
    dense = (latent_rank + 1) * source_count * receiver_count
    factorized = (
        source_count * source_rank
        + receiver_count * receiver_rank
        + (latent_rank + 1) * source_rank * receiver_rank
    )
    return dense, factorized, factorized / dense


def _fit_coefficients(values: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    gram = basis.conj().T @ basis
    scale = float(np.trace(gram).real / max(len(gram), 1))
    regularizer = max(scale, 1e-12) * 1e-8
    projector = np.linalg.solve(
        gram + regularizer * np.eye(len(gram), dtype=gram.dtype), basis.conj().T
    )
    return (values - mean) @ projector.T


def full_table_errors(
    archive: CrossDomainResponseArchive,
    subspaces: list[FrequencySubspace],
    split: str,
) -> np.ndarray:
    systems = np.flatnonzero(archive.split == split)
    response = archive.responses[systems]
    squared_error = np.zeros(len(systems), dtype=np.float64)
    squared_target = np.zeros(len(systems), dtype=np.float64)
    for frequency, subspace in enumerate(subspaces):
        truth = response[:, :, frequency].reshape(len(systems), -1)
        coefficient = _fit_coefficients(truth, subspace.mean, subspace.basis)
        prediction = subspace.mean + coefficient @ subspace.basis.T
        squared_error += np.sum(np.abs(prediction - truth) ** 2, axis=-1)
        squared_target += np.sum(np.abs(truth) ** 2, axis=-1)
    return np.sqrt(squared_error / np.maximum(squared_target, 1e-24))


def _query_prediction(
    subspace: FrequencySubspace,
    response: np.ndarray,
    context: np.ndarray,
    target: int,
    receiver_count: int,
    ridge: float,
) -> np.ndarray:
    rank = subspace.basis.shape[1]
    observed = np.concatenate(
        [np.arange(source * receiver_count, (source + 1) * receiver_count) for source in context]
    )
    target_slice = slice(target * receiver_count, (target + 1) * receiver_count)
    design = subspace.basis[observed]
    gram = design.conj().T @ design + ridge * np.eye(rank, dtype=np.complex64)
    projector = np.linalg.solve(gram, design.conj().T)
    values = response[:, context].reshape(len(response), -1)
    coefficient = (values - subspace.mean[observed]) @ projector.T
    return subspace.mean[target_slice] + coefficient @ subspace.basis[target_slice].T


def prompt_query_errors(
    archive: CrossDomainResponseArchive,
    subspaces: list[FrequencySubspace],
    split: str,
    budget: int,
    ridge: float,
    repeats: int,
    seed: int,
) -> np.ndarray:
    systems = np.flatnonzero(archive.split == split)
    response = archive.responses[systems]
    source_count, frequency_count, receiver_count = response.shape[1:]
    orders = source_orders(source_count, repeats, seed)
    squared_error = np.zeros(len(systems), dtype=np.float64)
    squared_target = np.zeros(len(systems), dtype=np.float64)
    for target in range(source_count):
        for order in orders[target]:
            context = order[:budget]
            for frequency in range(frequency_count):
                truth = response[:, target, frequency]
                prediction = _query_prediction(
                    subspaces[frequency],
                    response[:, :, frequency],
                    context,
                    target,
                    receiver_count,
                    ridge,
                )
                squared_error += np.sum(np.abs(prediction - truth) ** 2, axis=-1)
                squared_target += np.sum(np.abs(truth) ** 2, axis=-1)
    return np.sqrt(squared_error / np.maximum(squared_target, 1e-24))


def reciprocity_residual(
    subspaces: list[FrequencySubspace],
    source_count: int,
    receiver_count: int,
    receiver_index: np.ndarray,
) -> float:
    valid = np.flatnonzero(receiver_index >= 0)
    if not len(valid):
        return float("nan")
    matched_receiver = receiver_index[valid]
    numerator = 0.0
    denominator = 0.0
    for subspace in subspaces:
        entries = np.concatenate((subspace.mean[:, None], subspace.basis), axis=1)
        tables = entries.T.reshape(-1, source_count, receiver_count)
        block = tables[:, valid][:, :, matched_receiver]
        numerator += float(np.sum(np.abs(block - np.swapaxes(block, 1, 2)) ** 2))
        denominator += float(np.sum(np.abs(block) ** 2))
    return float(np.sqrt(numerator / max(denominator, 1e-24)))


def audit_domain(
    spec: DomainSpec,
    functional_ranks: tuple[int, ...],
    tucker_receiver_ranks: tuple[int, ...],
    repeats: int,
    seed: int,
) -> list[dict[str, float | int | str | bool]]:
    archive = load_cross_domain_archive(spec.dataset)
    source_count, _, receiver_count = archive.responses.shape[1:]
    dense_subspaces = _fit_subspaces(archive, max_rank=spec.latent_rank)
    dense_full = full_table_errors(archive, dense_subspaces, "validation")
    dense_query = prompt_query_errors(
        archive,
        dense_subspaces,
        "validation",
        spec.budget,
        spec.ridge,
        repeats,
        seed,
    )
    dense_reciprocity = reciprocity_residual(
        dense_subspaces, source_count, receiver_count, archive.source_receiver_index
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for functional_rank in functional_ranks:
        factorized_subspaces = [
            factorize_subspace(
                subspace,
                source_count,
                receiver_count,
                spec.latent_rank,
                functional_rank,
                archive.source_receiver_index,
                enforce_reciprocity=True,
            )
            for subspace in dense_subspaces
        ]
        factor_full = full_table_errors(archive, factorized_subspaces, "validation")
        factor_query = prompt_query_errors(
            archive,
            factorized_subspaces,
            "validation",
            spec.budget,
            spec.ridge,
            repeats,
            seed,
        )
        dense_storage, factor_storage, storage_ratio = representation_storage(
            source_count,
            receiver_count,
            spec.latent_rank,
            functional_rank,
        )
        factor_reciprocity = reciprocity_residual(
            factorized_subspaces,
            source_count,
            receiver_count,
            archive.source_receiver_index,
        )
        rows.append(
            {
                "domain": spec.name,
                "method": "independent_cp",
                "functional_rank": functional_rank,
                "source_rank": functional_rank,
                "receiver_rank": functional_rank,
                "latent_rank": spec.latent_rank,
                "budget": spec.budget,
                "dense_storage_complex": dense_storage,
                "factor_storage_complex": factor_storage,
                "storage_ratio": storage_ratio,
                "dense_full_rel_l2": float(dense_full.mean()),
                "factor_full_rel_l2": float(factor_full.mean()),
                "full_error_ratio": float(factor_full.mean() / dense_full.mean()),
                "dense_query_rel_l2": float(dense_query.mean()),
                "factor_query_rel_l2": float(factor_query.mean()),
                "query_error_ratio": float(factor_query.mean() / dense_query.mean()),
                "dense_reciprocity_residual": dense_reciprocity,
                "factor_reciprocity_residual": factor_reciprocity,
                "storage_gate": bool(storage_ratio <= 0.5),
                "full_error_gate": bool(factor_full.mean() <= 1.10 * dense_full.mean()),
                "query_no_collapse": bool(factor_query.mean() <= 1.25 * dense_query.mean()),
            }
        )
    for receiver_rank in tucker_receiver_ranks:
        if receiver_rank > receiver_count:
            continue
        source_rank = source_count
        factorized_subspaces = [
            shared_tucker_subspace(
                subspace,
                source_count,
                receiver_count,
                spec.latent_rank,
                source_rank,
                receiver_rank,
                archive.source_receiver_index,
                enforce_reciprocity=True,
            )
            for subspace in dense_subspaces
        ]
        factor_full = full_table_errors(archive, factorized_subspaces, "validation")
        factor_query = prompt_query_errors(
            archive,
            factorized_subspaces,
            "validation",
            spec.budget,
            spec.ridge,
            repeats,
            seed,
        )
        dense_storage, factor_storage, storage_ratio = tucker_storage(
            source_count,
            receiver_count,
            spec.latent_rank,
            source_rank,
            receiver_rank,
        )
        factor_reciprocity = reciprocity_residual(
            factorized_subspaces,
            source_count,
            receiver_count,
            archive.source_receiver_index,
        )
        rows.append(
            {
                "domain": spec.name,
                "method": "shared_tucker",
                "functional_rank": receiver_rank,
                "source_rank": source_rank,
                "receiver_rank": receiver_rank,
                "latent_rank": spec.latent_rank,
                "budget": spec.budget,
                "dense_storage_complex": dense_storage,
                "factor_storage_complex": factor_storage,
                "storage_ratio": storage_ratio,
                "dense_full_rel_l2": float(dense_full.mean()),
                "factor_full_rel_l2": float(factor_full.mean()),
                "full_error_ratio": float(factor_full.mean() / dense_full.mean()),
                "dense_query_rel_l2": float(dense_query.mean()),
                "factor_query_rel_l2": float(factor_query.mean()),
                "query_error_ratio": float(factor_query.mean() / dense_query.mean()),
                "dense_reciprocity_residual": dense_reciprocity,
                "factor_reciprocity_residual": factor_reciprocity,
                "storage_gate": bool(storage_ratio <= 0.5),
                "full_error_gate": bool(factor_full.mean() <= 1.10 * dense_full.mean()),
                "query_no_collapse": bool(factor_query.mean() <= 1.25 * dense_query.mean()),
            }
        )
    return rows


def _plot(rows: list[dict[str, float | int | str | bool]], output: Path) -> None:
    domains = list(dict.fromkeys(str(row["domain"]) for row in rows))
    figure, axes = plt.subplots(1, len(domains), figsize=(5.0 * len(domains), 4.0), squeeze=False)
    for axis, domain in zip(axes[0], domains):
        selected = [row for row in rows if row["domain"] == domain]
        for method, marker in (("independent_cp", "o"), ("shared_tucker", "s")):
            method_rows = [row for row in selected if row["method"] == method]
            method_storage = np.asarray(
                [row["storage_ratio"] for row in method_rows], dtype=float
            )
            axis.plot(
                method_storage,
                [row["full_error_ratio"] for row in method_rows],
                marker + "-",
                label=f"{method}: full",
            )
            axis.plot(
                method_storage,
                [row["query_error_ratio"] for row in method_rows],
                marker + "--",
                label=f"{method}: query",
            )
        axis.axhline(1.0, color="black", linewidth=1, alpha=0.5)
        axis.axhline(1.1, color="tab:green", linewidth=1, linestyle="--", alpha=0.7)
        axis.axvline(0.5, color="tab:gray", linewidth=1, linestyle=":")
        axis.set_title(domain)
        axis.set_xlabel("factorized / dense complex storage")
        axis.set_ylabel("error / dense-subspace error")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(
    specs: tuple[DomainSpec, ...],
    output: Path,
    functional_ranks: tuple[int, ...],
    tucker_receiver_ranks: tuple[int, ...],
    repeats: int,
    seed: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        rows.extend(
            audit_domain(spec, functional_ranks, tucker_receiver_ranks, repeats, seed)
        )
    with (output / "functional_rank_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, output / "functional_rank_pareto.png")
    domain_gate = {}
    for spec in specs:
        candidates = [
            row
            for row in rows
            if row["domain"] == spec.name and row["storage_gate"] and row["full_error_gate"]
        ]
        domain_gate[spec.name] = {
            "passed": bool(candidates),
            "best_candidate": min(candidates, key=lambda row: row["query_error_ratio"])
            if candidates
            else None,
        }
    summary = {
        "protocol": {
            "split": "validation only",
            "repeats": repeats,
            "seed": seed,
            "functional_ranks": list(functional_ranks),
            "tucker_receiver_ranks": list(tucker_receiver_ranks),
            "reciprocity": "complex-symmetric output projection on co-located sub-array",
        },
        "domain_gate": domain_gate,
        "passed_domains": sum(int(value["passed"]) for value in domain_gate.values()),
        "m0_storage_full_error_gate": sum(int(value["passed"]) for value in domain_gate.values())
        >= 2,
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    specs = tuple(
        DomainSpec(
            name=name,
            dataset=Path(item["dataset"]),
            budget=int(item["budget"]),
            latent_rank=int(item["latent_rank"]),
            ridge=float(item["ridge"]),
        )
        for name, item in payload["domains"].items()
    )
    summary = run(
        specs,
        args.output,
        tuple(int(rank) for rank in payload["functional_ranks"]),
        tuple(int(rank) for rank in payload["tucker_receiver_ranks"]),
        int(payload["repeats"]),
        int(payload["seed"]),
    )
    print(json.dumps(summary["domain_gate"], indent=2))


if __name__ == "__main__":
    main()
