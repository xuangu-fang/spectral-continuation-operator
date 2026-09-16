"""Diagnose whether low-frequency wavefields span a reusable global spatial latent space."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .representation_data import load_representation_bundle


def _basis(matrix: torch.Tensor, maximum_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left singular vectors and singular values for a tall complex matrix."""

    vectors, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    rank = min(maximum_rank, vectors.shape[1])
    return vectors[:, :rank], singular_values


def _project(basis: torch.Tensor, target: torch.Tensor, rank: int) -> torch.Tensor:
    current = basis[:, : min(rank, basis.shape[1])]
    return current @ (current.mH @ target)


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    denominator = target.abs().square().sum().clamp_min(1e-12)
    relative = ((prediction - target).abs().square().sum() / denominator).sqrt()
    weight = target.abs()
    coherence = (
        weight * torch.cos(torch.angle(prediction * target.conj()))
    ).sum() / weight.sum().clamp_min(1e-12)
    return float(relative), float(coherence)


def _gauge(
    fields: torch.Tensor,
    frequencies: torch.Tensor,
    visible_count: int,
) -> torch.Tensor:
    """Remove a carrier estimated only from the final two visible frequencies."""

    endpoint = fields[visible_count - 1]
    previous = fields[visible_count - 2]
    step = frequencies[visible_count - 1] - frequencies[visible_count - 2]
    slope = torch.angle(endpoint * previous.conj()) / step
    reference = endpoint / endpoint.abs().clamp_min(1e-8)
    delta = frequencies - frequencies[visible_count - 1]
    carrier = reference[None] * torch.polar(
        torch.ones_like(fields.real), slope[None] * delta[:, None, None]
    )
    return fields * carrier.conj()


def _polynomial_prediction(
    basis: torch.Tensor,
    observed: torch.Tensor,
    observed_frequencies: torch.Tensor,
    query_frequency: float,
    rank: int,
    degree: int,
) -> torch.Tensor:
    """Fit complex latent coefficients as a low-degree frequency polynomial."""

    current = basis[:, : min(rank, basis.shape[1])]
    coefficients = current.mH @ observed
    center = observed_frequencies.mean()
    scale = (observed_frequencies.max() - observed_frequencies.min()).clamp_min(1.0)
    normalized = (observed_frequencies - center) / scale
    degree = min(degree, len(observed_frequencies) - 1)
    design = torch.stack([normalized**power for power in range(degree + 1)], dim=1)
    weights = torch.linalg.pinv(design.to(coefficients.dtype)) @ coefficients.T
    query = torch.tensor(
        (query_frequency - float(center)) / float(scale),
        device=observed.device,
        dtype=observed.real.dtype,
    )
    query_design = torch.stack([query**power for power in range(degree + 1)])
    query_coefficients = query_design.to(weights.dtype) @ weights
    return current @ query_coefficients


def _patch_projection(
    visible: torch.Tensor,
    target: torch.Tensor,
    grid: tuple[int, int],
    ranks: tuple[int, ...],
) -> dict[int, tuple[float, float]]:
    """Project each spatial patch onto its own low-frequency snapshot span."""

    height, width = target.shape
    rows = torch.tensor_split(torch.arange(height, device=target.device), grid[0])
    columns = torch.tensor_split(torch.arange(width, device=target.device), grid[1])
    predictions = {rank: torch.zeros_like(target) for rank in ranks}
    for row in rows:
        for column in columns:
            patch = visible[:, row[:, None], column[None, :]]
            matrix = patch.flatten(1).T
            basis, _ = _basis(matrix, max(ranks))
            target_patch = target[row[:, None], column[None, :]].flatten()
            for rank in ranks:
                prediction = _project(basis, target_patch, rank)
                predictions[rank][row[:, None], column[None, :]] = prediction.reshape(
                    len(row), len(column)
                )
    return {rank: _metrics(prediction, target) for rank, prediction in predictions.items()}


def _energy_rank(singular_values: torch.Tensor, fraction: float) -> int:
    cumulative = singular_values.square().cumsum(0)
    threshold = fraction * cumulative[-1]
    return int(torch.searchsorted(cumulative, threshold).item() + 1)


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    visible = tuple(map(float, args.visible.split(",")))
    queries = tuple(map(float, args.queries.split(",")))
    frequencies = visible + tuple(value for value in queries if value not in visible)
    bundle = load_representation_bundle(
        args.domain,
        args.dataset,
        train_count=args.synthetic_train,
        validation_count=args.synthetic_validation,
        height=args.synthetic_height,
        width=args.synthetic_width,
        frequencies=frequencies,
        seed=args.data_seed,
    )
    positions = np.flatnonzero(bundle.split == "val")[: args.max_environments]
    ranks = tuple(sorted(set(args.ranks)))
    maximum_rank = max(ranks)
    frequency_tensor = torch.tensor(frequencies, device=args.device)
    visible_count = len(visible)
    query_to_index = {frequency: index for index, frequency in enumerate(frequencies)}
    records: list[dict] = []
    rank_records: list[dict] = []

    for environment_index, position in enumerate(positions):
        fields = torch.stack(
            [
                torch.complex(
                    bundle.fields[frequency][position, 0],
                    bundle.fields[frequency][position, 1],
                )
                for frequency in frequencies
            ]
        ).to(args.device)
        representations = {
            "raw": fields,
            "carrier_gauged": _gauge(fields, frequency_tensor, visible_count),
        }
        for representation, current_fields in representations.items():
            visible_matrix = current_fields[:visible_count].flatten(1).T
            full_matrix = current_fields.flatten(1).T
            visible_basis, visible_singular = _basis(visible_matrix, maximum_rank)
            full_basis, full_singular = _basis(full_matrix, maximum_rank)
            rank_records.append(
                {
                    "environment_index": environment_index,
                    "representation": representation,
                    "visible_rank_95": _energy_rank(visible_singular, 0.95),
                    "visible_rank_99": _energy_rank(visible_singular, 0.99),
                    "full_rank_95": _energy_rank(full_singular, 0.95),
                    "full_rank_99": _energy_rank(full_singular, 0.99),
                }
            )
            for frequency in queries:
                target = current_fields[query_to_index[frequency]].flatten()
                for rank in ranks:
                    for basis_name, basis in (
                        ("low_band_span", visible_basis),
                        ("full_band_oracle", full_basis),
                    ):
                        prediction = _project(basis, target, rank)
                        error, coherence = _metrics(prediction, target)
                        records.append(
                            {
                                "environment_index": environment_index,
                                "representation": representation,
                                "method": basis_name,
                                "rank": rank,
                                "frequency_hz": frequency,
                                "complex_rel_l2": error,
                                "phase_coherence": coherence,
                            }
                        )

                polynomial = _polynomial_prediction(
                    visible_basis,
                    visible_matrix,
                    frequency_tensor[:visible_count],
                    frequency,
                    min(args.polynomial_rank, visible_count),
                    args.polynomial_degree,
                )
                error, coherence = _metrics(polynomial, target)
                records.append(
                    {
                        "environment_index": environment_index,
                        "representation": representation,
                        "method": "latent_polynomial",
                        "rank": min(args.polynomial_rank, visible_count),
                        "frequency_hz": frequency,
                        "complex_rel_l2": error,
                        "phase_coherence": coherence,
                    }
                )

            observed_indices = tuple(range(0, visible_count, 2))
            masked_indices = tuple(index for index in range(visible_count) if index % 2)
            observed_matrix = current_fields[list(observed_indices)].flatten(1).T
            observed_basis, _ = _basis(observed_matrix, args.polynomial_rank)
            for index in masked_indices:
                prediction = _polynomial_prediction(
                    observed_basis,
                    observed_matrix,
                    frequency_tensor[list(observed_indices)],
                    frequencies[index],
                    min(args.polynomial_rank, len(observed_indices)),
                    args.polynomial_degree,
                )
                target = current_fields[index].flatten()
                error, coherence = _metrics(prediction, target)
                records.append(
                    {
                        "environment_index": environment_index,
                        "representation": representation,
                        "method": "masked_frequency_polynomial",
                        "rank": min(args.polynomial_rank, len(observed_indices)),
                        "frequency_hz": frequencies[index],
                        "complex_rel_l2": error,
                        "phase_coherence": coherence,
                    }
                )

        query = args.patch_query
        if query in query_to_index:
            for representation, current_fields in representations.items():
                patch_metrics = _patch_projection(
                    current_fields[:visible_count],
                    current_fields[query_to_index[query]],
                    tuple(args.patch_grid),
                    ranks,
                )
                for rank, (error, coherence) in patch_metrics.items():
                    records.append(
                        {
                            "environment_index": environment_index,
                            "representation": representation,
                            "method": f"patch_span_{args.patch_grid[0]}x{args.patch_grid[1]}",
                            "rank": rank,
                            "frequency_hz": query,
                            "complex_rel_l2": error,
                            "phase_coherence": coherence,
                        }
                    )
        print(f"environment {environment_index + 1}/{len(positions)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with (args.output_dir / "projection_records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with (args.output_dir / "rank_records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rank_records[0]))
        writer.writeheader()
        writer.writerows(rank_records)

    aggregate: dict[str, dict] = {}
    keys = sorted(
        {
            (row["representation"], row["method"], row["rank"], row["frequency_hz"])
            for row in records
        }
    )
    for representation, method, rank, frequency in keys:
        selected = [
            row
            for row in records
            if row["representation"] == representation
            and row["method"] == method
            and row["rank"] == rank
            and row["frequency_hz"] == frequency
        ]
        key = f"{representation}/{method}/rank{rank}/f{frequency:g}"
        aggregate[key] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "std": float(np.std([row[metric] for row in selected], ddof=1))
                if len(selected) > 1
                else 0.0,
            }
            for metric in ("complex_rel_l2", "phase_coherence")
        }
    output = {
        "schema_version": 1,
        "status": "linear representation diagnostic; projection methods using target coefficients are oracles",
        "domain": args.domain,
        "visible_frequencies_hz": visible,
        "query_frequencies_hz": queries,
        "validation_environments": len(positions),
        "ranks": ranks,
        "definitions": {
            "low_band_span": "target projected with oracle coefficients onto the per-environment low-band spatial span",
            "full_band_oracle": "target projected onto a basis fitted with all query fields; representation lower bound only",
            "latent_polynomial": "deployable per-environment polynomial extrapolation of low-band latent coefficients",
            "masked_frequency_polynomial": "odd visible frequencies withheld and reconstructed from even frequency tokens",
            "patch_span": "each spatial patch has an independent low-band basis; target coefficients remain oracle",
        },
        "mean_effective_ranks": {
            representation: {
                metric: float(
                    np.mean(
                        [row[metric] for row in rank_records if row["representation"] == representation]
                    )
                )
                for metric in (
                    "visible_rank_95",
                    "visible_rank_99",
                    "full_rank_95",
                    "full_rank_99",
                )
            }
            for representation in ("raw", "carrier_gauged")
        },
        "metrics": aggregate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    _plot(records, ranks, queries, args)


def _plot(records: list[dict], ranks: tuple[int, ...], queries: tuple[float, ...], args) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.25), constrained_layout=True)
    rank = max(ranks)
    for representation, linestyle in (("raw", "--"), ("carrier_gauged", "-")):
        for method, label in (
            ("low_band_span", "Low-band span"),
            ("full_band_oracle", "Full-band oracle"),
        ):
            frequencies, means = [], []
            for frequency in queries:
                selected = [
                    row["complex_rel_l2"]
                    for row in records
                    if row["representation"] == representation
                    and row["method"] == method
                    and row["rank"] == rank
                    and row["frequency_hz"] == frequency
                ]
                if selected:
                    frequencies.append(frequency)
                    means.append(np.mean(selected))
            axes[0].plot(
                frequencies,
                means,
                marker="o",
                markersize=3,
                linestyle=linestyle,
                label=f"{representation}: {label}",
            )
    axes[0].set(
        xlabel="Query frequency (Hz)",
        ylabel="Complex RelL2",
        title=f"Rank {rank} representation horizon",
        yscale="log",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=6)

    for representation in ("raw", "carrier_gauged"):
        means = []
        for current_rank in ranks:
            selected = [
                row["complex_rel_l2"]
                for row in records
                if row["representation"] == representation
                and row["method"] == "low_band_span"
                and row["rank"] == current_rank
                and row["frequency_hz"] == args.patch_query
            ]
            means.append(np.mean(selected))
        axes[1].plot(ranks, means, marker="o", label=representation)
    axes[1].set(
        xlabel="Latent rank",
        ylabel="Complex RelL2",
        title=f"Global span at {args.patch_query:g} Hz",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=7)

    best_representation = min(
        ("raw", "carrier_gauged"),
        key=lambda representation: np.mean(
            [
                row["complex_rel_l2"]
                for row in records
                if row["representation"] == representation
                and row["method"] == "low_band_span"
                and row["rank"] == rank
                and row["frequency_hz"] == args.patch_query
            ]
        ),
    )
    methods = ("low_band_span", f"patch_span_{args.patch_grid[0]}x{args.patch_grid[1]}")
    labels = ("Global", "Patch tokens")
    x = np.arange(len(ranks))
    width = 0.36
    for method_index, (method, label) in enumerate(zip(methods, labels, strict=True)):
        values = []
        for current_rank in ranks:
            selected = [
                row["complex_rel_l2"]
                for row in records
                if row["representation"] == best_representation
                and row["method"] == method
                and row["rank"] == current_rank
                and row["frequency_hz"] == args.patch_query
            ]
            values.append(np.mean(selected))
        axes[2].bar(x + (method_index - 0.5) * width, values, width, label=label)
    axes[2].set_xticks(x, ranks)
    axes[2].set(
        xlabel="Rank per global field / patch",
        ylabel="Complex RelL2",
        title=f"Spatial tokenization ({best_representation})",
    )
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].grid(axis="y", alpha=0.2)
    figure.savefig(args.output_dir / "global_subspace.png", dpi=250)
    figure.savefig(args.output_dir / "global_subspace.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("ocean", "synthetic"), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visible", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--polynomial-rank", type=int, default=6)
    parser.add_argument("--polynomial-degree", type=int, default=3)
    parser.add_argument("--patch-grid", type=int, nargs=2, default=(4, 8))
    parser.add_argument("--patch-query", type=float, required=True)
    parser.add_argument("--max-environments", type=int, default=32)
    parser.add_argument("--synthetic-train", type=int, default=64)
    parser.add_argument("--synthetic-validation", type=int, default=32)
    parser.add_argument("--synthetic-height", type=int, default=32)
    parser.add_argument("--synthetic-width", type=int, default=64)
    parser.add_argument("--data-seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
