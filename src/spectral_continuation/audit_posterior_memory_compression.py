"""Validation-only prototype compression audit for posterior memory operators."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import GlobalConfig, GlobalQ3Evaluator, fit_global_subspace
from .posterior_memory_response_operator import MemoryConfig, evaluate_memory_config_batched
from .probabilistic_response_operator import paired_summary


def farthest_point_indices(features: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic greedy k-center indices, starting farthest from the mean."""

    if count >= len(features):
        return torch.arange(len(features), device=features.device)
    squared_norm = features.abs().square().sum(dim=-1)
    center = features.mean(dim=0, keepdim=True)
    first = torch.argmax((features - center).abs().square().sum(dim=-1))
    selected = torch.empty(count, dtype=torch.long, device=features.device)
    selected[0] = first
    minimum = (
        squared_norm
        + squared_norm[first]
        - 2.0 * torch.real(features @ features[first].conj())
    ).clamp_min(0.0)
    for position in range(1, count):
        index = torch.argmax(minimum)
        selected[position] = index
        distance = (
            squared_norm
            + squared_norm[index]
            - 2.0 * torch.real(features @ features[index].conj())
        ).clamp_min(0.0)
        minimum = torch.minimum(minimum, distance)
    return selected


def prototype_indices(
    evaluator: GlobalQ3Evaluator,
    operator: GlobalConfig,
    dimension: int,
    count: int,
    method: str,
    seed: int,
) -> np.ndarray:
    basis = evaluator.basis[:, : operator.rank]
    latent = evaluator.train_encoded @ basis.conj()
    scale = torch.sqrt(evaluator.variance[:dimension]).clamp_min(1e-15)
    features = latent[:, :dimension] / scale
    if method == "fps":
        selected = farthest_point_indices(features, count)
    elif method == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = torch.randperm(len(features), generator=generator)[:count].to(features.device)
    else:
        raise ValueError(f"unknown prototype method: {method}")
    return selected.cpu().numpy()


def kmeans_operator_bank(
    evaluator: GlobalQ3Evaluator,
    operator: GlobalConfig,
    dimension: int,
    count: int,
    iterations: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress train operators into latent k-means centroids and their masses."""

    basis = evaluator.basis[:, : operator.rank]
    latent = evaluator.train_encoded @ basis.conj()
    scale = torch.sqrt(evaluator.variance[:dimension]).clamp_min(1e-15)
    complex_features = latent[:, :dimension] / scale
    features = torch.cat((complex_features.real, complex_features.imag), dim=-1)
    initial = farthest_point_indices(complex_features, count)
    centers = features[initial].clone()
    labels = torch.zeros(len(features), dtype=torch.long, device=features.device)
    for _ in range(iterations):
        distance = torch.cdist(features, centers).square()
        new_labels = distance.argmin(dim=1)
        if torch.equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        mass = torch.bincount(labels, minlength=count)
        empty = torch.where(mass == 0)[0]
        if len(empty):
            nearest = distance.gather(1, labels[:, None]).squeeze(1)
            replacement = torch.topk(nearest, len(empty)).indices
            labels[replacement] = empty
            mass = torch.bincount(labels, minlength=count)
        centers.zero_()
        centers.index_add_(0, labels, features)
        centers /= mass[:, None].clamp_min(1)
    mass = torch.bincount(labels, minlength=count).to(features.dtype)
    encoded = torch.zeros(
        (count, evaluator.train_encoded.shape[1]),
        dtype=evaluator.train_encoded.dtype,
        device=features.device,
    )
    raw = torch.zeros(
        (count, evaluator.train_raw.shape[1]),
        dtype=evaluator.train_raw.dtype,
        device=features.device,
    )
    encoded.index_add_(0, labels, evaluator.train_encoded)
    raw.index_add_(0, labels, evaluator.train_raw)
    encoded /= mass[:, None]
    raw /= mass[:, None]
    return encoded, raw, mass


def _config(payload: dict) -> tuple[GlobalConfig, MemoryConfig]:
    operator = payload["operator"]
    memory = payload["memory"]
    return (
        GlobalConfig(
            rank=int(operator["rank"]),
            strength=float(operator["strength"]),
            prior=str(operator.get("prior", "spectrum")),
            normalization=str(operator["normalization"]),
        ),
        MemoryConfig(
            latent_dimensions=int(memory["latent_dimensions"]),
            neighbors=int(memory["neighbors"]),
            temperature=float(memory["temperature"]),
            blend=float(memory["blend"]),
            bank_mode=str(memory.get("bank_mode", "posterior")),
        ),
    )


def run(config_path: Path, output: Path, device: str) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summary: dict = {}
    for domain, setting in payload["domains"].items():
        archive = load_cross_domain_archive(Path(setting["dataset"]))
        domain_result = {}
        for budget_text, budget_setting in setting["budgets"].items():
            budget = int(budget_text)
            operator, memory = _config(budget_setting)
            subspace = fit_global_subspace(archive, operator.rank, operator.normalization, device)
            validation = GlobalQ3Evaluator(
                archive,
                subspace,
                "validation",
                budget,
                int(payload["validation_repeats"]),
                int(payload["selection_seed"]),
                device,
            )
            threshold = float(validation.latent_energy(operator).max().cpu())
            full_count = len(validation.train_raw)
            candidates = []
            for fraction in payload["fractions"]:
                count = min(full_count, max(memory.neighbors, round(full_count * fraction)))
                for method in payload["methods"] if count < full_count else ["full"]:
                    encoded = raw = weights = None
                    if method == "full":
                        indices = np.arange(full_count)
                    elif method.startswith("kmeans"):
                        encoded, raw, mass = kmeans_operator_bank(
                            validation,
                            operator,
                            int(setting.get("selection_dimensions", memory.latent_dimensions)),
                            count,
                        )
                        indices = None
                        weights = mass if method == "kmeans_mass" else None
                    else:
                        indices = prototype_indices(
                            validation,
                            operator,
                            int(setting.get("selection_dimensions", memory.latent_dimensions)),
                            count,
                            method,
                            int(payload["selection_seed"]),
                        )
                    values = evaluate_memory_config_batched(
                        validation,
                        operator,
                        memory,
                        threshold if setting.get("use_latent_gate", False) else None,
                        indices,
                        encoded,
                        raw,
                        weights,
                    )
                    value = float(values.mean())
                    candidates.append((value, count, method, indices, encoded, raw, weights))
                    rows.append(
                        {
                            "domain": domain,
                            "budget": budget,
                            "split": "validation",
                            "method": method,
                            "prototype_count": count,
                            "storage_fraction": count / full_count,
                            "raw_complex_rel_l2": value,
                        }
                    )
            full_error = min(value for value, count, *_ in candidates if count == full_count)
            eligible = [
                item for item in candidates
                if item[1] <= full_count / 2 and item[0] <= full_error * (1.0 + payload["validation_tolerance"])
            ]
            if eligible:
                selected = min(eligible, key=lambda item: (item[1], item[0]))
            else:
                selected = min(
                    (item for item in candidates if item[1] == full_count),
                    key=lambda item: item[0],
                )
            _, count, method, indices, encoded, raw, weights = selected
            split_result = {}
            for split in ("test", "ood"):
                if not np.any(archive.split == split):
                    continue
                compressed_seeds = []
                full_seeds = []
                for seed in payload["test_seeds"]:
                    evaluator = GlobalQ3Evaluator(
                        archive,
                        subspace,
                        split,
                        budget,
                        int(payload["test_repeats"]),
                        int(seed),
                        device,
                    )
                    gate = threshold if setting.get("use_latent_gate", False) else None
                    compressed_seeds.append(
                        evaluate_memory_config_batched(
                            evaluator, operator, memory, gate, indices, encoded, raw, weights
                        )
                    )
                    full_seeds.append(
                        evaluate_memory_config_batched(evaluator, operator, memory, gate)
                    )
                compressed = np.stack(compressed_seeds)
                full = np.stack(full_seeds)
                stats = paired_summary(
                    full.mean(axis=0), compressed.mean(axis=0), int(payload["test_seeds"][0]) + budget
                )
                stats.update(
                    {
                        "compressed_seed_means": compressed.mean(axis=1).tolist(),
                        "full_seed_means": full.mean(axis=1).tolist(),
                    }
                )
                split_result[split] = stats
                np.savez_compressed(
                    output / f"{domain.lower()}_k{budget}_{split}_systems.npz",
                    compressed=compressed,
                    full=full,
                )
            domain_result[budget_text] = {
                "operator": operator.name,
                "memory": memory.name,
                "prototype_method": method,
                "prototype_count": count,
                "full_count": full_count,
                "storage_fraction": count / full_count,
                "validation_full_error": full_error,
                "validation_compressed_error": float(selected[0]),
                **split_result,
            }
        summary[domain] = domain_result
    with (output / "validation_pareto.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
