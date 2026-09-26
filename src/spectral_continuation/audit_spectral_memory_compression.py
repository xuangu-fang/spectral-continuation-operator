"""Compress a posterior-memory value bank into shared response-basis coefficients."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import GlobalConfig, GlobalQ3Evaluator, fit_global_subspace
from .posterior_memory_response_operator import MemoryConfig, evaluate_memory_config_batched
from .probabilistic_response_operator import paired_summary


def run(config_path: Path, output: Path, device: str) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    result = {}
    for domain, setting in payload["domains"].items():
        archive = load_cross_domain_archive(Path(setting["dataset"]))
        domain_result = {}
        for budget_text, spec in setting["budgets"].items():
            budget = int(budget_text)
            operator = GlobalConfig(
                int(spec["operator"]["rank"]),
                float(spec["operator"]["strength"]),
                "spectrum",
                str(spec["operator"]["normalization"]),
            )
            memory = MemoryConfig(**spec["memory"])
            maximum_rank = max(operator.rank, max(payload["coefficient_ranks"]))
            subspace = fit_global_subspace(archive, maximum_rank, operator.normalization, device)
            validation = GlobalQ3Evaluator(
                archive, subspace, "validation", budget,
                int(payload["validation_repeats"]), int(payload["selection_seed"]), device,
            )
            threshold = float(validation.latent_energy(operator).max().cpu())
            gate = threshold if setting.get("use_latent_gate", False) else None
            full_error = float(evaluate_memory_config_batched(validation, operator, memory, gate).mean())
            basis = validation.basis
            full_coefficients = validation.train_encoded @ basis.conj()
            candidates = []
            raw_dimension = validation.train_raw.shape[-1]
            for rank in payload["coefficient_ranks"]:
                coefficients = full_coefficients[:, :rank]
                value = float(
                    evaluate_memory_config_batched(
                        validation, operator, memory, gate,
                        memory_coefficients=coefficients,
                    ).mean()
                )
                storage_fraction = rank / raw_dimension
                candidates.append((value, rank, storage_fraction, coefficients))
                rows.append(
                    {
                        "domain": domain,
                        "budget": budget,
                        "coefficient_rank": rank,
                        "storage_fraction": storage_fraction,
                        "validation_error": value,
                        "validation_full_bank_error": full_error,
                        "relative_change": value / full_error - 1.0,
                    }
                )
            eligible = [
                candidate for candidate in candidates
                if candidate[0] <= full_error * (1.0 + payload["validation_tolerance"])
            ]
            selected = min(eligible, key=lambda candidate: candidate[2]) if eligible else None
            if selected is None:
                rank = raw_dimension
                storage_fraction = 1.0
                validation_error = full_error
                coefficients = None
            else:
                validation_error, rank, storage_fraction, coefficients = selected
            split_result = {}
            for split in ("test", "ood"):
                if not np.any(archive.split == split):
                    continue
                compressed_by_seed = []
                full_by_seed = []
                for seed in payload["test_seeds"]:
                    evaluator = GlobalQ3Evaluator(
                        archive, subspace, split, budget,
                        int(payload["test_repeats"]), int(seed), device,
                    )
                    compressed_by_seed.append(
                        evaluate_memory_config_batched(
                            evaluator, operator, memory, gate,
                            memory_coefficients=coefficients,
                        ) if coefficients is not None else
                        evaluate_memory_config_batched(evaluator, operator, memory, gate)
                    )
                    full_by_seed.append(evaluate_memory_config_batched(evaluator, operator, memory, gate))
                compressed = np.stack(compressed_by_seed)
                full = np.stack(full_by_seed)
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
                    compressed=compressed, full=full,
                )
            domain_result[budget_text] = {
                "operator": operator.name,
                "memory": memory.name,
                "coefficient_rank": rank,
                "raw_dimension": raw_dimension,
                "storage_fraction": storage_fraction,
                "validation_full_error": full_error,
                "validation_compressed_error": validation_error,
                **split_result,
            }
        result[domain] = domain_result
    with (output / "validation_pareto.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
