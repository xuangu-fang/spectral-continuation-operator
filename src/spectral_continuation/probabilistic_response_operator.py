"""Probabilistic latent response operator for sparse wave prompts.

The baseline uses an isotropic ridge penalty when identifying a new system in
the learned complex response subspace.  PCA directions have very different
training-system variances, so isotropic shrinkage is statistically mismatched.
This module learns the latent Gaussian spectrum on train systems and performs
closed-form MAP system identification under sparse source-response prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .cross_domain_query_audit import relative_l2, source_orders
from .cross_domain_response_data import CrossDomainResponseArchive, load_cross_domain_archive


@dataclass(frozen=True)
class ProbabilisticSubspace:
    mean: np.ndarray
    basis: np.ndarray
    variance: np.ndarray


@dataclass(frozen=True)
class LatentConfig:
    rank: int
    strength: float
    prior: str

    @property
    def name(self) -> str:
        prefix = "scalar_ridge" if self.prior == "isotropic" else "spectrum_prior"
        return f"{prefix}_r{self.rank}_l{self.strength:g}"


def fit_probabilistic_subspaces(
    archive: CrossDomainResponseArchive, maximum_rank: int
) -> list[ProbabilisticSubspace]:
    train = archive.responses[archive.split == "train"]
    result = []
    for frequency in range(train.shape[2]):
        table = train[:, :, frequency].reshape(len(train), -1)
        mean = table.mean(axis=0)
        _, singular, right = np.linalg.svd(table - mean, full_matrices=False)
        rank = min(maximum_rank, len(singular))
        variance = np.square(singular[:rank]) / max(len(train) - 1, 1)
        result.append(
            ProbabilisticSubspace(
                mean=mean.astype(np.complex64),
                basis=right[:rank].T.astype(np.complex64),
                variance=variance.astype(np.float64),
            )
        )
    return result


def latent_penalty(subspace: ProbabilisticSubspace, config: LatentConfig) -> np.ndarray:
    if config.prior == "isotropic":
        return np.full(config.rank, config.strength, dtype=np.float64)
    if config.prior != "spectrum":
        raise ValueError(f"unknown prior {config.prior!r}")
    variance = subspace.variance[: config.rank]
    reference = float(np.median(variance))
    return config.strength * reference / np.maximum(variance, 1e-30)


def predict_query(
    subspace: ProbabilisticSubspace,
    response: np.ndarray,
    context: np.ndarray,
    target: int,
    receiver_count: int,
    config: LatentConfig,
) -> np.ndarray:
    observed = np.concatenate(
        [np.arange(source * receiver_count, (source + 1) * receiver_count) for source in context]
    )
    target_slice = slice(target * receiver_count, (target + 1) * receiver_count)
    basis = subspace.basis[:, : config.rank]
    design = basis[observed]
    penalty = latent_penalty(subspace, config)
    gram = design.conj().T @ design + np.diag(penalty).astype(np.complex64)
    projector = np.linalg.solve(gram, design.conj().T)
    values = response[:, context].reshape(len(response), -1)
    coefficient = (values - subspace.mean[observed]) @ projector.T
    return subspace.mean[target_slice] + coefficient @ basis[target_slice].T


def evaluate_config(
    archive: CrossDomainResponseArchive,
    subspaces: list[ProbabilisticSubspace],
    split: str,
    budget: int,
    config: LatentConfig,
    repeats: int,
    seed: int,
) -> np.ndarray:
    systems = np.flatnonzero(archive.split == split)
    response = archive.responses[systems]
    source_count, frequency_count, receiver_count = response.shape[1:]
    orders = source_orders(source_count, repeats, seed)
    total = np.zeros(len(systems), dtype=np.float64)
    count = 0
    for target in range(source_count):
        for order in orders[target]:
            context = order[:budget]
            for frequency in range(frequency_count):
                truth = response[:, target, frequency]
                prediction = predict_query(
                    subspaces[frequency],
                    response[:, :, frequency],
                    context,
                    target,
                    receiver_count,
                    config,
                )
                total += relative_l2(prediction, truth)
                count += 1
    return total / count


class BatchedQueryEvaluator:
    """Cache one frozen prompt protocol and evaluate all queries in batches."""

    def __init__(
        self,
        archive: CrossDomainResponseArchive,
        subspaces: list[ProbabilisticSubspace],
        split: str,
        budget: int,
        repeats: int,
        seed: int,
        device: str,
    ) -> None:
        systems = np.flatnonzero(archive.split == split)
        if not len(systems):
            raise ValueError(f"archive has no {split!r} systems")
        response = archive.responses[systems]
        source_count, frequency_count, receiver_count = response.shape[1:]
        orders = source_orders(source_count, repeats, seed)
        context_values = []
        truth_values = []
        observed_indices = []
        target_indices = []
        frequency_indices = []
        receiver = np.arange(receiver_count)
        for target in range(source_count):
            for order in orders[target]:
                context = order[:budget]
                observed = np.concatenate(
                    [source * receiver_count + receiver for source in context]
                )
                selected = response[:, context]
                for frequency in range(frequency_count):
                    context_values.append(selected[:, :, frequency].reshape(len(response), -1))
                    truth_values.append(response[:, target, frequency])
                    observed_indices.append(observed)
                    target_indices.append(target * receiver_count + receiver)
                    frequency_indices.append(frequency)

        torch_device = torch.device(device)
        self.context = torch.as_tensor(
            np.stack(context_values), dtype=torch.complex64, device=torch_device
        )
        self.truth = torch.as_tensor(
            np.stack(truth_values), dtype=torch.complex64, device=torch_device
        )
        self.observed = torch.as_tensor(
            np.stack(observed_indices), dtype=torch.long, device=torch_device
        )
        self.target = torch.as_tensor(
            np.stack(target_indices), dtype=torch.long, device=torch_device
        )
        self.frequency = torch.as_tensor(
            frequency_indices, dtype=torch.long, device=torch_device
        )
        self.mean = torch.as_tensor(
            np.stack([value.mean for value in subspaces]),
            dtype=torch.complex64,
            device=torch_device,
        )
        self.basis = torch.as_tensor(
            np.stack([value.basis for value in subspaces]),
            dtype=torch.complex64,
            device=torch_device,
        )
        self.variance = torch.as_tensor(
            np.stack([value.variance for value in subspaces]),
            dtype=torch.float32,
            device=torch_device,
        )
        self.receiver_count = receiver_count
        self.budget = budget

    @torch.inference_mode()
    def predict(
        self,
        config: LatentConfig,
        observed: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observed = self.observed if observed is None else observed
        context = self.context if context is None else context
        target = self.target if target is None else target
        rank = config.rank
        basis = self.basis[self.frequency, :, :rank]
        mean = self.mean[self.frequency]
        variance = self.variance[self.frequency, :rank]
        observed_basis = torch.gather(
            basis,
            1,
            observed[:, :, None].expand(-1, -1, rank),
        )
        observed_mean = torch.gather(mean, 1, observed)
        if config.prior == "isotropic":
            penalty = torch.full_like(variance, config.strength)
        elif config.prior == "spectrum":
            reference = torch.quantile(variance, 0.5, dim=1, keepdim=True)
            penalty = config.strength * reference / variance.clamp_min(1e-30)
        else:
            raise ValueError(f"unknown prior {config.prior!r}")
        adjoint = observed_basis.conj().transpose(1, 2)
        gram = adjoint @ observed_basis + torch.diag_embed(penalty).to(torch.complex64)
        projector = torch.linalg.solve(gram, adjoint)
        coefficient = torch.einsum(
            "eno,elo->enl", context - observed_mean[:, None], projector
        )
        target_basis = torch.gather(
            basis,
            1,
            target[:, :, None].expand(-1, -1, rank),
        )
        target_mean = torch.gather(mean, 1, target)
        return target_mean[:, None] + torch.einsum(
            "enl,erl->enr", coefficient, target_basis
        )

    @torch.inference_mode()
    def evaluate(self, config: LatentConfig) -> np.ndarray:
        prediction = self.predict(config)
        numerator = torch.linalg.vector_norm(prediction - self.truth, dim=-1)
        denominator = torch.linalg.vector_norm(self.truth, dim=-1).clamp_min(1e-12)
        return (numerator / denominator).mean(dim=0).cpu().numpy().astype(np.float64)

    @torch.inference_mode()
    def cross_validation_error(self, config: LatentConfig) -> torch.Tensor:
        """Leave one prompt source out without reading the target query."""

        if self.budget < 2:
            raise ValueError("leave-one-prompt-out calibration requires budget >= 2")
        event_count, system_count, _ = self.context.shape
        observed = self.observed.reshape(event_count, self.budget, self.receiver_count)
        context = self.context.reshape(
            event_count, system_count, self.budget, self.receiver_count
        )
        total = torch.zeros(
            (event_count, system_count), dtype=torch.float32, device=self.context.device
        )
        for held in range(self.budget):
            keep = [index for index in range(self.budget) if index != held]
            prediction = self.predict(
                config,
                observed[:, keep].reshape(event_count, -1),
                context[:, :, keep].reshape(event_count, system_count, -1),
                observed[:, held],
            )
            truth = context[:, :, held]
            total += torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
                truth, dim=-1
            ).clamp_min(1e-12)
        return total / self.budget

    @torch.inference_mode()
    def evaluate_adaptive(
        self, baseline: LatentConfig, candidate: LatentConfig
    ) -> tuple[np.ndarray, float]:
        """Choose a prior per episode/system using prompt-only cross-validation."""

        baseline_prediction = self.predict(baseline)
        candidate_prediction = self.predict(candidate)
        baseline_cv = self.cross_validation_error(baseline)
        candidate_cv = self.cross_validation_error(candidate)
        use_candidate = candidate_cv <= baseline_cv
        prediction = torch.where(
            use_candidate[:, :, None], candidate_prediction, baseline_prediction
        )
        error = torch.linalg.vector_norm(prediction - self.truth, dim=-1) / torch.linalg.vector_norm(
            self.truth, dim=-1
        ).clamp_min(1e-12)
        return (
            error.mean(dim=0).cpu().numpy().astype(np.float64),
            float(use_candidate.float().mean().cpu()),
        )


def select_config(
    archive: CrossDomainResponseArchive,
    subspaces: list[ProbabilisticSubspace],
    budget: int,
    candidates: tuple[LatentConfig, ...],
    repeats: int,
    seed: int,
    device: str = "cpu",
) -> tuple[LatentConfig, list[dict[str, float | int | str]]]:
    rows = []
    best_config = candidates[0]
    best_error = float("inf")
    evaluator = BatchedQueryEvaluator(
        archive, subspaces, "validation", budget, repeats, seed, device
    )
    for config in candidates:
        if config.rank > subspaces[0].basis.shape[1]:
            continue
        errors = evaluator.evaluate(config)
        error = float(errors.mean())
        rows.append(
            {
                "budget": budget,
                "method": config.name,
                "prior": config.prior,
                "rank": config.rank,
                "strength": config.strength,
                "validation_raw_complex_rel_l2": error,
            }
        )
        if error < best_error:
            best_error = error
            best_config = config
    return best_config, rows


def paired_summary(
    baseline: np.ndarray, candidate: np.ndarray, seed: int, draws: int = 20_000
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        count = min(1000, draws - start)
        index = generator.integers(0, len(baseline), size=(count, len(baseline)))
        baseline_mean = baseline[index].mean(axis=1)
        candidate_mean = candidate[index].mean(axis=1)
        samples[start : start + count] = 1.0 - candidate_mean / baseline_mean
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "ratio_of_means_gain": float(1.0 - candidate.mean() / baseline.mean()),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "system_win_rate": float(np.mean(candidate < baseline)),
    }


def _write_system_rows(
    output: Path,
    domain: str,
    split: str,
    budget: int,
    system_ids: np.ndarray,
    scalar: np.ndarray,
    spectrum: np.ndarray,
) -> None:
    rows = []
    for system_id, scalar_error, spectrum_error in zip(
        system_ids, scalar, spectrum, strict=True
    ):
        rows.append(
            {
                "domain": domain,
                "split": split,
                "system_id": str(system_id),
                "budget": budget,
                "scalar_ridge_rel_l2": float(scalar_error),
                "spectrum_prior_rel_l2": float(spectrum_error),
            }
        )
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_domain(
    domain: str,
    dataset: Path,
    output: Path,
    budgets: tuple[int, ...],
    ranks: tuple[int, ...],
    scalar_strengths: tuple[float, ...],
    spectrum_strengths: tuple[float, ...],
    validation_repeats: int,
    test_repeats: int,
    seed: int,
    device: str,
) -> dict:
    archive = load_cross_domain_archive(dataset)
    subspaces = fit_probabilistic_subspaces(archive, max(ranks))
    scalar_candidates = tuple(
        LatentConfig(rank, strength, "isotropic")
        for rank in ranks
        for strength in scalar_strengths
    )
    spectrum_candidates = tuple(
        LatentConfig(rank, strength, "spectrum")
        for rank in ranks
        for strength in spectrum_strengths
    )
    output.mkdir(parents=True, exist_ok=True)
    system_csv = output / "system_metrics.csv"
    if system_csv.exists():
        system_csv.unlink()
    validation_rows = []
    results = {}
    selected = {}
    for budget in budgets:
        scalar_config, scalar_rows = select_config(
            archive,
            subspaces,
            budget,
            scalar_candidates,
            validation_repeats,
            seed,
            device,
        )
        spectrum_config, spectrum_rows = select_config(
            archive,
            subspaces,
            budget,
            spectrum_candidates,
            validation_repeats,
            seed,
            device,
        )
        validation_rows.extend(scalar_rows)
        validation_rows.extend(spectrum_rows)
        selected[str(budget)] = {
            "scalar": scalar_config.name,
            "spectrum": spectrum_config.name,
        }
        test_evaluator = BatchedQueryEvaluator(
            archive, subspaces, "test", budget, test_repeats, seed, device
        )
        scalar_test = test_evaluator.evaluate(scalar_config)
        spectrum_test = test_evaluator.evaluate(spectrum_config)
        results[str(budget)] = paired_summary(
            scalar_test, spectrum_test, seed + budget
        )
        _write_system_rows(
            system_csv,
            domain,
            "test",
            budget,
            archive.system_ids[archive.split == "test"],
            scalar_test,
            spectrum_test,
        )
        if np.any(archive.split == "ood"):
            ood_evaluator = BatchedQueryEvaluator(
                archive, subspaces, "ood", budget, test_repeats, seed, device
            )
            scalar_ood = ood_evaluator.evaluate(scalar_config)
            spectrum_ood = ood_evaluator.evaluate(spectrum_config)
            results[str(budget)]["ood"] = paired_summary(
                scalar_ood, spectrum_ood, seed + 1000 + budget
            )
            _write_system_rows(
                system_csv,
                domain,
                "ood",
                budget,
                archive.system_ids[archive.split == "ood"],
                scalar_ood,
                spectrum_ood,
            )
    with (output / "validation_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=validation_rows[0].keys())
        writer.writeheader()
        writer.writerows(validation_rows)
    full_budget = max(budgets)
    full_scalar = results[str(full_budget)]["baseline_mean"]
    equivalent_budget = next(
        (
            budget
            for budget in budgets
            if results[str(budget)]["candidate_mean"] <= full_scalar
        ),
        None,
    )
    summary = {
        "domain": domain,
        "dataset": str(dataset.resolve()),
        "protocol": {
            "selection": "validation only",
            "validation_repeats": validation_repeats,
            "test_repeats": test_repeats,
            "seed": seed,
            "primary_metric": "per-query relative L2, macro-averaged within physical system",
        },
        "selected": selected,
        "test": results,
        "measurement_efficiency": {
            "full_budget": full_budget,
            "full_scalar_error": full_scalar,
            "equivalent_spectrum_budget": equivalent_budget,
            "probe_saving": None
            if equivalent_budget is None
            else 1.0 - equivalent_budget / full_budget,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        budgets,
        [results[str(value)]["baseline_mean"] for value in budgets],
        "o-",
        label="validation-selected scalar ridge",
    )
    axis.plot(
        budgets,
        [results[str(value)]["candidate_mean"] for value in budgets],
        "s-",
        label="learned latent spectrum prior",
    )
    axis.set_xlabel("Context source budget K")
    axis.set_ylabel("Raw complex relative L2")
    axis.set_title(domain)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "budget_curve.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    domain = payload["domains"][args.domain]
    summary = run_domain(
        args.domain,
        Path(domain["dataset"]),
        args.output,
        tuple(int(value) for value in domain["budgets"]),
        tuple(int(value) for value in domain["ranks"]),
        tuple(float(value) for value in payload["scalar_strengths"]),
        tuple(float(value) for value in payload["spectrum_strengths"]),
        int(payload["validation_repeats"]),
        int(payload["test_repeats"]),
        int(payload["seed"] if args.seed is None else args.seed),
        args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
