"""Global Bayesian response operator for strict joint source--frequency queries.

Unlike the per-frequency response subspace, this module learns one latent state
for the complete source--frequency--receiver response table.  It is the linear
special case and initialization target of the Prompted Bayesian Neural Operator.
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

from .cross_domain_response_data import CrossDomainResponseArchive, load_cross_domain_archive
from .probabilistic_response_operator import paired_summary


@dataclass(frozen=True)
class GlobalSubspace:
    center: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    variance: np.ndarray
    reconstruction_error: np.ndarray
    normalization: str


@dataclass(frozen=True)
class GlobalConfig:
    rank: int
    strength: float
    prior: str
    normalization: str

    @property
    def name(self) -> str:
        return (
            f"global_{self.normalization}_{self.prior}_"
            f"r{self.rank}_l{self.strength:g}"
        )


def fit_global_subspace(
    archive: CrossDomainResponseArchive,
    maximum_rank: int,
    normalization: str,
    device: str = "cpu",
) -> GlobalSubspace:
    """Fit a complex PCA model to complete train-system response operators."""

    train = archive.responses[archive.split == "train"].reshape(
        int(np.sum(archive.split == "train")), -1
    )
    center = train.mean(axis=0).astype(np.complex64)
    if normalization == "raw":
        scale = np.ones(train.shape[1], dtype=np.float32)
    elif normalization == "coordinate_rms":
        scale = np.sqrt(np.mean(np.abs(train - center[None]) ** 2, axis=0)).astype(
            np.float32
        )
        positive = scale[scale > 0]
        floor = 0.02 * float(np.median(positive)) if len(positive) else 1.0
        scale = np.maximum(scale, max(floor, 1e-12))
    else:
        raise ValueError(f"unknown normalization {normalization!r}")
    encoded = ((train - center[None]) / scale[None]).astype(np.complex64)
    matrix = torch.as_tensor(encoded, device=torch.device(device))
    _, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    rank = min(maximum_rank, len(singular))
    # Store the transpose of Vh so that each response column satisfies x = B z.
    # Using the Hermitian transpose here would conjugate the response basis twice.
    basis = right[:rank].T.cpu().numpy().astype(np.complex64)
    singular_np = singular.cpu().numpy().astype(np.float64)
    variance = np.square(singular_np[:rank]) / max(len(train) - 1, 1)
    total_energy = float(np.square(singular_np).sum())
    cumulative = np.cumsum(np.square(singular_np))
    reconstruction_error = np.sqrt(
        np.maximum(total_energy - cumulative[:rank], 0.0) / max(total_energy, 1e-30)
    )
    return GlobalSubspace(
        center=center,
        scale=scale,
        basis=basis,
        variance=variance,
        reconstruction_error=reconstruction_error,
        normalization=normalization,
    )


def _q3_events(
    source_count: int,
    frequency_count: int,
    budget: int,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen nested strict-Q3 contexts and all target pairs."""

    contexts = []
    targets = []
    for source in range(source_count):
        for frequency in range(frequency_count):
            candidates = np.asarray(
                [
                    (other_source, other_frequency)
                    for other_source in range(source_count)
                    for other_frequency in range(frequency_count)
                    if other_source != source and other_frequency != frequency
                ],
                dtype=np.int64,
            )
            if budget > len(candidates):
                raise ValueError(
                    f"budget {budget} exceeds strict-Q3 capacity {len(candidates)}"
                )
            for repeat in range(repeats):
                # Each target/repeat has an independent but reproducible order.
                target_seed = seed + 1_000_003 * source + 10_007 * frequency + repeat
                order = np.random.default_rng(target_seed).permutation(len(candidates))
                contexts.append(candidates[order[:budget]])
                targets.append((source, frequency))
    return np.stack(contexts), np.asarray(targets, dtype=np.int64)


class GlobalQ3Evaluator:
    """Batched evaluator with one frozen strict-Q3 prompt protocol."""

    def __init__(
        self,
        archive: CrossDomainResponseArchive,
        subspace: GlobalSubspace,
        split: str,
        budget: int,
        repeats: int,
        seed: int,
        device: str,
    ) -> None:
        indices = np.flatnonzero(archive.split == split)
        if not len(indices):
            raise ValueError(f"archive has no split {split!r}")
        raw = archive.responses[indices]
        systems, source_count, frequency_count, receiver_count = raw.shape
        context_pairs, target_pairs = _q3_events(
            source_count, frequency_count, budget, repeats, seed
        )
        receiver = np.arange(receiver_count, dtype=np.int64)
        context_query = context_pairs[..., 0] * frequency_count + context_pairs[..., 1]
        target_query = target_pairs[:, 0] * frequency_count + target_pairs[:, 1]
        observed = (
            context_query[..., None] * receiver_count + receiver[None, None]
        ).reshape(len(context_pairs), -1)
        target = target_query[:, None] * receiver_count + receiver[None]

        torch_device = torch.device(device)
        raw_flat = torch.as_tensor(
            raw.reshape(systems, -1), dtype=torch.complex64, device=torch_device
        )
        center = torch.as_tensor(subspace.center, device=torch_device)
        scale = torch.as_tensor(subspace.scale, device=torch_device)
        observed_t = torch.as_tensor(observed, dtype=torch.long, device=torch_device)
        target_t = torch.as_tensor(target, dtype=torch.long, device=torch_device)
        encoded = (raw_flat - center[None]) / scale[None]
        self.context = encoded[:, observed_t].permute(1, 0, 2).contiguous()
        self.truth = raw_flat[:, target_t].permute(1, 0, 2).contiguous()
        self.raw = raw_flat
        self.observed = observed_t
        self.target = target_t
        self.context_pairs = context_pairs
        self.target_pairs = target_pairs
        self.center = center
        self.scale = scale
        self.basis = torch.as_tensor(subspace.basis, device=torch_device)
        self.variance = torch.as_tensor(
            subspace.variance, dtype=torch.float32, device=torch_device
        )
        self.receiver_count = receiver_count
        self.frequency_count = frequency_count
        train = archive.responses[archive.split == "train"].reshape(
            int(np.sum(archive.split == "train")), -1
        )
        self.train_raw = torch.as_tensor(
            train, dtype=torch.complex64, device=torch_device
        )
        self.train_encoded = (self.train_raw - center[None]) / scale[None]

    @torch.inference_mode()
    def predict(
        self,
        config: GlobalConfig,
        observed: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observed = self.observed if observed is None else observed
        context = self.context if context is None else context
        target = self.target if target is None else target
        rank = config.rank
        basis = self.basis[:, :rank]
        observed_basis = basis[observed]
        if config.prior == "isotropic":
            penalty = torch.full(
                (rank,), config.strength, dtype=torch.float32, device=basis.device
            )
        elif config.prior == "spectrum":
            variance = self.variance[:rank]
            reference = torch.quantile(variance, 0.5)
            penalty = config.strength * reference / variance.clamp_min(1e-30)
        else:
            raise ValueError(f"unknown prior {config.prior!r}")
        adjoint = observed_basis.mH
        gram = adjoint @ observed_basis + torch.diag_embed(
            penalty[None].expand(len(observed_basis), -1)
        ).to(torch.complex64)
        projector = torch.linalg.solve(gram, adjoint)
        coefficient = torch.einsum("eno,elo->enl", context, projector)
        target_basis = basis[target]
        normalized = torch.einsum("enl,erl->enr", coefficient, target_basis)
        target_center = self.center[target]
        target_scale = self.scale[target]
        return target_center[:, None] + target_scale[:, None] * normalized

    @torch.inference_mode()
    def latent_energy(self, config: GlobalConfig) -> torch.Tensor:
        """Prompt-only standardized MAP energy for distribution-shift detection."""

        rank = config.rank
        basis = self.basis[:, :rank]
        observed_basis = basis[self.observed]
        if config.prior == "isotropic":
            penalty = torch.full(
                (rank,), config.strength, dtype=torch.float32, device=basis.device
            )
        else:
            variance = self.variance[:rank]
            reference = torch.quantile(variance, 0.5)
            penalty = config.strength * reference / variance.clamp_min(1e-30)
        adjoint = observed_basis.mH
        gram = adjoint @ observed_basis + torch.diag_embed(
            penalty[None].expand(len(observed_basis), -1)
        ).to(torch.complex64)
        coefficient = torch.einsum(
            "eno,elo->enl", self.context, torch.linalg.solve(gram, adjoint)
        )
        return (
            coefficient.abs().square() / self.variance[:rank].clamp_min(1e-30)[None, None]
        ).mean(dim=-1)

    @torch.inference_mode()
    def evaluate_latent_gate(
        self, config: GlobalConfig, threshold: float
    ) -> tuple[np.ndarray, float]:
        """Fall back to the train mean when prompt latent energy is OOD."""

        use_operator = self.latent_energy(config) <= threshold
        operator = self.predict(config)
        mean = self.center[self.target][:, None]
        prediction = torch.where(use_operator[:, :, None], operator, mean)
        error = torch.linalg.vector_norm(prediction - self.truth, dim=-1) / torch.linalg.vector_norm(
            self.truth, dim=-1
        ).clamp_min(1e-12)
        return (
            error.mean(dim=0).cpu().numpy().astype(np.float64),
            float(use_operator.float().mean().cpu()),
        )

    @torch.inference_mode()
    def evaluate(self, config: GlobalConfig) -> np.ndarray:
        prediction = self.predict(config)
        error = torch.linalg.vector_norm(prediction - self.truth, dim=-1) / torch.linalg.vector_norm(
            self.truth, dim=-1
        ).clamp_min(1e-12)
        return error.mean(dim=0).cpu().numpy().astype(np.float64)

    @torch.inference_mode()
    def evaluate_adaptive_mean(
        self, config: GlobalConfig
    ) -> tuple[np.ndarray, float]:
        """Use prompt-only leave-one-out error to gate an OOD-safe mean fallback."""

        budget = self.observed.shape[1] // self.receiver_count
        if budget < 2:
            raise ValueError("prompt-only calibration requires at least two prompts")
        event_count, system_count, _ = self.context.shape
        observed = self.observed.reshape(event_count, budget, self.receiver_count)
        context = self.context.reshape(
            event_count, system_count, budget, self.receiver_count
        )
        operator_cv = torch.zeros(
            (event_count, system_count), dtype=torch.float32, device=self.context.device
        )
        mean_cv = torch.zeros_like(operator_cv)
        for held in range(budget):
            keep = [index for index in range(budget) if index != held]
            held_index = observed[:, held]
            truth = (
                self.center[held_index][:, None]
                + self.scale[held_index][:, None] * context[:, :, held]
            )
            operator = self.predict(
                config,
                observed[:, keep].reshape(event_count, -1),
                context[:, :, keep].reshape(event_count, system_count, -1),
                held_index,
            )
            mean = self.center[held_index][:, None]
            denominator = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)
            operator_cv += torch.linalg.vector_norm(operator - truth, dim=-1) / denominator
            mean_cv += torch.linalg.vector_norm(mean - truth, dim=-1) / denominator
        use_operator = operator_cv <= mean_cv
        operator_target = self.predict(config)
        mean_target = self.center[self.target][:, None]
        prediction = torch.where(
            use_operator[:, :, None], operator_target, mean_target
        )
        error = torch.linalg.vector_norm(prediction - self.truth, dim=-1) / torch.linalg.vector_norm(
            self.truth, dim=-1
        ).clamp_min(1e-12)
        return (
            error.mean(dim=0).cpu().numpy().astype(np.float64),
            float(use_operator.float().mean().cpu()),
        )

    @torch.inference_mode()
    def evaluate_mean(self) -> np.ndarray:
        prediction = self.center[self.target][:, None]
        error = torch.linalg.vector_norm(prediction - self.truth, dim=-1) / torch.linalg.vector_norm(
            self.truth, dim=-1
        ).clamp_min(1e-12)
        return error.mean(dim=0).cpu().numpy().astype(np.float64)

    @torch.inference_mode()
    def evaluate_nearest_query(
        self, sources: np.ndarray, frequencies: np.ndarray
    ) -> np.ndarray:
        source = np.asarray(sources, dtype=np.float64)
        source_range = np.maximum(source.max(axis=0) - source.min(axis=0), 1e-12)
        source = (source - source.min(axis=0)) / source_range
        log_frequency = np.log(np.asarray(frequencies, dtype=np.float64))
        log_frequency = (log_frequency - log_frequency.min()) / max(
            float(np.ptp(log_frequency)), 1e-12
        )
        prediction = []
        for context, target in zip(self.context_pairs, self.target_pairs, strict=True):
            target_coordinate = np.concatenate((source[target[0]], [log_frequency[target[1]]]))
            coordinates = np.concatenate(
                (source[context[:, 0]], log_frequency[context[:, 1], None]), axis=1
            )
            selected = int(np.argmin(np.sum((coordinates - target_coordinate) ** 2, axis=1)))
            query = int(
                context[selected, 0] * self.frequency_count + context[selected, 1]
            )
            start = query * self.receiver_count
            prediction.append(self.raw[:, start : start + self.receiver_count])
        values = torch.stack(prediction)
        error = torch.linalg.vector_norm(values - self.truth, dim=-1) / torch.linalg.vector_norm(
            self.truth, dim=-1
        ).clamp_min(1e-12)
        return error.mean(dim=0).cpu().numpy().astype(np.float64)

    @torch.inference_mode()
    def evaluate_system_retrieval(self) -> np.ndarray:
        """Retrieve a train system using prompts only, then answer the target."""

        errors = []
        for event in range(len(self.observed)):
            observed = self.observed[event]
            query = self.context[event]
            bank = self.train_encoded[:, observed]
            query_energy = query.abs().square().sum(dim=1, keepdim=True)
            bank_energy = bank.abs().square().sum(dim=1)[None]
            distance = query_energy + bank_energy - 2.0 * torch.real(
                query @ bank.conj().T
            )
            selected = torch.argmin(distance, dim=1)
            prediction = self.train_raw[selected][:, self.target[event]]
            truth = self.truth[event]
            errors.append(
                torch.linalg.vector_norm(prediction - truth, dim=-1)
                / torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)
            )
        return torch.stack(errors).mean(dim=0).cpu().numpy().astype(np.float64)


def _candidate_configs(payload: dict, normalization: str) -> tuple[GlobalConfig, ...]:
    result = []
    for prior, key in (("isotropic", "scalar_strengths"), ("spectrum", "spectrum_strengths")):
        for rank in payload["ranks"]:
            for strength in payload[key]:
                result.append(
                    GlobalConfig(int(rank), float(strength), prior, normalization)
                )
    return tuple(result)


def _write_system_metrics(
    path: Path,
    domain: str,
    split: str,
    budget: int,
    system_ids: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> None:
    fieldnames = ["domain", "split", "budget", "system_id", *metrics]
    append = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for index, system_id in enumerate(system_ids):
            row = {
                "domain": domain,
                "split": split,
                "budget": budget,
                "system_id": str(system_id),
            }
            row.update({name: float(value[index]) for name, value in metrics.items()})
            writer.writerow(row)


def run_domain(
    domain: str,
    dataset: Path,
    output: Path,
    payload: dict,
    device: str,
) -> dict:
    archive = load_cross_domain_archive(dataset)
    output.mkdir(parents=True, exist_ok=True)
    metric_path = output / "system_metrics.csv"
    if metric_path.exists():
        metric_path.unlink()
    maximum_rank = min(max(payload["ranks"]), int(np.sum(archive.split == "train")))
    subspaces = {
        normalization: fit_global_subspace(
            archive, maximum_rank, normalization, device
        )
        for normalization in payload["normalizations"]
    }
    budgets = tuple(
        int(value)
        for value in payload["budgets"]
        if value
        <= (archive.responses.shape[1] - 1) * (archive.responses.shape[2] - 1)
    )
    validation_rows: list[dict] = []
    selected: dict[str, str] = {}
    results: dict[str, dict] = {}
    for budget in budgets:
        best_config = None
        best_error = float("inf")
        for normalization, subspace in subspaces.items():
            evaluator = GlobalQ3Evaluator(
                archive,
                subspace,
                "validation",
                budget,
                int(payload["validation_repeats"]),
                int(payload["seed"]),
                device,
            )
            for config in _candidate_configs(payload, normalization):
                if config.rank > subspace.basis.shape[1]:
                    continue
                errors = evaluator.evaluate(config)
                error = float(errors.mean())
                validation_rows.append(
                    {
                        "budget": budget,
                        "method": config.name,
                        "normalization": normalization,
                        "prior": config.prior,
                        "rank": config.rank,
                        "strength": config.strength,
                        "validation_raw_complex_rel_l2": error,
                    }
                )
                if error < best_error:
                    best_error = error
                    best_config = config
        assert best_config is not None
        selected[str(budget)] = best_config.name
        subspace = subspaces[best_config.normalization]
        calibration_evaluator = GlobalQ3Evaluator(
            archive,
            subspace,
            "validation",
            budget,
            int(payload["validation_repeats"]),
            int(payload["seed"]),
            device,
        )
        # This is an abstention rule for catastrophic prior shift, not a generic
        # OOD classifier.  A full order-of-magnitude margin above every
        # validation prompt avoids sacrificing moderate, still-predictable shift.
        latent_threshold = float(
            10.0 * calibration_evaluator.latent_energy(best_config).max().cpu()
        )
        split_results = {
            "validation": {
                "operator_mean": float(
                    calibration_evaluator.evaluate(best_config).mean()
                ),
                "train_mean": float(calibration_evaluator.evaluate_mean().mean()),
                "nearest_query": float(
                    calibration_evaluator.evaluate_nearest_query(
                        archive.sources, archive.frequencies
                    ).mean()
                ),
                "system_retrieval": float(
                    calibration_evaluator.evaluate_system_retrieval().mean()
                ),
            }
        }
        for split in ("test", "ood"):
            if not np.any(archive.split == split):
                continue
            evaluator = GlobalQ3Evaluator(
                archive,
                subspace,
                split,
                budget,
                int(payload["test_repeats"]),
                int(payload["seed"]),
                device,
            )
            operator = evaluator.evaluate(best_config)
            mean = evaluator.evaluate_mean()
            nearest = evaluator.evaluate_nearest_query(
                archive.sources, archive.frequencies
            )
            retrieval = evaluator.evaluate_system_retrieval()
            adaptive = mean
            operator_selection_rate = 0.0
            if budget >= 2:
                adaptive, operator_selection_rate = evaluator.evaluate_adaptive_mean(
                    best_config
                )
            latent_gated, latent_gate_rate = evaluator.evaluate_latent_gate(
                best_config, latent_threshold
            )
            split_results[split] = {
                "operator_mean": float(operator.mean()),
                "train_mean": float(mean.mean()),
                "nearest_query": float(nearest.mean()),
                "system_retrieval": float(retrieval.mean()),
                "adaptive_mean": float(adaptive.mean()),
                "adaptive_operator_selection_rate": operator_selection_rate,
                "latent_gated_mean": float(latent_gated.mean()),
                "latent_gate_operator_selection_rate": latent_gate_rate,
                "latent_gate_validation_threshold": latent_threshold,
                "vs_train_mean": paired_summary(mean, operator, int(payload["seed"]) + budget),
                "vs_nearest_query": paired_summary(
                    nearest, operator, int(payload["seed"]) + 100 + budget
                ),
                "vs_system_retrieval": paired_summary(
                    retrieval, operator, int(payload["seed"]) + 200 + budget
                ),
            }
            _write_system_metrics(
                metric_path,
                domain,
                split,
                budget,
                archive.system_ids[archive.split == split],
                {
                    "global_operator": operator,
                    "train_mean": mean,
                    "nearest_query": nearest,
                    "system_retrieval": retrieval,
                    "adaptive_mean": adaptive,
                    "latent_gated_mean": latent_gated,
                },
            )
        results[str(budget)] = split_results

    with (output / "validation_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=validation_rows[0].keys())
        writer.writeheader()
        writer.writerows(validation_rows)
    summary = {
        "domain": domain,
        "dataset": str(dataset.resolve()),
        "task": "strict Q3: target source and target frequency absent from all prompts",
        "selected": selected,
        "reconstruction_error": {
            normalization: {
                str(rank): float(subspace.reconstruction_error[rank - 1])
                for rank in payload["ranks"]
                if rank <= len(subspace.reconstruction_error)
            }
            for normalization, subspace in subspaces.items()
        },
        "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    axis.plot(
        budgets,
        [results[str(value)]["test"]["train_mean"] for value in budgets],
        "o--",
        label="train mean",
    )
    axis.plot(
        budgets,
        [results[str(value)]["test"]["nearest_query"] for value in budgets],
        "^--",
        label="nearest prompt",
    )
    axis.plot(
        budgets,
        [results[str(value)]["test"]["system_retrieval"] for value in budgets],
        "d--",
        label="system retrieval",
    )
    axis.plot(
        budgets,
        [results[str(value)]["test"]["operator_mean"] for value in budgets],
        "s-",
        label="global Bayesian operator",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, labels=[str(value) for value in budgets])
    axis.set_xlabel("Strict-Q3 prompt budget K")
    axis.set_ylabel("Raw complex relative L2")
    axis.set_title(domain)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "q3_budget_curve.png", dpi=220)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_domain(
        args.domain,
        Path(payload["domains"][args.domain]),
        args.output,
        payload,
        args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
