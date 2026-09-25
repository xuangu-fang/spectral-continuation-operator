"""Learn a query-optimized complex response basis from sparse prompt episodes.

This bridge experiment keeps the deployable closed-form latent solve of the
response-subspace baseline, but optimizes its mean and basis for held-out source
queries rather than full-table explained variance.  It determines whether an
episodic PFTO objective has headroom before fitting a continuous coordinate
decoder to the learned basis bank.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit_functional_tensor_rank import prompt_query_errors
from .cross_domain_query_audit import FrequencySubspace, _fit_subspaces
from .cross_domain_response_data import load_cross_domain_archive


class EpisodicResponseBasis(nn.Module):
    def __init__(self, subspaces: list[FrequencySubspace], source_count: int, receiver_count: int):
        super().__init__()
        mean = np.stack([subspace.mean for subspace in subspaces])
        basis = np.stack([subspace.basis for subspace in subspaces])
        self.mean_parts = nn.Parameter(torch.view_as_real(torch.from_numpy(mean)).float())
        self.basis_parts = nn.Parameter(torch.view_as_real(torch.from_numpy(basis)).float())
        self.source_count = source_count
        self.receiver_count = receiver_count

    def complex_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.view_as_complex(self.mean_parts), torch.view_as_complex(self.basis_parts)

    def forward(
        self,
        context_value: torch.Tensor,
        context_source: torch.Tensor,
        target_source: torch.Tensor,
        frequency: torch.Tensor,
        ridge: float,
    ) -> torch.Tensor:
        mean, basis = self.complex_parameters()
        batch, budget, receivers = context_value.shape
        rank = basis.shape[-1]
        receiver = torch.arange(receivers, device=context_value.device)
        observed = (
            context_source[:, :, None] * receivers + receiver[None, None]
        ).reshape(batch, -1)
        mean_batch = mean[frequency]
        basis_batch = basis[frequency]
        observed_mean = torch.gather(mean_batch, 1, observed)
        observed_basis = torch.gather(
            basis_batch, 1, observed[:, :, None].expand(-1, -1, rank)
        )
        centered = context_value.reshape(batch, budget * receivers) - observed_mean
        adjoint = observed_basis.conj().transpose(1, 2)
        gram = adjoint @ observed_basis
        identity = torch.eye(rank, dtype=gram.dtype, device=gram.device)[None]
        coefficient = torch.linalg.solve(
            gram + ridge * identity,
            adjoint @ centered[:, :, None],
        )
        target_index = target_source[:, None] * receivers + receiver[None]
        target_mean = torch.gather(mean_batch, 1, target_index)
        target_basis = torch.gather(
            basis_batch, 1, target_index[:, :, None].expand(-1, -1, rank)
        )
        return target_mean + (target_basis @ coefficient)[:, :, 0]

    @torch.no_grad()
    def retract(self, receiver_index: np.ndarray, orthogonalize: bool = False) -> None:
        """Enforce reciprocity and fix column scale without changing complex phase."""

        mean, basis = self.complex_parameters()
        valid = np.flatnonzero(receiver_index >= 0)
        matched_receiver = receiver_index[valid]
        for frequency in range(len(mean)):
            mean_table = mean[frequency].reshape(self.source_count, self.receiver_count)
            block = mean_table[valid][:, matched_receiver]
            symmetric = 0.5 * (block + block.T)
            for local_source, source in enumerate(valid):
                mean_table[source, matched_receiver] = symmetric[local_source]
            for column in range(basis.shape[-1]):
                table = basis[frequency, :, column].reshape(
                    self.source_count, self.receiver_count
                )
                block = table[valid][:, matched_receiver]
                symmetric = 0.5 * (block + block.T)
                for local_source, source in enumerate(valid):
                    table[source, matched_receiver] = symmetric[local_source]
            if orthogonalize:
                orthogonal, _ = torch.linalg.qr(basis[frequency], mode="reduced")
                basis[frequency].copy_(orthogonal)
            else:
                norm = torch.linalg.vector_norm(basis[frequency], dim=0).clamp_min(1e-8)
                basis[frequency].div_(norm[None])

    def numpy_subspaces(self) -> list[FrequencySubspace]:
        mean, basis = self.complex_parameters()
        mean_array = mean.detach().cpu().numpy().astype(np.complex64)
        basis_array = basis.detach().cpu().numpy().astype(np.complex64)
        return [
            FrequencySubspace(mean=mean_array[index], basis=basis_array[index])
            for index in range(len(mean_array))
        ]


def _sample_episode(
    generator: np.random.Generator,
    train_systems: np.ndarray,
    response: torch.Tensor,
    batch_size: int,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_count = response.shape[1]
    frequency_count = response.shape[2]
    system = generator.choice(train_systems, size=batch_size)
    target = generator.integers(source_count, size=batch_size)
    frequency = generator.integers(frequency_count, size=batch_size)
    context = np.empty((batch_size, budget), dtype=np.int64)
    for index, target_source in enumerate(target):
        candidates = np.delete(np.arange(source_count), target_source)
        context[index] = generator.choice(candidates, size=budget, replace=False)
    device = response.device
    system_tensor = torch.from_numpy(system).to(device)
    target_tensor = torch.from_numpy(target).to(device)
    frequency_tensor = torch.from_numpy(frequency).to(device)
    context_tensor = torch.from_numpy(context).to(device)
    context_value = response[
        system_tensor[:, None], context_tensor, frequency_tensor[:, None]
    ]
    target_value = response[system_tensor, target_tensor, frequency_tensor]
    return context_value, context_tensor, target_tensor, frequency_tensor, target_value


def _validation_error(
    model: EpisodicResponseBasis,
    archive,
    budget: int,
    ridge: float,
    repeats: int,
    seed: int,
) -> float:
    errors = prompt_query_errors(
        archive,
        model.numpy_subspaces(),
        "validation",
        budget,
        ridge,
        repeats,
        seed,
    )
    return float(errors.mean())


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    archive = load_cross_domain_archive(args.dataset)
    source_count, _, receiver_count = archive.responses.shape[1:]
    initial = _fit_subspaces(archive, max_rank=args.rank)
    model = EpisodicResponseBasis(initial, source_count, receiver_count).to(device)
    model.retract(archive.source_receiver_index, orthogonalize=True)
    response = torch.from_numpy(archive.responses).to(device)
    train_systems = np.flatnonzero(archive.split == "train")
    budgets = tuple(int(value) for value in args.train_budgets.split(","))
    generator = np.random.default_rng(args.seed)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.mean_parts], "lr": args.mean_learning_rate},
            {"params": [model.basis_parts], "lr": args.learning_rate},
        ]
    )
    initial_validation = _validation_error(
        model, archive, args.eval_budget, args.ridge, args.eval_repeats, args.eval_seed
    )
    best_validation = initial_validation
    best_step = 0
    best_mean, best_basis = [part.detach().cpu().clone() for part in model.complex_parameters()]
    history = [{"step": 0, "validation_rel_l2": initial_validation}]
    start_time = time.time()
    for step in range(1, args.steps + 1):
        budget = budgets[int(generator.integers(len(budgets)))]
        context_value, context, target, frequency, truth = _sample_episode(
            generator,
            train_systems,
            response,
            args.batch_size,
            budget,
        )
        prediction = model(context_value, context, target, frequency, args.ridge)
        per_example = torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.clamp(
            torch.linalg.vector_norm(truth, dim=-1), min=1e-8
        )
        loss = per_example.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        model.retract(archive.source_receiver_index, orthogonalize=False)
        if step % args.eval_every == 0 or step == args.steps:
            validation = _validation_error(
                model,
                archive,
                args.eval_budget,
                args.ridge,
                args.eval_repeats,
                args.eval_seed,
            )
            history.append(
                {
                    "step": step,
                    "train_rel_l2": float(loss.detach().cpu()),
                    "validation_rel_l2": validation,
                }
            )
            if validation < best_validation:
                best_validation = validation
                best_step = step
                best_mean, best_basis = [
                    part.detach().cpu().clone() for part in model.complex_parameters()
                ]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "best_basis.npz",
        mean=best_mean.numpy().astype(np.complex64),
        basis=best_basis.numpy().astype(np.complex64),
    )
    summary = {
        "dataset": str(args.dataset.resolve()),
        "split": "train optimization; validation selection; test unopened",
        "seed": args.seed,
        "rank": args.rank,
        "ridge": args.ridge,
        "basis_learning_rate": args.learning_rate,
        "mean_learning_rate": args.mean_learning_rate,
        "train_budgets": list(budgets),
        "eval_budget": args.eval_budget,
        "parameter_count_real": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation_rel_l2": initial_validation,
        "best_validation_rel_l2": best_validation,
        "relative_improvement": 1.0 - best_validation / initial_validation,
        "best_step": best_step,
        "elapsed_seconds": time.time() - start_time,
        "history": history,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--ridge", type=float, required=True)
    parser.add_argument("--train-budgets", required=True)
    parser.add_argument("--eval-budget", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--mean-learning-rate", type=float, default=2e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-repeats", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=20260925)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
