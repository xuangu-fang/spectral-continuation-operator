"""Interleaved receiver-coordinate holdout for Wave-as-Prompt operators."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline

from .cross_domain_query_audit import source_orders
from .cross_domain_response_data import load_cross_domain_archive
from .probabilistic_response_operator import LatentConfig, paired_summary


@dataclass(frozen=True)
class RestrictedSubspace:
    mean: np.ndarray
    basis: np.ndarray
    variance: np.ndarray


def receiver_split(
    receiver_count: int, periodic: bool, holdout_stride: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    if holdout_stride < 2:
        raise ValueError("holdout_stride must be at least 2")
    query = np.arange(1, receiver_count, holdout_stride)
    if not periodic:
        query = query[query != receiver_count - 1]
    observed = np.setdiff1d(np.arange(receiver_count), query)
    return observed, query


def interpolation_matrix(
    receiver_count: int,
    observed: np.ndarray,
    query: np.ndarray,
    method: str,
) -> np.ndarray:
    x_observed = observed / receiver_count
    x_query = query / receiver_count
    identity = np.eye(len(observed))
    if method == "linear":
        matrix = np.stack(
            [np.interp(x_query, x_observed, column) for column in identity.T], axis=1
        )
    elif method == "periodic_linear":
        x_extended = np.append(x_observed, 1.0)
        values = np.concatenate((identity, identity[:1]), axis=0)
        matrix = np.stack(
            [np.interp(x_query, x_extended, column) for column in values.T], axis=1
        )
    elif method == "natural_cubic":
        matrix = CubicSpline(x_observed, identity, axis=0, bc_type="natural")(x_query)
    elif method == "periodic_cubic":
        x_extended = np.append(x_observed, 1.0)
        values = np.concatenate((identity, identity[:1]), axis=0)
        matrix = CubicSpline(x_extended, values, axis=0, bc_type="periodic")(x_query)
    elif method == "periodic_fourier":
        count = len(observed)
        modes = np.fft.fftfreq(count) * count
        matrix = (
            np.exp(2j * np.pi * x_query[:, None] * modes[None])
            @ np.exp(-2j * np.pi * modes[:, None] * x_observed[None])
            / count
        )
    else:
        raise ValueError(f"unknown interpolation method {method!r}")
    return np.asarray(matrix, dtype=np.complex64)


def fit_restricted_subspaces(
    response: np.ndarray,
    split: np.ndarray,
    observed: np.ndarray,
    maximum_rank: int,
) -> list[RestrictedSubspace]:
    train = response[split == "train"][..., observed]
    result = []
    for frequency in range(train.shape[2]):
        table = train[:, :, frequency].reshape(len(train), -1)
        mean = table.mean(axis=0)
        _, singular, right = np.linalg.svd(table - mean, full_matrices=False)
        rank = min(maximum_rank, len(singular))
        result.append(
            RestrictedSubspace(
                mean.astype(np.complex64),
                right[:rank].T.astype(np.complex64),
                (np.square(singular[:rank]) / max(len(train) - 1, 1)).astype(np.float32),
            )
        )
    return result


class ContinuousQueryEvaluator:
    def __init__(
        self,
        response: np.ndarray,
        split_labels: np.ndarray,
        subspaces: list[RestrictedSubspace],
        split: str,
        observed_receiver: np.ndarray,
        query_receiver: np.ndarray,
        projection: np.ndarray,
        budget: int,
        repeats: int,
        seed: int,
        device: str,
    ) -> None:
        system = np.flatnonzero(split_labels == split)
        selected_response = response[system]
        observed_response = selected_response[..., observed_receiver]
        source_count, frequency_count, receiver_count = observed_response.shape[1:]
        orders = source_orders(source_count, repeats, seed)
        contexts = []
        truths = []
        observed_indices = []
        target_indices = []
        frequencies = []
        receiver = np.arange(receiver_count)
        for target in range(source_count):
            for order in orders[target]:
                context = order[:budget]
                observed = np.concatenate(
                    [source * receiver_count + receiver for source in context]
                )
                prompt = observed_response[:, context]
                for frequency in range(frequency_count):
                    contexts.append(prompt[:, :, frequency].reshape(len(system), -1))
                    truths.append(selected_response[:, target, frequency, query_receiver])
                    observed_indices.append(observed)
                    target_indices.append(target * receiver_count + receiver)
                    frequencies.append(frequency)
        torch_device = torch.device(device)
        self.context = torch.as_tensor(
            np.stack(contexts), dtype=torch.complex64, device=torch_device
        )
        self.truth = torch.as_tensor(
            np.stack(truths), dtype=torch.complex64, device=torch_device
        )
        self.observed = torch.as_tensor(
            np.stack(observed_indices), dtype=torch.long, device=torch_device
        )
        self.target = torch.as_tensor(
            np.stack(target_indices), dtype=torch.long, device=torch_device
        )
        self.frequency = torch.as_tensor(frequencies, dtype=torch.long, device=torch_device)
        self.projection = torch.as_tensor(projection, dtype=torch.complex64, device=torch_device)
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

    @torch.inference_mode()
    def evaluate(self, config: LatentConfig) -> np.ndarray:
        rank = config.rank
        basis = self.basis[self.frequency, :, :rank]
        mean = self.mean[self.frequency]
        variance = self.variance[self.frequency, :rank]
        design = torch.gather(
            basis, 1, self.observed[:, :, None].expand(-1, -1, rank)
        )
        observed_mean = torch.gather(mean, 1, self.observed)
        if config.prior == "isotropic":
            penalty = torch.full_like(variance, config.strength)
        elif config.prior == "spectrum":
            reference = torch.quantile(variance, 0.5, dim=1, keepdim=True)
            penalty = config.strength * reference / variance.clamp_min(1e-30)
        else:
            raise ValueError(config.prior)
        adjoint = design.conj().transpose(1, 2)
        projector = torch.linalg.solve(
            adjoint @ design + torch.diag_embed(penalty).to(torch.complex64), adjoint
        )
        coefficient = torch.einsum(
            "eno,elo->enl", self.context - observed_mean[:, None], projector
        )
        target_basis = torch.gather(
            basis, 1, self.target[:, :, None].expand(-1, -1, rank)
        )
        target_mean = torch.gather(mean, 1, self.target)
        target_basis = torch.einsum("qr,erl->eql", self.projection, target_basis)
        target_mean = torch.einsum("qr,er->eq", self.projection, target_mean)
        prediction = target_mean[:, None] + torch.einsum(
            "enl,eql->enq", coefficient, target_basis
        )
        error = torch.linalg.vector_norm(prediction - self.truth, dim=-1) / torch.linalg.vector_norm(
            self.truth, dim=-1
        ).clamp_min(1e-12)
        return error.mean(dim=0).cpu().numpy().astype(np.float64)


def oracle_interpolation_error(
    response: np.ndarray,
    split: np.ndarray,
    observed: np.ndarray,
    query: np.ndarray,
    projection: np.ndarray,
    split_name: str,
) -> float:
    values = response[split == split_name]
    prediction = np.einsum("qr,nsfr->nsfq", projection, values[..., observed])
    truth = values[..., query]
    error = np.linalg.norm(prediction - truth, axis=-1) / np.maximum(
        np.linalg.norm(truth, axis=-1), 1e-12
    )
    return float(error.mean())


def run_domain(
    domain: str,
    payload: dict,
    ranks: tuple[int, ...],
    scalar_strengths: tuple[float, ...],
    spectrum_strengths: tuple[float, ...],
    seeds: tuple[int, ...],
    validation_seed: int,
    validation_repeats: int,
    test_repeats: int,
    device: str,
) -> dict:
    archive = load_cross_domain_archive(Path(payload["dataset"]))
    periodic = bool(payload["periodic"])
    budget = int(payload["budget"])
    holdout_stride = int(payload.get("holdout_stride", 2))
    observed, query = receiver_split(
        archive.responses.shape[-1], periodic, holdout_stride
    )
    subspaces = fit_restricted_subspaces(
        archive.responses, archive.split, observed, max(ranks)
    )
    methods = tuple(payload["interpolation"])
    candidates = {
        "scalar": tuple(
            LatentConfig(rank, strength, "isotropic")
            for rank in ranks
            for strength in scalar_strengths
        ),
        "spectrum": tuple(
            LatentConfig(rank, strength, "spectrum")
            for rank in ranks
            for strength in spectrum_strengths
        ),
    }
    selected = {}
    validation_rows = []
    for family, configs in candidates.items():
        best = (float("inf"), methods[0], configs[0])
        for method in methods:
            projection = interpolation_matrix(
                archive.responses.shape[-1], observed, query, method
            )
            evaluator = ContinuousQueryEvaluator(
                archive.responses,
                archive.split,
                subspaces,
                "validation",
                observed,
                query,
                projection,
                budget,
                validation_repeats,
                validation_seed,
                device,
            )
            for config in configs:
                if config.rank > subspaces[0].basis.shape[1]:
                    continue
                error = float(evaluator.evaluate(config).mean())
                validation_rows.append(
                    {
                        "family": family,
                        "method": method,
                        "config": config.name,
                        "error": error,
                    }
                )
                if error < best[0]:
                    best = (error, method, config)
        selected[family] = {"method": best[1], "config": best[2]}

    test_errors = {"scalar": [], "spectrum": []}
    for family, family_errors in test_errors.items():
        choice = selected[family]
        projection = interpolation_matrix(
            archive.responses.shape[-1], observed, query, choice["method"]
        )
        for seed in seeds:
            evaluator = ContinuousQueryEvaluator(
                archive.responses,
                archive.split,
                subspaces,
                "test",
                observed,
                query,
                projection,
                budget,
                test_repeats,
                seed,
                device,
            )
            family_errors.append(evaluator.evaluate(choice["config"]))
    scalar = np.stack(test_errors["scalar"]).mean(axis=0)
    spectrum = np.stack(test_errors["spectrum"]).mean(axis=0)
    oracle = {
        method: oracle_interpolation_error(
            archive.responses,
            archive.split,
            observed,
            query,
            interpolation_matrix(archive.responses.shape[-1], observed, query, method),
            "test",
        )
        for method in methods
    }
    return {
        "domain": domain,
        "budget": budget,
        "receiver_count": int(archive.responses.shape[-1]),
        "observed_receivers": len(observed),
        "query_receivers": len(query),
        "holdout_stride": holdout_stride,
        "coordinate_policy": "globally interleaved receiver holdout",
        "selected": {
            family: {
                "method": value["method"],
                "config": value["config"].name,
            }
            for family, value in selected.items()
        },
        "oracle_interpolation_rel_l2": oracle,
        "test": paired_summary(scalar, spectrum, seeds[0] + budget),
        "validation_grid": validation_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    result = {}
    for domain, domain_payload in payload["domains"].items():
        result[domain] = run_domain(
            domain,
            domain_payload,
            tuple(int(value) for value in payload["ranks"]),
            tuple(float(value) for value in payload["scalar_strengths"]),
            tuple(float(value) for value in payload["spectrum_strengths"]),
            tuple(int(value) for value in payload["test_seeds"]),
            int(payload["validation_seed"]),
            int(payload["validation_repeats"]),
            int(payload["test_repeats"]),
            args.device,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, len(result), figsize=(5.2 * len(result), 4.0))
    if len(result) == 1:
        axes = [axes]
    for axis, (domain, values) in zip(axes, result.items(), strict=True):
        names = ["scalar ridge", "spectrum prior"]
        errors = [values["test"]["baseline_mean"], values["test"]["candidate_mean"]]
        axis.bar(names, errors, color=("#777777", "#2a6fbb"))
        axis.set_title(domain)
        axis.set_ylabel("Held-out receiver raw complex RelL2")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output / "continuous_query.png", dpi=220)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
