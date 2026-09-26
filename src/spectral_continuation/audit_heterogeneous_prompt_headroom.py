"""Audit representation headroom under heterogeneous receiver prompts.

Each strict-Q3 context response is observed on an independently sampled subset
of receivers.  The audit compares the frozen global Bayesian operator, its
posterior-memory extension, an event-specific supervised complex ridge upper
baseline, and a prompt-shortlist dictionary oracle.  The latter two are
diagnostics rather than deployable shared operators.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import (
    GlobalConfig,
    _q3_events,
    fit_global_subspace,
    solve_complex_posterior,
)
from .probabilistic_response_operator import paired_summary
from .train_prompted_bayesian_neural_operator import response_coordinates


@dataclass(frozen=True)
class FrozenDomain:
    name: str
    budget: int
    rank: int
    value_rank: int
    operator: GlobalConfig
    memory_dimensions: int
    memory_neighbors: int
    memory_temperature: float
    memory_blend: float
    response: torch.Tensor
    split: np.ndarray
    center: torch.Tensor
    scale: torch.Tensor
    basis: torch.Tensor
    variance: torch.Tensor
    penalty: torch.Tensor
    train_raw: torch.Tensor
    train_encoded: torch.Tensor
    memory_coefficients: torch.Tensor
    source_count: int
    frequency_count: int
    receiver_count: int
    coordinates: torch.Tensor


def _penalty(variance: torch.Tensor, config: GlobalConfig) -> torch.Tensor:
    if config.prior == "isotropic":
        return torch.full_like(variance, config.strength)
    reference = torch.quantile(variance, 0.5)
    return config.strength * reference / variance.clamp_min(1e-30)


def _build_domain(
    name: str,
    payload: dict,
    budget: int,
    rank: int,
    value_rank: int,
    device: torch.device,
) -> FrozenDomain:
    archive = load_cross_domain_archive(Path(payload["dataset"]))
    specification = payload["budgets"][str(budget)]
    operator_payload = specification["operator"]
    memory_payload = specification["memory"]
    operator = GlobalConfig(
        rank=rank,
        strength=float(operator_payload["strength"]),
        prior="spectrum",
        normalization=str(operator_payload["normalization"]),
    )
    subspace = fit_global_subspace(archive, rank, operator.normalization, str(device))
    center = torch.as_tensor(subspace.center, device=device)
    scale = torch.as_tensor(subspace.scale, device=device)
    basis = torch.as_tensor(subspace.basis[:, :rank], device=device)
    variance = torch.as_tensor(
        subspace.variance[:rank], dtype=torch.float32, device=device
    )
    response = torch.as_tensor(archive.responses, dtype=torch.complex64, device=device)
    flat = response.reshape(len(response), -1)
    train_raw = flat[torch.as_tensor(archive.split == "train", device=device)]
    train_encoded = (train_raw - center[None]) / scale[None]
    memory_coefficients = (train_encoded @ basis.conj())[:, :value_rank]
    source_count, frequency_count, receiver_count = archive.responses.shape[1:]
    return FrozenDomain(
        name=name,
        budget=budget,
        rank=rank,
        value_rank=value_rank,
        operator=operator,
        memory_dimensions=int(memory_payload["latent_dimensions"]),
        memory_neighbors=int(memory_payload["neighbors"]),
        memory_temperature=float(memory_payload["temperature"]),
        memory_blend=float(memory_payload["blend"]),
        response=response,
        split=archive.split,
        center=center,
        scale=scale,
        basis=basis,
        variance=variance,
        penalty=_penalty(variance, operator),
        train_raw=train_raw,
        train_encoded=train_encoded,
        memory_coefficients=memory_coefficients,
        source_count=source_count,
        frequency_count=frequency_count,
        receiver_count=receiver_count,
        coordinates=torch.as_tensor(response_coordinates(archive), device=device),
    )


def _receiver_masks(
    context: np.ndarray,
    receiver_count: int,
    fraction: float,
    seed: int,
) -> list[np.ndarray]:
    count = max(1, round(fraction * receiver_count))
    masks = []
    for prompt, (source, frequency) in enumerate(context):
        generator = np.random.default_rng(
            seed + 1_000_003 * int(source) + 10_007 * int(frequency) + 97 * prompt
        )
        masks.append(np.sort(generator.choice(receiver_count, count, replace=False)))
    return masks


def _indices(
    context: np.ndarray,
    target_pair: np.ndarray,
    masks: list[np.ndarray],
    frequency_count: int,
    receiver_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    observed = []
    for (source, frequency), receiver in zip(context, masks, strict=True):
        query = int(source) * frequency_count + int(frequency)
        observed.extend((query * receiver_count + receiver).tolist())
    target_query = int(target_pair[0]) * frequency_count + int(target_pair[1])
    target = target_query * receiver_count + np.arange(receiver_count)
    return (
        torch.as_tensor(observed, dtype=torch.long, device=device),
        torch.as_tensor(target, dtype=torch.long, device=device),
    )


def _relative_error(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
        truth, dim=-1
    ).clamp_min(1e-12)


def _memory_prediction(
    domain: FrozenDomain,
    query_coefficient: torch.Tensor,
    bank_coefficient: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    exclude_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = min(domain.memory_dimensions, domain.rank)
    metric_scale = torch.sqrt(domain.variance[:dimension]).clamp_min(1e-15)
    query_metric = query_coefficient[:, :dimension] / metric_scale[None]
    bank_metric = bank_coefficient[:, :dimension] / metric_scale[None]
    distance = (
        query_metric.abs().square().sum(dim=-1, keepdim=True)
        + bank_metric.abs().square().sum(dim=-1)[None]
        - 2.0 * torch.real(query_metric @ bank_metric.conj().T)
    ).clamp_min(0.0)
    if exclude_indices is not None:
        distance = distance.clone()
        distance[
            torch.arange(len(distance), device=distance.device), exclude_indices
        ] = torch.inf
    selected_distance, order = torch.topk(
        distance,
        min(domain.memory_neighbors, len(domain.train_raw)),
        largest=False,
        dim=-1,
    )
    relative = selected_distance - selected_distance[:, :1]
    local_scale = relative[:, -1:].clamp_min(1e-8)
    if domain.memory_temperature <= 0.0 or selected_distance.shape[-1] == 1:
        weight = torch.full_like(relative, 1.0 / selected_distance.shape[-1])
    else:
        weight = torch.softmax(
            -relative / (domain.memory_temperature * local_scale), dim=-1
        )
    coefficient = (
        weight[..., None] * domain.memory_coefficients[order]
    ).sum(dim=1)
    memory_normalized = coefficient @ domain.basis[target, : domain.value_rank].T
    memory_prediction = (
        domain.center[target][None]
        + domain.scale[target][None] * memory_normalized
    )
    prediction = base_prediction + domain.memory_blend * (
        memory_prediction - base_prediction
    )
    return prediction, order


def _ridge_predictions(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    strengths: tuple[float, ...],
) -> list[torch.Tensor]:
    count = len(train_x)
    gram = train_x.mH @ train_x / count
    cross = train_x.mH @ train_y / count
    identity = torch.eye(
        gram.shape[-1], dtype=gram.dtype, device=gram.device
    )
    return [
        query_x @ torch.linalg.solve(gram + strength * identity, cross)
        for strength in strengths
    ]


@torch.inference_mode()
def _evaluate_layout(
    domain: FrozenDomain,
    fraction: float,
    layout_seed: int,
    prompt_seed: int,
    repeats: int,
    strengths: tuple[float, ...],
) -> dict[str, dict[str, np.ndarray]]:
    device = domain.basis.device
    contexts, targets = _q3_events(
        domain.source_count,
        domain.frequency_count,
        domain.budget,
        repeats,
        prompt_seed,
    )
    split_tensors = {}
    for split in ("validation", "test"):
        index = torch.as_tensor(domain.split == split, device=device)
        raw = domain.response[index].reshape(int(index.sum()), -1)
        split_tensors[split] = (raw, (raw - domain.center[None]) / domain.scale[None])
    totals = {
        split: {
            "pca_map": torch.zeros(len(values[0]), device=device),
            "posterior_memory": torch.zeros(len(values[0]), device=device),
            "shortlist_oracle": torch.zeros(len(values[0]), device=device),
            "wide_shortlist_oracle": torch.zeros(len(values[0]), device=device),
            "full_dictionary_oracle": torch.zeros(len(values[0]), device=device),
            "compressed_shortlist_oracle": torch.zeros(len(values[0]), device=device),
            "compressed_wide_shortlist_oracle": torch.zeros(
                len(values[0]), device=device
            ),
            "compressed_full_dictionary_oracle": torch.zeros(
                len(values[0]), device=device
            ),
            "target_subspace_oracle": torch.zeros(len(values[0]), device=device),
            **{
                f"complex_ridge_{strength:g}": torch.zeros(
                    len(values[0]), device=device
                )
                for strength in strengths
            },
        }
        for split, values in split_tensors.items()
    }
    for event, (context, target_pair) in enumerate(
        zip(contexts, targets, strict=True)
    ):
        masks = _receiver_masks(
            context,
            domain.receiver_count,
            fraction,
            layout_seed + event * 1_000_033,
        )
        observed, target = _indices(
            context,
            target_pair,
            masks,
            domain.frequency_count,
            domain.receiver_count,
            device,
        )
        observed_basis = domain.basis[observed]
        bank_coefficient, _ = solve_complex_posterior(
            observed_basis,
            domain.train_encoded[:, observed],
            domain.penalty,
        )
        train_x = domain.train_encoded[:, observed]
        train_y = domain.train_encoded[:, target]
        for split, (raw, encoded) in split_tensors.items():
            query_coefficient, _ = solve_complex_posterior(
                observed_basis, encoded[:, observed], domain.penalty
            )
            normalized = query_coefficient @ domain.basis[target].T
            base = domain.center[target][None] + domain.scale[target][None] * normalized
            truth = raw[:, target]
            totals[split]["pca_map"] += _relative_error(base, truth)
            memory, order = _memory_prediction(
                domain, query_coefficient, bank_coefficient, base, target
            )
            totals[split]["posterior_memory"] += _relative_error(memory, truth)
            shortlist = domain.train_raw[order][:, :, target]
            shortlist_error = _relative_error(shortlist, truth[:, None])
            totals[split]["shortlist_oracle"] += shortlist_error.min(dim=1).values
            compressed_shortlist = domain.center[target][None, None] + domain.scale[
                target
            ][None, None] * (
                domain.memory_coefficients[order]
                @ domain.basis[target, : domain.value_rank].T
            )
            compressed_shortlist_error = _relative_error(
                compressed_shortlist, truth[:, None]
            )
            totals[split]["compressed_shortlist_oracle"] += (
                compressed_shortlist_error.min(dim=1).values
            )
            wide_count = min(64, len(domain.train_raw))
            _, wide_order = torch.topk(
                (
                    (query_coefficient[:, : domain.memory_dimensions] / torch.sqrt(
                        domain.variance[: domain.memory_dimensions]
                    )[None].clamp_min(1e-15)).abs().square().sum(dim=-1, keepdim=True)
                    + (bank_coefficient[:, : domain.memory_dimensions] / torch.sqrt(
                        domain.variance[: domain.memory_dimensions]
                    )[None].clamp_min(1e-15)).abs().square().sum(dim=-1)[None]
                    - 2.0 * torch.real(
                        (query_coefficient[:, : domain.memory_dimensions] / torch.sqrt(
                            domain.variance[: domain.memory_dimensions]
                        )[None].clamp_min(1e-15))
                        @ (bank_coefficient[:, : domain.memory_dimensions] / torch.sqrt(
                            domain.variance[: domain.memory_dimensions]
                        )[None].clamp_min(1e-15)).conj().T
                    )
                ).clamp_min(0.0),
                wide_count,
                largest=False,
                dim=-1,
            )
            wide_error = _relative_error(
                domain.train_raw[wide_order][:, :, target], truth[:, None]
            )
            totals[split]["wide_shortlist_oracle"] += wide_error.min(dim=1).values
            compressed_wide = domain.center[target][None, None] + domain.scale[
                target
            ][None, None] * (
                domain.memory_coefficients[wide_order]
                @ domain.basis[target, : domain.value_rank].T
            )
            compressed_wide_error = _relative_error(compressed_wide, truth[:, None])
            totals[split]["compressed_wide_shortlist_oracle"] += (
                compressed_wide_error.min(dim=1).values
            )
            full_error = _relative_error(
                domain.train_raw[None, :, target], truth[:, None]
            )
            totals[split]["full_dictionary_oracle"] += full_error.min(dim=1).values
            compressed_full = domain.center[target][None] + domain.scale[target][None] * (
                domain.memory_coefficients
                @ domain.basis[target, : domain.value_rank].T
            )
            compressed_full_error = _relative_error(
                compressed_full[None], truth[:, None]
            )
            totals[split]["compressed_full_dictionary_oracle"] += (
                compressed_full_error.min(dim=1).values
            )
            truth_normalized = (
                truth - domain.center[target][None]
            ) / domain.scale[target][None]
            proposal = truth_normalized @ torch.linalg.pinv(domain.basis[target]).T
            projected = domain.center[target][None] + domain.scale[target][None] * (
                proposal @ domain.basis[target].T
            )
            totals[split]["target_subspace_oracle"] += _relative_error(
                projected, truth
            )
            for strength, prediction in zip(
                strengths,
                _ridge_predictions(
                    train_x,
                    train_y,
                    encoded[:, observed],
                    strengths,
                ),
                strict=True,
            ):
                decoded = domain.center[target][None] + domain.scale[target][None] * prediction
                totals[split][f"complex_ridge_{strength:g}"] += _relative_error(
                    decoded, truth
                )
    return {
        split: {
            name: (value / len(contexts)).cpu().numpy().astype(np.float64)
            for name, value in methods.items()
        }
        for split, methods in totals.items()
    }


def run(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    strengths = tuple(float(value) for value in config["ridge_strengths"])
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for domain_name, payload in config["domains"].items():
        domain = _build_domain(
            domain_name,
            payload,
            int(config["budget"]),
            int(config["rank"]),
            int(config["value_rank"]),
            device,
        )
        for fraction in config["receiver_fractions"]:
            key = f"{domain_name}_r{round(100 * fraction):03d}"
            result = _evaluate_layout(
                domain,
                float(fraction),
                int(config["layout_seed"]),
                int(config["prompt_seed"]),
                int(config["repeats"]),
                strengths,
            )
            validation = result["validation"]
            ridge_names = [name for name in validation if name.startswith("complex_ridge")]
            selected_ridge = min(ridge_names, key=lambda name: validation[name].mean())
            test = result["test"]
            methods = (
                "pca_map",
                "posterior_memory",
                selected_ridge,
                "shortlist_oracle",
                "wide_shortlist_oracle",
                "full_dictionary_oracle",
                "compressed_shortlist_oracle",
                "compressed_wide_shortlist_oracle",
                "compressed_full_dictionary_oracle",
                "target_subspace_oracle",
            )
            arrays = {name: test[name] for name in methods}
            np.savez_compressed(
                output / f"{key.lower()}_systems.npz",
                **arrays,
            )
            baseline = arrays["posterior_memory"]
            row = {
                "receiver_fraction": float(fraction),
                "receivers_per_prompt": max(
                    1, round(float(fraction) * domain.receiver_count)
                ),
                "selected_ridge": selected_ridge,
                "methods": {},
            }
            for offset, (name, value) in enumerate(arrays.items()):
                row["methods"][name] = {
                    "mean": float(value.mean()),
                    **(
                        {}
                        if name == "posterior_memory"
                        else paired_summary(baseline, value, 20260926 + offset)
                    ),
                }
            summary[key] = row
    payload = {"config": config, "results": summary}
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run(args.config, args.output, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
