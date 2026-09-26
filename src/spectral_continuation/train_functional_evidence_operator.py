"""Train a coordinate-aware functional evidence operator with analytic fallback."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit_heterogeneous_prompt_headroom import (
    FrozenDomain,
    _build_domain,
    _indices,
    _memory_prediction,
    _receiver_masks,
    _relative_error,
)
from .global_bayesian_response_operator import _q3_events, solve_complex_posterior
from .probabilistic_response_operator import paired_summary
from .train_prompted_bayesian_neural_operator import fourier_features


@dataclass
class EvidenceEpisode:
    fraction: float
    prompt_values: torch.Tensor
    prompt_coordinates: torch.Tensor
    target_coordinates: torch.Tensor
    baseline_normalized: torch.Tensor
    target_center: torch.Tensor
    target_scale: torch.Tensor
    truth: torch.Tensor


class FunctionalEvidenceOperator(nn.Module):
    """Nonlinear functional evidence accumulator and continuous complex decoder."""

    def __init__(
        self,
        coordinate_dimensions: int,
        latent_dimensions: int,
        width: int,
        residual_bound: float,
    ) -> None:
        super().__init__()
        coordinate_features = 11 * coordinate_dimensions
        self.latent_dimensions = latent_dimensions
        self.residual_bound = residual_bound
        self.point_encoder = nn.Sequential(
            nn.Linear(coordinate_features + 3, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.prompt_encoder = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, 3 * latent_dimensions),
        )
        self.query_decoder = nn.Sequential(
            nn.Linear(coordinate_features + 3, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 2 * latent_dimensions + 2),
        )
        self.prior_precision = nn.Parameter(torch.ones(latent_dimensions))
        final = self.query_decoder[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("query decoder must end in a linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        prompt_values: torch.Tensor,
        prompt_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        baseline_normalized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = prompt_values.shape[0]
        prompt_values = prompt_values.clone()
        baseline_normalized = baseline_normalized.clone()
        coordinate = fourier_features(prompt_coordinates.clone())
        coordinate = coordinate[None].expand(batch, -1, -1, -1)
        point_input = torch.cat(
            (
                coordinate,
                prompt_values.real[..., None],
                prompt_values.imag[..., None],
                torch.log1p(prompt_values.abs())[..., None],
            ),
            dim=-1,
        )
        point = self.point_encoder(point_input)
        pooled = torch.cat((point.mean(dim=2), point.amax(dim=2)), dim=-1)
        evidence = self.prompt_encoder(pooled)
        real, imaginary, raw_precision = evidence.split(
            self.latent_dimensions, dim=-1
        )
        candidate = torch.complex(real, imaginary)
        precision = torch.nn.functional.softplus(raw_precision) + 1e-3
        prior = torch.nn.functional.softplus(self.prior_precision) + 1e-3
        posterior_precision = prior[None] + precision.sum(dim=1)
        latent = (precision * candidate).sum(dim=1) / posterior_precision
        uncertainty = posterior_precision.reciprocal().mean(dim=-1)

        query_coordinate = fourier_features(target_coordinates.clone())
        query_coordinate = query_coordinate[None].expand(batch, -1, -1)
        query_input = torch.cat(
            (
                query_coordinate,
                baseline_normalized.real[..., None],
                baseline_normalized.imag[..., None],
                torch.log1p(uncertainty)[:, None, None].expand(
                    -1, len(target_coordinates), -1
                ),
            ),
            dim=-1,
        )
        decoded = self.query_decoder(query_input)
        basis_real, basis_imaginary, direct = torch.split(
            decoded,
            (self.latent_dimensions, self.latent_dimensions, 2),
            dim=-1,
        )
        basis = torch.complex(basis_real, basis_imaginary)
        residual = (basis * latent[:, None]).sum(dim=-1) / np.sqrt(
            self.latent_dimensions
        )
        residual = residual + torch.complex(direct[..., 0], direct[..., 1])
        bounded = self.residual_bound * torch.complex(
            torch.tanh(residual.real), torch.tanh(residual.imag)
        )
        return baseline_normalized + bounded, uncertainty


@torch.inference_mode()
def _cache_episodes(
    domain: FrozenDomain,
    split: str,
    fractions: tuple[float, ...],
    repeats: int,
    prompt_seed: int,
    layout_seed: int,
) -> dict[float, list[EvidenceEpisode]]:
    device = domain.basis.device
    index = torch.as_tensor(domain.split == split, device=device)
    raw = domain.response[index].reshape(int(index.sum()), -1)
    encoded = (raw - domain.center[None]) / domain.scale[None]
    contexts, targets = _q3_events(
        domain.source_count,
        domain.frequency_count,
        domain.budget,
        repeats,
        prompt_seed,
    )
    result = {fraction: [] for fraction in fractions}
    for fraction in fractions:
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
            query_coefficient, _ = solve_complex_posterior(
                observed_basis, encoded[:, observed], domain.penalty
            )
            bank_coefficient, _ = solve_complex_posterior(
                observed_basis,
                domain.train_encoded[:, observed],
                domain.penalty,
            )
            normalized = query_coefficient @ domain.basis[target].T
            base = domain.center[target][None] + domain.scale[target][None] * normalized
            exclude = (
                torch.arange(len(raw), device=device) if split == "train" else None
            )
            memory, _ = _memory_prediction(
                domain,
                query_coefficient,
                bank_coefficient,
                base,
                target,
                exclude,
            )
            count = len(masks[0])
            result[fraction].append(
                EvidenceEpisode(
                    fraction=fraction,
                    prompt_values=encoded[:, observed].reshape(
                        len(raw), domain.budget, count
                    ),
                    prompt_coordinates=domain.coordinates[observed].reshape(
                        domain.budget, count, -1
                    ),
                    target_coordinates=domain.coordinates[target],
                    baseline_normalized=(
                        memory - domain.center[target][None]
                    )
                    / domain.scale[target][None],
                    target_center=domain.center[target],
                    target_scale=domain.scale[target],
                    truth=raw[:, target],
                )
            )
    return result


def _decode(normalized: torch.Tensor, episode: EvidenceEpisode) -> torch.Tensor:
    return episode.target_center.clone()[None] + episode.target_scale.clone()[None] * normalized


@torch.inference_mode()
def evaluate(
    model: FunctionalEvidenceOperator,
    cached: dict[float, list[EvidenceEpisode]],
    batch_size: int,
) -> dict[str, dict[str, np.ndarray | float]]:
    model.eval()
    result = {}
    all_baseline = None
    all_candidate = None
    for fraction, episodes in cached.items():
        systems = len(episodes[0].truth)
        baseline_total = torch.zeros(systems, device=episodes[0].truth.device)
        candidate_total = torch.zeros_like(baseline_total)
        uncertainty_total = torch.zeros_like(baseline_total)
        for episode in episodes:
            baseline = _decode(episode.baseline_normalized, episode)
            baseline_total += _relative_error(baseline, episode.truth)
            for start in range(0, systems, batch_size):
                stop = min(start + batch_size, systems)
                normalized, uncertainty = model(
                    episode.prompt_values[start:stop],
                    episode.prompt_coordinates,
                    episode.target_coordinates,
                    episode.baseline_normalized[start:stop],
                )
                prediction = _decode(normalized, episode)
                candidate_total[start:stop] += _relative_error(
                    prediction, episode.truth[start:stop]
                )
                uncertainty_total[start:stop] += uncertainty
        baseline_error = (baseline_total / len(episodes)).cpu().numpy().astype(np.float64)
        candidate_error = (
            candidate_total / len(episodes)
        ).cpu().numpy().astype(np.float64)
        uncertainty = (
            uncertainty_total / len(episodes)
        ).cpu().numpy().astype(np.float64)
        key = f"r{round(100 * fraction):03d}"
        result[key] = {
            "baseline_mean": float(baseline_error.mean()),
            "candidate_mean": float(candidate_error.mean()),
            **paired_summary(baseline_error, candidate_error, 20260926),
            "uncertainty_error_correlation": float(
                np.corrcoef(uncertainty, candidate_error)[0, 1]
            ),
            "_baseline": baseline_error,
            "_candidate": candidate_error,
            "_uncertainty": uncertainty,
        }
        all_baseline = baseline_error if all_baseline is None else all_baseline + baseline_error
        all_candidate = candidate_error if all_candidate is None else all_candidate + candidate_error
    if all_baseline is None or all_candidate is None:
        raise ValueError("evaluation cache is empty")
    result["all"] = {
        "baseline_mean": float(all_baseline.mean() / len(cached)),
        "candidate_mean": float(all_candidate.mean() / len(cached)),
    }
    return result


def _model(config: dict, device: torch.device) -> FunctionalEvidenceOperator:
    return FunctionalEvidenceOperator(
        coordinate_dimensions=5,
        latent_dimensions=int(config["latent_dimensions"]),
        width=int(config["width"]),
        residual_bound=float(config["residual_bound"]),
    ).to(device)


def train_domain(
    name: str,
    payload: dict,
    config: dict,
    output: Path,
    device: torch.device,
) -> dict:
    torch.manual_seed(int(config["seed"]))
    generator = np.random.default_rng(int(config["seed"]))
    domain = _build_domain(
        name,
        payload,
        int(config["budget"]),
        int(config["rank"]),
        int(config["value_rank"]),
        device,
    )
    fractions = tuple(float(value) for value in config["receiver_fractions"])
    started = time.perf_counter()
    train_cache = _cache_episodes(
        domain,
        "train",
        fractions,
        int(config["train_repeats"]),
        int(config["seed"]),
        int(config["layout_seed"]),
    )
    validation_cache = _cache_episodes(
        domain,
        "validation",
        fractions,
        int(config["validation_repeats"]),
        int(config["selection_seed"]),
        int(config["validation_layout_seed"]),
    )
    model = _model(config, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= 3_000_000:
        raise ValueError(f"model has {parameters:,} parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    initial = evaluate(model, validation_cache, int(config["eval_batch_size"]))
    anchors = {
        key: float(value["candidate_mean"])
        for key, value in initial.items()
        if key != "all"
    }
    best_score = 1.25
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    history = [{"step": 0, "selection_score": best_score}]
    batch_size = int(config["batch_size"])
    for step in range(1, int(config["steps"]) + 1):
        fraction = fractions[int(generator.integers(len(fractions)))]
        episodes = train_cache[fraction]
        episode = episodes[int(generator.integers(len(episodes)))]
        index = torch.as_tensor(
            generator.integers(0, len(episode.truth), size=batch_size),
            dtype=torch.long,
            device=device,
        )
        normalized, _ = model(
            episode.prompt_values[index],
            episode.prompt_coordinates,
            episode.target_coordinates,
            episode.baseline_normalized[index],
        )
        prediction = _decode(normalized, episode)
        loss = _relative_error(prediction, episode.truth[index]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0:
            validation = evaluate(
                model, validation_cache, int(config["eval_batch_size"])
            )
            ratios = [
                value["candidate_mean"] / anchors[key]
                for key, value in validation.items()
                if key != "all"
            ]
            score = float(max(ratios) + 0.25 * np.mean(ratios))
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "selection_score": score,
                    "mean_ratio": float(np.mean(ratios)),
                    "worst_ratio": float(max(ratios)),
                }
            )
            if score < best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    validation = evaluate(model, validation_cache, int(config["eval_batch_size"]))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state, "config": config, "domain": name}, output / "best.pt"
    )
    machine = {}
    for key, value in validation.items():
        if key == "all":
            machine[key] = value
            continue
        np.savez_compressed(
            output / f"{key}_validation_systems.npz",
            baseline_error=value.pop("_baseline"),
            candidate_error=value.pop("_candidate"),
            uncertainty=value.pop("_uncertainty"),
        )
        machine[key] = value
    summary = {
        "domain": name,
        "parameters": parameters,
        "best_selection_score": best_score,
        "seconds": time.perf_counter() - started,
        "validation": machine,
        "history": history,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    result = {}
    for name, payload in config["domains"].items():
        result[name] = train_domain(
            name, payload, config, output / name.lower(), device
        )
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
