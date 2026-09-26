"""Train a coordinate-aware functional prompt reranker with an exact fallback."""

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
class RetrievalEpisode:
    fraction: float
    prompt_values: torch.Tensor
    bank_prompt_values: torch.Tensor
    prompt_coordinates: torch.Tensor
    candidate_order: torch.Tensor
    candidate_logits: torch.Tensor
    candidate_targets: torch.Tensor
    baseline_normalized: torch.Tensor
    target_center: torch.Tensor
    target_scale: torch.Tensor
    truth: torch.Tensor


def _candidate_order(
    domain: FrozenDomain,
    query: torch.Tensor,
    bank: torch.Tensor,
    count: int,
    exclude_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = min(domain.memory_dimensions, domain.rank)
    scale = torch.sqrt(domain.variance[:dimension]).clamp_min(1e-15)
    query_metric = query[:, :dimension] / scale[None]
    bank_metric = bank[:, :dimension] / scale[None]
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
    selected, order = torch.topk(
        distance, min(count, len(bank)), largest=False, dim=-1
    )
    relative = selected - selected[:, :1]
    local_scale = relative[:, -1:].clamp_min(1e-8)
    logits = -relative / local_scale
    return order, logits


class FunctionalPromptReranker(nn.Module):
    """Encode irregular complex prompts and rerank a physical-system memory."""

    def __init__(
        self,
        coordinate_dimensions: int,
        embedding_dimensions: int,
        width: int,
        heads: int,
        transformer_layers: int,
        mix_bound: float,
    ) -> None:
        super().__init__()
        coordinate_features = 11 * coordinate_dimensions
        self.mix_bound = mix_bound
        self.point_encoder = nn.Sequential(
            nn.Linear(coordinate_features + 3, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prompt_transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers
        )
        self.system_projection = nn.Sequential(
            nn.Linear(2 * width + 1, width),
            nn.SiLU(),
            nn.Linear(width, embedding_dimensions),
        )
        score_features = 4 * embedding_dimensions + 1
        self.score_head = nn.Sequential(
            nn.Linear(score_features, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(embedding_dimensions + 2, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        for module in (self.score_head[-1], self.gate_head[-1]):
            if not isinstance(module, nn.Linear):
                raise TypeError("residual heads must end in linear layers")
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(
        self, values: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        values = values.clone()
        coordinates = coordinates.clone()
        batch, _, receivers = values.shape
        coordinate = fourier_features(coordinates)[None].expand(batch, -1, -1, -1)
        point_input = torch.cat(
            (
                coordinate,
                values.real[..., None],
                values.imag[..., None],
                torch.log1p(values.abs())[..., None],
            ),
            dim=-1,
        )
        point = self.point_encoder(point_input)
        token = torch.cat((point.mean(dim=2), point.amax(dim=2)), dim=-1)
        token = token[..., : point.shape[-1]] + token[..., point.shape[-1] :]
        token = self.prompt_transformer(token)
        pooled = torch.cat((token.mean(dim=1), token.amax(dim=1)), dim=-1)
        density = torch.full(
            (batch, 1),
            np.log1p(receivers) / 8.0,
            dtype=pooled.dtype,
            device=pooled.device,
        )
        embedding = self.system_projection(torch.cat((pooled, density), dim=-1))
        return torch.nn.functional.normalize(embedding, dim=-1)

    def score_candidates(
        self,
        prompt_values: torch.Tensor,
        bank_prompt_values: torch.Tensor,
        prompt_coordinates: torch.Tensor,
        candidate_order: torch.Tensor,
        candidate_logits: torch.Tensor,
        query_bank_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bank_embedding = self.encode(bank_prompt_values, prompt_coordinates)
        query_embedding = (
            bank_embedding[query_bank_indices]
            if query_bank_indices is not None
            else self.encode(prompt_values, prompt_coordinates)
        )
        candidate_embedding = bank_embedding[candidate_order]
        query_expanded = query_embedding[:, None].expand_as(candidate_embedding)
        score_input = torch.cat(
            (
                query_expanded,
                candidate_embedding,
                query_expanded * candidate_embedding,
                (query_expanded - candidate_embedding).abs(),
                candidate_logits[..., None],
            ),
            dim=-1,
        )
        learned_logits = self.score_head(score_input).squeeze(-1)
        score = candidate_logits + learned_logits
        return score, query_embedding

    def forward(
        self,
        prompt_values: torch.Tensor,
        bank_prompt_values: torch.Tensor,
        prompt_coordinates: torch.Tensor,
        candidate_order: torch.Tensor,
        candidate_logits: torch.Tensor,
        candidate_targets: torch.Tensor,
        baseline_normalized: torch.Tensor,
        query_bank_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score, query_embedding = self.score_candidates(
            prompt_values,
            bank_prompt_values,
            prompt_coordinates,
            candidate_order,
            candidate_logits,
            query_bank_indices,
        )
        weight = torch.softmax(score, dim=-1)
        candidates = candidate_targets[candidate_order]
        retrieved = (weight[..., None] * candidates).sum(dim=1)
        entropy = -(weight * weight.clamp_min(1e-12).log()).sum(dim=-1)
        entropy = entropy / np.log(weight.shape[-1])
        gate_input = torch.cat(
            (
                query_embedding,
                entropy[:, None],
                candidate_logits.std(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        mix = self.mix_bound * torch.tanh(self.gate_head(gate_input).squeeze(-1))
        prediction = baseline_normalized + mix[:, None] * (
            retrieved - baseline_normalized
        )
        return prediction, entropy, mix


@torch.inference_mode()
def _cache_episodes(
    domain: FrozenDomain,
    split: str,
    fractions: tuple[float, ...],
    repeats: int,
    prompt_seed: int,
    layout_seed: int,
    candidates: int,
) -> dict[float, list[RetrievalEpisode]]:
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
                observed_basis, domain.train_encoded[:, observed], domain.penalty
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
            order, logits = _candidate_order(
                domain,
                query_coefficient,
                bank_coefficient,
                candidates,
                exclude,
            )
            candidate_targets = (
                domain.memory_coefficients
                @ domain.basis[target, : domain.value_rank].T
            )
            count = len(masks[0])
            result[fraction].append(
                RetrievalEpisode(
                    fraction=fraction,
                    prompt_values=encoded[:, observed].reshape(
                        len(raw), domain.budget, count
                    ),
                    bank_prompt_values=domain.train_encoded[:, observed].reshape(
                        len(domain.train_encoded), domain.budget, count
                    ),
                    prompt_coordinates=domain.coordinates[observed].reshape(
                        domain.budget, count, -1
                    ),
                    candidate_order=order,
                    candidate_logits=logits,
                    candidate_targets=candidate_targets,
                    baseline_normalized=(memory - domain.center[target][None])
                    / domain.scale[target][None],
                    target_center=domain.center[target],
                    target_scale=domain.scale[target],
                    truth=raw[:, target],
                )
            )
    return result


def _decode(normalized: torch.Tensor, episode: RetrievalEpisode) -> torch.Tensor:
    return episode.target_center.clone()[None] + episode.target_scale.clone()[None] * normalized


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.empty_like(left, dtype=np.float64)
    right_rank = np.empty_like(right, dtype=np.float64)
    left_rank[np.argsort(left)] = np.arange(len(left), dtype=np.float64)
    right_rank[np.argsort(right)] = np.arange(len(right), dtype=np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


@torch.inference_mode()
def evaluate(
    model: FunctionalPromptReranker,
    cached: dict[float, list[RetrievalEpisode]],
    batch_size: int,
) -> dict[str, dict[str, np.ndarray | float]]:
    model.eval()
    result = {}
    for fraction, episodes in cached.items():
        systems = len(episodes[0].truth)
        baseline_total = torch.zeros(systems, device=episodes[0].truth.device)
        candidate_total = torch.zeros_like(baseline_total)
        uncertainty_total = torch.zeros_like(baseline_total)
        mix_total = torch.zeros_like(baseline_total)
        for episode in episodes:
            baseline = _decode(episode.baseline_normalized, episode)
            baseline_total += _relative_error(baseline, episode.truth)
            for start in range(0, systems, batch_size):
                stop = min(start + batch_size, systems)
                normalized, uncertainty, mix = model(
                    episode.prompt_values[start:stop],
                    episode.bank_prompt_values,
                    episode.prompt_coordinates,
                    episode.candidate_order[start:stop],
                    episode.candidate_logits[start:stop],
                    episode.candidate_targets,
                    episode.baseline_normalized[start:stop],
                )
                prediction = _decode(normalized, episode)
                candidate_total[start:stop] += _relative_error(
                    prediction, episode.truth[start:stop]
                )
                uncertainty_total[start:stop] += uncertainty
                mix_total[start:stop] += mix
        baseline_error = (baseline_total / len(episodes)).cpu().numpy().astype(np.float64)
        candidate_error = (candidate_total / len(episodes)).cpu().numpy().astype(np.float64)
        uncertainty = (uncertainty_total / len(episodes)).cpu().numpy().astype(np.float64)
        mix = (mix_total / len(episodes)).cpu().numpy().astype(np.float64)
        key = f"r{round(100 * fraction):03d}"
        result[key] = {
            "baseline_mean": float(baseline_error.mean()),
            "candidate_mean": float(candidate_error.mean()),
            **paired_summary(baseline_error, candidate_error, 20260926),
            "uncertainty_error_spearman": _rank_correlation(
                uncertainty, candidate_error
            ),
            "mean_mix": float(mix.mean()),
            "_baseline": baseline_error,
            "_candidate": candidate_error,
            "_uncertainty": uncertainty,
            "_mix": mix,
        }
    return result


def _model(config: dict, device: torch.device) -> FunctionalPromptReranker:
    return FunctionalPromptReranker(
        coordinate_dimensions=5,
        embedding_dimensions=int(config["embedding_dimensions"]),
        width=int(config["width"]),
        heads=int(config["heads"]),
        transformer_layers=int(config["transformer_layers"]),
        mix_bound=float(config["mix_bound"]),
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
        int(config["candidates"]),
    )
    validation_cache = _cache_episodes(
        domain,
        "validation",
        fractions,
        int(config["validation_repeats"]),
        int(config["selection_seed"]),
        int(config["validation_layout_seed"]),
        int(config["candidates"]),
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
    anchors = {key: float(value["candidate_mean"]) for key, value in initial.items()}
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
        normalized, _, _ = model(
            episode.prompt_values[index],
            episode.bank_prompt_values,
            episode.prompt_coordinates,
            episode.candidate_order[index],
            episode.candidate_logits[index],
            episode.candidate_targets,
            episode.baseline_normalized[index],
            index,
        )
        prediction = _decode(normalized, episode)
        loss = _relative_error(prediction, episode.truth[index]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0:
            validation = evaluate(model, validation_cache, int(config["eval_batch_size"]))
            ratios = [
                value["candidate_mean"] / anchors[key]
                for key, value in validation.items()
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
    torch.save({"model": best_state, "config": config, "domain": name}, output / "best.pt")
    machine = {}
    for key, value in validation.items():
        np.savez_compressed(
            output / f"{key}_validation_systems.npz",
            baseline_error=value.pop("_baseline"),
            candidate_error=value.pop("_candidate"),
            uncertainty=value.pop("_uncertainty"),
            mix=value.pop("_mix"),
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
        result[name] = train_domain(name, payload, config, output / name.lower(), device)
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
