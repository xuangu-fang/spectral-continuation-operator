"""Train the minimal semilinear Prompted Bayesian Neural Operator.

The model starts exactly at a validation-selected global complex PCA operator.
A coordinate MLP learns only a functional residual to its response basis, while
test-time system identification remains an analytic Bayesian posterior solve.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cross_domain_response_data import CrossDomainResponseArchive, load_cross_domain_archive
from .global_bayesian_response_operator import (
    GlobalConfig,
    _q3_events,
    fit_global_subspace,
    solve_complex_posterior,
)


def _q2_events(
    source_count: int,
    frequency_count: int,
    budget: int,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    contexts = []
    targets = []
    if budget > source_count - 1:
        raise ValueError("Q2 budget exceeds the number of non-target sources")
    for source in range(source_count):
        for frequency in range(frequency_count):
            candidates = np.asarray(
                [
                    (other_source, frequency)
                    for other_source in range(source_count)
                    if other_source != source
                ],
                dtype=np.int64,
            )
            for repeat in range(repeats):
                target_seed = seed + 1_000_003 * source + 10_007 * frequency + repeat
                order = np.random.default_rng(target_seed).permutation(len(candidates))
                contexts.append(candidates[order[:budget]])
                targets.append((source, frequency))
    return np.stack(contexts), np.asarray(targets, dtype=np.int64)


def response_coordinates(archive: CrossDomainResponseArchive) -> np.ndarray:
    """Return normalized [source_xy, log-frequency, receiver_xy] coordinates."""

    def normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        span = values.max(axis=0) - values.min(axis=0)
        return 2.0 * (values - values.min(axis=0)) / np.maximum(span, 1e-12) - 1.0

    source = normalize(archive.sources)
    receiver = normalize(archive.receivers)
    source = np.pad(source, ((0, 0), (0, max(0, 2 - source.shape[1]))))[:, :2]
    receiver = np.pad(receiver, ((0, 0), (0, max(0, 2 - receiver.shape[1]))))[:, :2]
    frequency = normalize(np.log(archive.frequencies))[:, 0]
    result = []
    for source_value in source:
        for frequency_value in frequency:
            for receiver_value in receiver:
                result.append(
                    np.concatenate((source_value, [frequency_value], receiver_value))
                )
    return np.asarray(result, dtype=np.float32)


def fourier_features(coordinates: torch.Tensor) -> torch.Tensor:
    features = [coordinates]
    for scale in (1.0, 2.0, 4.0, 8.0, 16.0):
        features.extend(
            (
                torch.sin(torch.pi * scale * coordinates),
                torch.cos(torch.pi * scale * coordinates),
            )
        )
    return torch.cat(features, dim=-1)


class PromptedBayesianNeuralOperator(nn.Module):
    """Query-conditioned complex basis with an analytic prompt posterior."""

    def __init__(
        self,
        coordinates: np.ndarray,
        base_basis: np.ndarray,
        variance: np.ndarray,
        strength: float,
        prior: str,
        hidden: int,
        use_posterior_variance: bool = False,
        receiver_count: int | None = None,
        use_prompt_set: bool = False,
        use_receiver_residual: bool = False,
        periodic_receiver: bool = False,
        residual_scale: np.ndarray | None = None,
        receiver_residual_bound: float = 0.05,
    ) -> None:
        super().__init__()
        rank = base_basis.shape[1]
        feature_count = coordinates.shape[1] * 11
        self.register_buffer("coordinates", torch.as_tensor(coordinates))
        self.register_buffer("base_basis", torch.as_tensor(base_basis))
        self.register_buffer("variance", torch.as_tensor(variance, dtype=torch.float32))
        self.prior = prior
        self.rank = rank
        self.use_posterior_variance = use_posterior_variance
        self.use_prompt_set = use_prompt_set
        self.receiver_count = receiver_count
        self.use_receiver_residual = use_receiver_residual
        self.receiver_residual_bound = receiver_residual_bound
        self.residual = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * rank),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        latent_input = (3 if use_posterior_variance else 2) * rank
        if use_prompt_set:
            if receiver_count is None:
                raise ValueError("receiver_count is required for the prompt-set updater")
            self.prompt_response = nn.Sequential(
                nn.Linear(2 * receiver_count, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.prompt_query = nn.Sequential(
                nn.Linear(33, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.prompt_fuse = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU())
            latent_input += 2 * hidden
        self.latent_residual = nn.Sequential(
            nn.Linear(latent_input, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * rank),
        )
        nn.init.zeros_(self.latent_residual[-1].weight)
        nn.init.zeros_(self.latent_residual[-1].bias)
        self.log_strength = nn.Parameter(torch.tensor(float(np.log(strength))))
        self.basis_residual_scale = 0.1 / np.sqrt(len(coordinates))
        if use_receiver_residual:
            if receiver_count is None:
                raise ValueError("receiver_count is required for receiver residual")
            if residual_scale is None:
                raise ValueError("residual_scale is required for receiver residual")
            self.register_buffer(
                "residual_scale", torch.as_tensor(residual_scale, dtype=torch.float32)
            )
            width = 64
            padding_mode = "circular" if periodic_receiver else "zeros"
            self.response_encoder = nn.Sequential(
                nn.Conv1d(2, width, 5, padding=2, padding_mode=padding_mode),
                nn.GroupNorm(8, width),
                nn.SiLU(),
                nn.Conv1d(width, width, 5, padding=2, padding_mode=padding_mode),
                nn.GroupNorm(8, width),
                nn.SiLU(),
            )
            self.baseline_encoder = nn.Sequential(
                nn.Conv1d(2, width, 5, padding=2, padding_mode=padding_mode), nn.SiLU()
            )
            self.context_metadata = nn.Sequential(
                nn.Linear(33, width), nn.SiLU(), nn.Linear(width, 2 * width)
            )
            self.target_metadata = nn.Sequential(
                nn.Linear(33, width), nn.SiLU(), nn.Linear(width, width)
            )
            self.receiver_metadata = nn.Sequential(
                nn.Linear(22, width), nn.SiLU(), nn.Linear(width, width)
            )
            self.attention_key = nn.Linear(width, width)
            self.response_decoder = nn.Sequential(
                nn.Conv1d(5 * width, 2 * width, 5, padding=2, padding_mode=padding_mode),
                nn.GroupNorm(8, 2 * width),
                nn.SiLU(),
                nn.Conv1d(2 * width, width, 5, padding=2, padding_mode=padding_mode),
                nn.GroupNorm(8, width),
                nn.SiLU(),
                nn.Conv1d(width, 2, 1),
            )
            nn.init.zeros_(self.response_decoder[-1].weight)
            nn.init.zeros_(self.response_decoder[-1].bias)

    def basis(self, indices: torch.Tensor) -> torch.Tensor:
        coordinate = self.coordinates[indices]
        value = self.residual(fourier_features(coordinate))
        real, imaginary = value.split(self.rank, dim=-1)
        residual = torch.complex(real, imaginary)
        return self.base_basis[indices] + self.basis_residual_scale * residual

    def penalty(self) -> torch.Tensor:
        strength = torch.exp(self.log_strength).clamp(1e-8, 1e2)
        if self.prior == "isotropic":
            return strength.expand(self.rank)
        if self.prior == "spectrum":
            reference = torch.quantile(self.variance, 0.5)
            return strength * reference / self.variance.clamp_min(1e-30)
        raise ValueError(f"unknown prior {self.prior!r}")

    def transport_latent(
        self,
        coefficient: torch.Tensor,
        posterior_variance: torch.Tensor,
        context: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Correct the latent chart using the full prompt belief, not only its mean."""

        standard_deviation = torch.sqrt(self.variance.clamp_min(1e-30))
        whitened = coefficient / standard_deviation[None]
        features = torch.cat((whitened.real, whitened.imag), dim=-1)
        if self.use_posterior_variance:
            uncertainty = torch.log1p(
                posterior_variance.clamp_min(0.0)
                / self.variance.clamp_min(1e-30)
            )
            features = torch.cat(
                (features, uncertainty[None].expand(len(coefficient), -1)), dim=-1
            )
        if self.use_prompt_set:
            if context is None or observed is None or self.receiver_count is None:
                raise ValueError("prompt-set updater requires context and observed indices")
            batch = len(context)
            prompt_count = context.shape[1] // self.receiver_count
            response = context.reshape(batch, prompt_count, self.receiver_count)
            response_features = torch.cat((response.real, response.imag), dim=-1)
            response_token = self.prompt_response(response_features)
            query_index = observed.reshape(prompt_count, self.receiver_count)[:, 0]
            query_coordinate = self.coordinates[query_index, :3]
            query_token = self.prompt_query(fourier_features(query_coordinate))
            token = self.prompt_fuse(response_token + query_token[None])
            mean = token.mean(dim=1)
            token_standard_deviation = torch.sqrt(
                token.var(dim=1, unbiased=False).clamp_min(0.0) + 1e-5
            )
            features = torch.cat((features, mean, token_standard_deviation), dim=-1)
        value = self.latent_residual(features)
        real, imaginary = value.split(self.rank, dim=-1)
        correction = torch.complex(real, imaginary) * standard_deviation[None]
        return coefficient + 0.1 * correction

    def correct_response(
        self,
        context: torch.Tensor,
        observed: torch.Tensor,
        target: torch.Tensor,
        baseline: torch.Tensor,
        base_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Receiver-preserving target-query residual around the Bayesian prediction."""

        if not self.use_receiver_residual:
            return baseline
        if self.receiver_count is None:
            raise ValueError("receiver_count is required")
        batch = len(context)
        prompt_count = context.shape[1] // self.receiver_count
        scale_ratio = base_scale[observed] / self.residual_scale[observed]
        response = (context * scale_ratio[None]).reshape(
            batch, prompt_count, self.receiver_count
        )
        response_channels = torch.stack((response.real, response.imag), dim=2)
        token = self.response_encoder(
            response_channels.reshape(batch * prompt_count, 2, self.receiver_count)
        ).reshape(batch, prompt_count, 64, self.receiver_count)
        query_index = observed.reshape(prompt_count, self.receiver_count)[:, 0]
        query_coordinate = self.coordinates[query_index, :3]
        scale, shift = self.context_metadata(
            fourier_features(query_coordinate)
        ).chunk(2, dim=-1)
        token = token * (1.0 + 0.1 * torch.tanh(scale[None, :, :, None]))
        token = token + shift[None, :, :, None]
        mean = token.mean(dim=1)
        deviation = torch.sqrt(
            (token.square().mean(dim=1) - mean.square()).clamp_min(0.0) + 1e-5
        )
        target_coordinate = self.coordinates[target[0], :3]
        target_token = self.target_metadata(fourier_features(target_coordinate))
        pooled = token.mean(dim=-1)
        score = torch.einsum(
            "bkh,h->bk", self.attention_key(pooled), target_token
        ) / 8.0
        attention = torch.softmax(score, dim=1)
        attended = torch.einsum("bk,bkhr->bhr", attention, token)
        baseline_in_residual_units = (
            baseline * base_scale[target][None] / self.residual_scale[target][None]
        )
        baseline_channels = torch.stack(
            (baseline_in_residual_units.real, baseline_in_residual_units.imag), dim=1
        )
        baseline_feature = self.baseline_encoder(baseline_channels)
        receiver_coordinate = self.coordinates[target, 3:]
        receiver_feature = self.receiver_metadata(
            fourier_features(receiver_coordinate)
        ).T[None].expand(batch, -1, -1)
        conditioned = receiver_feature + target_token[None, :, None]
        residual = self.response_decoder(
            torch.cat(
                (mean, deviation, attended, baseline_feature, conditioned), dim=1
            )
        )
        bounded = self.receiver_residual_bound * torch.tanh(residual)
        corrected = baseline_in_residual_units + torch.complex(
            bounded[:, 0], bounded[:, 1]
        )
        return corrected * self.residual_scale[target][None] / base_scale[target][None]


def _indices(
    context: np.ndarray,
    target: np.ndarray,
    frequency_count: int,
    receiver_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    receiver = np.arange(receiver_count, dtype=np.int64)
    context_query = context[:, 0] * frequency_count + context[:, 1]
    observed = (context_query[:, None] * receiver_count + receiver[None]).reshape(-1)
    target_query = int(target[0] * frequency_count + target[1])
    target_index = target_query * receiver_count + receiver
    return (
        torch.as_tensor(observed, dtype=torch.long, device=device),
        torch.as_tensor(target_index, dtype=torch.long, device=device),
    )


def predict(
    model: PromptedBayesianNeuralOperator,
    encoded: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    observed: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    observed_basis = model.basis(observed)
    target_basis = model.basis(target)
    prompt_values = encoded[:, observed]
    coefficient, posterior_variance = solve_complex_posterior(
        observed_basis, prompt_values, model.penalty(), return_variance=True
    )
    if posterior_variance is None:
        raise RuntimeError("posterior variance was requested but not returned")
    coefficient = model.transport_latent(
        coefficient, posterior_variance, prompt_values, observed
    )
    normalized = coefficient @ target_basis.T
    normalized = model.correct_response(
        prompt_values, observed, target, normalized, scale
    )
    return center[target][None] + scale[target][None] * normalized


@torch.inference_mode()
def evaluate(
    model: PromptedBayesianNeuralOperator,
    encoded: torch.Tensor,
    raw: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    source_count: int,
    frequency_count: int,
    receiver_count: int,
    budgets: tuple[int, ...],
    repeats: int,
    seed: int,
    regime: str = "q3",
) -> dict[int, np.ndarray]:
    model.eval()
    result = {}
    device = raw.device
    for budget in budgets:
        if regime == "q3":
            contexts, targets = _q3_events(
                source_count, frequency_count, budget, repeats, seed
            )
        elif regime == "q2":
            contexts, targets = _q2_events(
                source_count, frequency_count, budget, repeats, seed
            )
        else:
            raise ValueError(f"unknown regime {regime!r}")
        total = torch.zeros(len(raw), dtype=torch.float32, device=device)
        for context, target_pair in zip(contexts, targets, strict=True):
            observed, target = _indices(
                context, target_pair, frequency_count, receiver_count, device
            )
            prediction = predict(model, encoded, center, scale, observed, target)
            truth = raw[:, target]
            total += torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
                truth, dim=-1
            ).clamp_min(1e-12)
        result[budget] = (total / len(contexts)).cpu().numpy().astype(np.float64)
    return result


def train_domain(
    domain: str,
    dataset: Path,
    output: Path,
    config: dict,
    device_name: str,
) -> dict:
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device(device_name)
    archive = load_cross_domain_archive(dataset)
    setting = config["domains"][domain]
    base = GlobalConfig(
        rank=int(setting["rank"]),
        strength=float(setting["strength"]),
        prior=str(setting["prior"]),
        normalization=str(setting["normalization"]),
    )
    subspace = fit_global_subspace(
        archive, base.rank, base.normalization, device_name
    )
    source_count, frequency_count, receiver_count = archive.responses.shape[1:]
    train_flat = archive.responses[archive.split == "train"].reshape(
        int(np.sum(archive.split == "train")), -1
    )
    residual_scale = np.sqrt(
        np.mean(np.abs(train_flat - subspace.center[None]) ** 2, axis=0)
    ).astype(np.float32)
    positive = residual_scale[residual_scale > 0]
    residual_floor = 0.02 * float(np.median(positive)) if len(positive) else 1.0
    residual_scale = np.maximum(residual_scale, max(residual_floor, 1e-12))
    model = PromptedBayesianNeuralOperator(
        response_coordinates(archive),
        subspace.basis[:, : base.rank],
        subspace.variance[: base.rank],
        base.strength,
        base.prior,
        int(config["hidden"]),
        bool(config.get("use_posterior_variance", False)),
        receiver_count,
        bool(config.get("use_prompt_set", False)),
        bool(config.get("use_receiver_residual", False)),
        bool(setting.get("periodic_receiver", False)),
        residual_scale,
        float(config.get("receiver_residual_bound", 0.05)),
    ).to(device)
    center = torch.as_tensor(subspace.center, device=device)
    scale = torch.as_tensor(subspace.scale, device=device)
    raw_by_split = {}
    encoded_by_split = {}
    for split in ("train", "validation", "test", "ood"):
        if not np.any(archive.split == split):
            continue
        raw = torch.as_tensor(
            archive.responses[archive.split == split].reshape(
                int(np.sum(archive.split == split)), -1
            ),
            dtype=torch.complex64,
            device=device,
        )
        raw_by_split[split] = raw
        encoded_by_split[split] = (raw - center[None]) / scale[None]

    budgets = tuple(int(value) for value in config["budgets"])
    training_budgets = tuple(
        int(value) for value in config.get("training_budgets", budgets)
    )
    initial = evaluate(
        model,
        encoded_by_split["validation"],
        raw_by_split["validation"],
        center,
        scale,
        source_count,
        frequency_count,
        receiver_count,
        budgets,
        int(config["validation_repeats"]),
        int(config["seed"]),
    )
    initial_mean = {str(k): float(v.mean()) for k, v in initial.items()}
    best_score = 1.0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"step": 0, "score": best_score, **initial_mean}]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = np.random.default_rng(int(config["seed"]))
    train_raw = raw_by_split["train"]
    train_encoded = encoded_by_split["train"]
    batch_size = int(config["batch_size"])
    started = time.perf_counter()
    for step in range(1, int(config["steps"]) + 1):
        model.train()
        budget = int(generator.choice(training_budgets))
        target_pair = np.asarray(
            [generator.integers(source_count), generator.integers(frequency_count)],
            dtype=np.int64,
        )
        candidates = np.asarray(
            [
                (source, frequency)
                for source in range(source_count)
                for frequency in range(frequency_count)
                if source != target_pair[0] and frequency != target_pair[1]
            ],
            dtype=np.int64,
        )
        context = candidates[generator.permutation(len(candidates))[:budget]]
        observed, target = _indices(
            context, target_pair, frequency_count, receiver_count, device
        )
        systems = torch.as_tensor(
            generator.integers(0, len(train_raw), size=batch_size),
            dtype=torch.long,
            device=device,
        )
        prediction = predict(
            model, train_encoded[systems], center, scale, observed, target
        )
        truth = train_raw[systems][:, target]
        loss = (
            torch.linalg.vector_norm(prediction - truth, dim=-1)
            / torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0 or step == int(config["steps"]):
            validation = evaluate(
                model,
                encoded_by_split["validation"],
                raw_by_split["validation"],
                center,
                scale,
                source_count,
                frequency_count,
                receiver_count,
                budgets,
                int(config["validation_repeats"]),
                int(config["seed"]),
            )
            means = {str(k): float(v.mean()) for k, v in validation.items()}
            score = float(np.mean([means[str(k)] / initial_mean[str(k)] for k in budgets]))
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "score": score,
                    "strength": float(torch.exp(model.log_strength).detach()),
                    **means,
                }
            )
            if score < best_score:
                best_score = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    split_metrics = {}
    output.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test", "ood"):
        if split not in raw_by_split:
            continue
        values = evaluate(
            model,
            encoded_by_split[split],
            raw_by_split[split],
            center,
            scale,
            source_count,
            frequency_count,
            receiver_count,
            budgets,
            int(config["test_repeats"] if split != "validation" else config["validation_repeats"]),
            int(config["seed"]),
        )
        split_metrics[split] = {str(k): float(v.mean()) for k, v in values.items()}
        np.savez_compressed(
            output / f"{split}_system_metrics.npz",
            **{f"k{k}": value for k, value in values.items()},
        )
    checkpoint = {
        "model": model.state_dict(),
        "domain": domain,
        "base_config": asdict(base),
        "config": config,
    }
    torch.save(checkpoint, output / "best.pt")
    summary = {
        "domain": domain,
        "dataset": str(dataset.resolve()),
        "task": "strict Q3",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_config": asdict(base),
        "initial_validation": initial_mean,
        "best_validation_ratio": best_score,
        "metrics": split_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    summary = train_domain(
        args.domain,
        Path(config["domains"][args.domain]["dataset"]),
        args.output,
        config,
        args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
