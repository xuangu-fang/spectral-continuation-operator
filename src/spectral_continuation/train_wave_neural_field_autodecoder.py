"""Learn a nonlinear complex neural-field manifold and infer systems from prompts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit_heterogeneous_prompt_headroom import _indices, _receiver_masks, _relative_error
from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import _q3_events
from .train_prompted_bayesian_neural_operator import response_coordinates


def _coordinate_features(coordinate: torch.Tensor) -> torch.Tensor:
    result = [coordinate]
    for scale in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        result.extend(
            (
                torch.sin(torch.pi * scale * coordinate),
                torch.cos(torch.pi * scale * coordinate),
            )
        )
    return torch.cat(result, dim=-1)


class ComplexWaveField(nn.Module):
    """Coordinate neural field modulated by a nonlinear system code."""

    def __init__(
        self,
        coordinate_dimensions: int,
        latent_dimensions: int,
        width: int,
        layers: int,
    ) -> None:
        super().__init__()
        feature_dimensions = 15 * coordinate_dimensions
        self.coordinate_stem = nn.Sequential(
            nn.Linear(feature_dimensions, width), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(layers)]
        )
        self.modulation = nn.Sequential(
            nn.Linear(latent_dimensions, width),
            nn.SiLU(),
            nn.Linear(width, 2 * layers * width),
        )
        self.output = nn.Linear(width, 2)

    def forward(self, latent: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        feature = _coordinate_features(coordinate.clone())
        hidden = self.coordinate_stem(feature)[None].expand(len(latent), -1, -1)
        modulation = self.modulation(latent).reshape(
            len(latent), len(self.blocks), 2, hidden.shape[-1]
        )
        for index, block in enumerate(self.blocks):
            gamma = 0.25 * torch.tanh(modulation[:, index, 0])
            beta = modulation[:, index, 1]
            hidden = torch.nn.functional.silu(
                block(hidden) * (1.0 + gamma[:, None]) + beta[:, None]
            )
        value = self.output(hidden)
        return torch.complex(value[..., 0], value[..., 1])


@torch.no_grad()
def _predict_chunks(
    model: ComplexWaveField,
    latent: torch.Tensor,
    coordinates: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    return torch.cat(
        [
            model(latent, coordinates[start : start + chunk])
            for start in range(0, len(coordinates), chunk)
        ],
        dim=1,
    )


def _fit_latent(
    model: ComplexWaveField,
    observed_value: torch.Tensor,
    observed_coordinate: torch.Tensor,
    latent_dimensions: int,
    steps: int,
    learning_rate: float,
    prior_strength: float,
    batch_size: int,
) -> torch.Tensor:
    result = []
    for start in range(0, len(observed_value), batch_size):
        value = observed_value[start : start + batch_size].clone()
        latent = torch.zeros(
            len(value),
            latent_dimensions,
            device=value.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([latent], lr=learning_rate)
        for _ in range(steps):
            prediction = model(latent, observed_coordinate)
            data_loss = (prediction - value).abs().square().mean(dim=-1)
            loss = (data_loss + prior_strength * latent.square().mean(dim=-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        result.append(latent.detach())
    return torch.cat(result)


@torch.inference_mode()
def _evaluate_target(
    model: ComplexWaveField,
    latent: torch.Tensor,
    target_coordinate: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    normalized = model(latent, target_coordinate)
    prediction = center + scale * normalized
    return _relative_error(prediction, truth)


def _evaluate_sparse(
    model: ComplexWaveField,
    response: torch.Tensor,
    coordinates: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    source_count: int,
    frequency_count: int,
    receiver_count: int,
    budget: int,
    fractions: tuple[float, ...],
    repeats: int,
    prompt_seed: int,
    layout_seed: int,
    config: dict,
) -> dict:
    contexts, targets = _q3_events(
        source_count, frequency_count, budget, repeats, prompt_seed
    )
    encoded = (response - center) / scale
    result = {}
    for fraction in fractions:
        total = torch.zeros(len(response), device=response.device)
        for event, (context, target_pair) in enumerate(
            zip(contexts, targets, strict=True)
        ):
            masks = _receiver_masks(
                context,
                receiver_count,
                fraction,
                layout_seed + event * 1_000_033,
            )
            observed, target = _indices(
                context,
                target_pair,
                masks,
                frequency_count,
                receiver_count,
                response.device,
            )
            latent = _fit_latent(
                model,
                encoded[:, observed],
                coordinates[observed],
                int(config["latent_dimensions"]),
                int(config["inference_steps"]),
                float(config["inference_learning_rate"]),
                float(config["inference_prior_strength"]),
                int(config["inference_batch_size"]),
            )
            total += _evaluate_target(
                model,
                latent,
                coordinates[target],
                center,
                scale,
                response[:, target],
            )
        errors = (total / len(contexts)).detach().cpu().numpy().astype(np.float64)
        result[f"r{round(100 * fraction):03d}"] = {
            "mean": float(errors.mean()),
            "systems": errors,
        }
    return result


def train_domain(
    name: str,
    payload: dict,
    config: dict,
    output: Path,
    device: torch.device,
) -> dict:
    torch.manual_seed(int(config["seed"]))
    generator = np.random.default_rng(int(config["seed"]))
    archive = load_cross_domain_archive(Path(payload["dataset"]))
    response = torch.as_tensor(
        archive.responses.reshape(len(archive.responses), -1),
        dtype=torch.complex64,
        device=device,
    )
    train_index = torch.as_tensor(archive.split == "train", device=device)
    validation_index = torch.as_tensor(archive.split == "validation", device=device)
    test_index = torch.as_tensor(archive.split == "test", device=device)
    train_raw = response[train_index]
    center = train_raw.mean()
    scale = torch.sqrt((train_raw - center).abs().square().mean()).clamp_min(1e-8)
    train = (train_raw - center) / scale
    coordinates = torch.as_tensor(response_coordinates(archive), device=device)
    model = ComplexWaveField(
        coordinates.shape[-1],
        int(config["latent_dimensions"]),
        int(config["width"]),
        int(config["layers"]),
    ).to(device)
    embeddings = nn.Embedding(len(train), int(config["latent_dimensions"])).to(device)
    nn.init.normal_(embeddings.weight, std=0.01)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(embeddings.parameters()),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    started = time.perf_counter()
    history = []
    best_loss = float("inf")
    best_state = None
    batch_systems = int(config["batch_systems"])
    batch_coordinates = int(config["batch_coordinates"])
    for step in range(1, int(config["steps"]) + 1):
        system = torch.as_tensor(
            generator.integers(0, len(train), size=batch_systems),
            dtype=torch.long,
            device=device,
        )
        coordinate_index = torch.as_tensor(
            generator.choice(
                len(coordinates), size=min(batch_coordinates, len(coordinates)), replace=False
            ),
            dtype=torch.long,
            device=device,
        )
        latent = embeddings(system)
        prediction = model(latent, coordinates[coordinate_index])
        truth = train[system][:, coordinate_index]
        data_loss = _relative_error(prediction, truth).mean()
        loss = data_loss + float(config["training_prior_strength"]) * latent.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(embeddings.parameters()), 1.0
        )
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0:
            reconstruction = []
            with torch.no_grad():
                for start in range(0, len(train), int(config["eval_batch_size"])):
                    stop = min(start + int(config["eval_batch_size"]), len(train))
                    prediction = _predict_chunks(
                        model,
                        embeddings.weight[start:stop],
                        coordinates,
                        int(config["coordinate_chunk"]),
                    )
                    reconstruction.append(
                        _relative_error(prediction, train[start:stop]).mean().item()
                    )
            reconstruction_error = float(np.mean(reconstruction))
            history.append(
                {
                    "step": step,
                    "sampled_train_loss": float(data_loss.detach()),
                    "full_train_reconstruction": reconstruction_error,
                }
            )
            if reconstruction_error < best_loss:
                best_loss = reconstruction_error
                best_state = {
                    "model": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "embeddings": embeddings.weight.detach().cpu().clone(),
                }
    if best_state is None:
        raise RuntimeError("no checkpoint was evaluated")
    model.load_state_dict(best_state["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source_count, frequency_count, receiver_count = archive.responses.shape[1:]
    fractions = tuple(float(value) for value in config["receiver_fractions"])
    validation = _evaluate_sparse(
        model,
        response[validation_index],
        coordinates,
        center,
        scale,
        source_count,
        frequency_count,
        receiver_count,
        int(payload["budget"]),
        fractions,
        int(config["evaluation_repeats"]),
        int(config["selection_seed"]),
        int(config["validation_layout_seed"]),
        config,
    )
    test = _evaluate_sparse(
        model,
        response[test_index],
        coordinates,
        center,
        scale,
        source_count,
        frequency_count,
        receiver_count,
        int(payload["budget"]),
        fractions,
        int(config["evaluation_repeats"]),
        int(config["test_seed"]),
        int(config["test_layout_seed"]),
        config,
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state["model"],
            "embeddings": best_state["embeddings"],
            "center": center.cpu(),
            "scale": scale.cpu(),
            "config": config,
        },
        output / "best.pt",
    )
    machine = {}
    for split, values in (("validation", validation), ("test", test)):
        machine[split] = {}
        for key, value in values.items():
            np.savez_compressed(output / f"{split}_{key}_systems.npz", error=value["systems"])
            machine[split][key] = {"mean": value["mean"]}
    summary = {
        "domain": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_latent_parameters": int(best_state["embeddings"].numel()),
        "best_train_reconstruction": best_loss,
        "seconds": time.perf_counter() - started,
        "metrics": machine,
        "history": history,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    result = {
        name: train_domain(name, payload, config, output / name.lower(), device)
        for name, payload in config["domains"].items()
    }
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
