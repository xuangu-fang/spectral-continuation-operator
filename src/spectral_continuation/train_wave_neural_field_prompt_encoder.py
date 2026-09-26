"""Amortize sparse prompt inference for a frozen nonlinear wave neural field."""

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
from .train_wave_neural_field_autodecoder import (
    ComplexWaveField,
    _coordinate_features,
)


class WavePromptEncoder(nn.Module):
    """Set Transformer from irregular complex samples to a system latent."""

    def __init__(
        self,
        coordinate_dimensions: int,
        latent_dimensions: int,
        width: int,
        heads: int,
        layers: int,
    ) -> None:
        super().__init__()
        feature_dimensions = 15 * coordinate_dimensions
        self.point_encoder = nn.Sequential(
            nn.Linear(feature_dimensions + 3, width),
            nn.SiLU(),
            nn.Linear(width, width),
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
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.readout = nn.Sequential(
            nn.Linear(2 * width + 1, width),
            nn.SiLU(),
            nn.Linear(width, latent_dimensions),
        )

    def forward(
        self, value: torch.Tensor, coordinate: torch.Tensor
    ) -> torch.Tensor:
        batch, samples = value.shape
        coordinate_feature = _coordinate_features(coordinate.clone())[None].expand(
            batch, -1, -1
        )
        point = self.point_encoder(
            torch.cat(
                (
                    coordinate_feature,
                    value.real[..., None],
                    value.imag[..., None],
                    torch.log1p(value.abs())[..., None],
                ),
                dim=-1,
            )
        )
        point = self.transformer(point)
        pooled = torch.cat((point.mean(dim=1), point.amax(dim=1)), dim=-1)
        density = torch.full(
            (batch, 1),
            np.log1p(samples) / 8.0,
            dtype=pooled.dtype,
            device=pooled.device,
        )
        return self.readout(torch.cat((pooled, density), dim=-1))


def _sample_q3(
    generator: np.random.Generator,
    source_count: int,
    frequency_count: int,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(
        [generator.integers(source_count), generator.integers(frequency_count)],
        dtype=np.int64,
    )
    candidates = np.asarray(
        [
            (source, frequency)
            for source in range(source_count)
            for frequency in range(frequency_count)
            if source != target[0] and frequency != target[1]
        ],
        dtype=np.int64,
    )
    context = candidates[generator.permutation(len(candidates))[:budget]]
    return context, target


def _refine_latent(
    decoder: ComplexWaveField,
    initial: torch.Tensor,
    observed_value: torch.Tensor,
    observed_coordinate: torch.Tensor,
    steps: int,
    learning_rate: float,
    prior_strength: float,
) -> torch.Tensor:
    if steps == 0:
        return initial
    latent = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=learning_rate)
    for _ in range(steps):
        prediction = decoder(latent, observed_coordinate)
        loss = (
            (prediction - observed_value).abs().square().mean(dim=-1)
            + prior_strength * latent.square().mean(dim=-1)
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return latent.detach()


def _evaluate(
    encoder: WavePromptEncoder,
    decoder: ComplexWaveField,
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
    refinement_steps: tuple[int, ...],
    config: dict,
) -> dict:
    contexts, targets = _q3_events(
        source_count, frequency_count, budget, repeats, prompt_seed
    )
    encoded = (response - center) / scale
    result = {}
    encoder.eval()
    for fraction in fractions:
        totals = {
            steps: torch.zeros(len(response), device=response.device)
            for steps in refinement_steps
        }
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
            for start in range(0, len(response), int(config["eval_batch_size"])):
                stop = min(start + int(config["eval_batch_size"]), len(response))
                value = encoded[start:stop, observed]
                with torch.no_grad():
                    initial = encoder(value, coordinates[observed])
                for steps in refinement_steps:
                    latent = _refine_latent(
                        decoder,
                        initial,
                        value,
                        coordinates[observed],
                        steps,
                        float(config["refinement_learning_rate"]),
                        float(config["refinement_prior_strength"]),
                    )
                    with torch.no_grad():
                        prediction = center + scale * decoder(
                            latent, coordinates[target]
                        )
                        totals[steps][start:stop] += _relative_error(
                            prediction, response[start:stop, target]
                        )
        key = f"r{round(100 * fraction):03d}"
        result[key] = {
            str(steps): (total / len(contexts)).cpu().numpy().astype(np.float64)
            for steps, total in totals.items()
        }
    return result


def train(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    torch.manual_seed(int(config["seed"]))
    generator = np.random.default_rng(int(config["seed"]))
    checkpoint = torch.load(
        Path(config["decoder_checkpoint"]), map_location=device, weights_only=False
    )
    decoder_config = checkpoint["config"]
    archive = load_cross_domain_archive(Path(config["dataset"]))
    response = torch.as_tensor(
        archive.responses.reshape(len(archive.responses), -1),
        dtype=torch.complex64,
        device=device,
    )
    split = archive.split
    train_response = response[torch.as_tensor(split == "train", device=device)]
    validation_response = response[
        torch.as_tensor(split == "validation", device=device)
    ]
    test_response = response[torch.as_tensor(split == "test", device=device)]
    center = checkpoint["center"].to(device)
    scale = checkpoint["scale"].to(device)
    train_encoded = (train_response - center) / scale
    coordinates = torch.as_tensor(response_coordinates(archive), device=device)
    decoder = ComplexWaveField(
        coordinates.shape[-1],
        int(decoder_config["latent_dimensions"]),
        int(decoder_config["width"]),
        int(decoder_config["layers"]),
    ).to(device)
    decoder.load_state_dict(checkpoint["model"])
    decoder.requires_grad_(False)
    encoder = WavePromptEncoder(
        coordinates.shape[-1],
        int(decoder_config["latent_dimensions"]),
        int(config["encoder_width"]),
        int(config["encoder_heads"]),
        int(config["encoder_layers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    source_count, frequency_count, receiver_count = archive.responses.shape[1:]
    fractions = tuple(float(value) for value in config["receiver_fractions"])
    budget = int(config["budget"])
    batch_size = int(config["batch_size"])
    best_score = float("inf")
    best_state = None
    history = []
    started = time.perf_counter()
    for step in range(1, int(config["steps"]) + 1):
        context, target_pair = _sample_q3(
            generator, source_count, frequency_count, budget
        )
        fraction = fractions[int(generator.integers(len(fractions)))]
        masks = _receiver_masks(
            context, receiver_count, fraction, int(generator.integers(2**31))
        )
        observed, target = _indices(
            context,
            target_pair,
            masks,
            frequency_count,
            receiver_count,
            device,
        )
        system = torch.as_tensor(
            generator.integers(0, len(train_response), size=batch_size),
            dtype=torch.long,
            device=device,
        )
        latent = encoder(train_encoded[system][:, observed], coordinates[observed])
        normalized = decoder(latent, coordinates[target])
        prediction = center + scale * normalized
        truth = train_response[system][:, target]
        loss = _relative_error(prediction, truth).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        if step % int(config["evaluation_interval"]) == 0:
            validation = _evaluate(
                encoder,
                decoder,
                validation_response,
                coordinates,
                center,
                scale,
                source_count,
                frequency_count,
                receiver_count,
                budget,
                fractions,
                int(config["validation_repeats"]),
                int(config["selection_seed"]),
                int(config["validation_layout_seed"]),
                (0,),
                config,
            )
            means = [values["0"].mean() for values in validation.values()]
            score = float(max(means) + 0.25 * np.mean(means))
            history.append(
                {"step": step, "train_loss": float(loss.detach()), "score": score}
            )
            if score < best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in encoder.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("no checkpoint was evaluated")
    encoder.load_state_dict(best_state)
    refinement_steps = tuple(int(value) for value in config["refinement_steps"])
    validation = _evaluate(
        encoder,
        decoder,
        validation_response,
        coordinates,
        center,
        scale,
        source_count,
        frequency_count,
        receiver_count,
        budget,
        fractions,
        int(config["validation_repeats"]),
        int(config["selection_seed"]),
        int(config["validation_layout_seed"]),
        refinement_steps,
        config,
    )
    selection = {}
    for steps in refinement_steps:
        means = [value[str(steps)].mean() for value in validation.values()]
        selection[steps] = float(max(means) + 0.25 * np.mean(means))
    selected_steps = min(selection, key=selection.get)
    test = _evaluate(
        encoder,
        decoder,
        test_response,
        coordinates,
        center,
        scale,
        source_count,
        frequency_count,
        receiver_count,
        budget,
        fractions,
        int(config["test_repeats"]),
        int(config["test_seed"]),
        int(config["test_layout_seed"]),
        (selected_steps,),
        config,
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoder": best_state, "config": config, "selected_steps": selected_steps},
        output / "best.pt",
    )
    machine = {"validation": {}, "test": {}}
    for split_name, values in (("validation", validation), ("test", test)):
        for key, by_steps in values.items():
            machine[split_name][key] = {}
            for steps, errors in by_steps.items():
                np.savez_compressed(
                    output / f"{split_name}_{key}_steps{steps}_systems.npz",
                    error=errors,
                )
                machine[split_name][key][steps] = {"mean": float(errors.mean())}
    summary = {
        "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "best_validation_score": best_score,
        "selected_refinement_steps": selected_steps,
        "seconds": time.perf_counter() - started,
        "metrics": machine,
        "history": history,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
