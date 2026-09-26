"""Evaluate frozen global and posterior-kernel operators with noisy prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import GlobalConfig, GlobalQ3Evaluator, fit_global_subspace
from .posterior_memory_response_operator import MemoryConfig, predict_memory_config_batched
from .probabilistic_response_operator import paired_summary


def noisy_context(
    evaluator: GlobalQ3Evaluator, snr_db: float, seed: int
) -> torch.Tensor:
    center = evaluator.center[evaluator.observed][:, None]
    scale = evaluator.scale[evaluator.observed][:, None]
    raw = center + scale * evaluator.context
    signal_rms = torch.sqrt(raw.abs().square().mean(dim=-1, keepdim=True).clamp_min(1e-20))
    generator = torch.Generator(device=evaluator.raw.device).manual_seed(seed)
    real = torch.randn(raw.shape, generator=generator, device=raw.device)
    imaginary = torch.randn(raw.shape, generator=generator, device=raw.device)
    noise = torch.complex(real, imaginary) / np.sqrt(2.0)
    noisy_raw = raw + noise * signal_rms * (10.0 ** (-snr_db / 20.0))
    return (noisy_raw - center) / scale


def system_error(prediction: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    value = torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
        truth, dim=-1
    ).clamp_min(1e-12)
    return value.mean(dim=0).cpu().numpy().astype(np.float64)


def run(config_path: Path, output: Path, device: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for domain, setting in config["domains"].items():
        archive = load_cross_domain_archive(Path(setting["dataset"]))
        operator = GlobalConfig(**setting["operator"])
        memory = MemoryConfig(**setting["memory"])
        subspace = fit_global_subspace(archive, operator.rank, operator.normalization, device)
        validation = GlobalQ3Evaluator(
            archive, subspace, "validation", int(config["budget"]),
            int(config["validation_repeats"]), int(config["selection_seed"]), device,
        )
        threshold = float(validation.latent_energy(operator).max().cpu())
        domain_result = {}
        for snr in config["snr_db"]:
            global_values = []
            memory_values = []
            for seed in config["test_seeds"]:
                evaluator = GlobalQ3Evaluator(
                    archive, subspace, "test", int(config["budget"]),
                    int(config["test_repeats"]), int(seed), device,
                )
                context = noisy_context(evaluator, float(snr), int(seed) + int(100 * snr))
                global_prediction = evaluator.predict(operator, context=context)
                memory_prediction = predict_memory_config_batched(
                    evaluator,
                    operator,
                    memory,
                    threshold if setting.get("use_latent_gate", False) else None,
                    context=context,
                )
                global_values.append(system_error(global_prediction, evaluator.truth))
                memory_values.append(system_error(memory_prediction, evaluator.truth))
            global_array = np.stack(global_values)
            memory_array = np.stack(memory_values)
            stats = paired_summary(
                global_array.mean(axis=0), memory_array.mean(axis=0),
                int(config["selection_seed"]) + int(snr),
            )
            stats.update(
                {
                    "global_seed_means": global_array.mean(axis=1).tolist(),
                    "memory_seed_means": memory_array.mean(axis=1).tolist(),
                }
            )
            domain_result[str(snr)] = stats
            np.savez_compressed(
                output / f"{domain.lower()}_snr{snr}_systems.npz",
                global_operator=global_array,
                posterior_kernel=memory_array,
            )
        result[domain] = domain_result
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
