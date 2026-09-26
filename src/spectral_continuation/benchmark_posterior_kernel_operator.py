"""Benchmark the analytic and posterior-kernel strict-Q3 operators."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .global_bayesian_response_operator import (
    GlobalConfig,
    GlobalQ3Evaluator,
    fit_global_subspace,
)
from .posterior_memory_response_operator import (
    MemoryConfig,
    evaluate_memory_config_batched,
)


def _memory_config(name: str) -> MemoryConfig:
    parts = name.split("_")
    return MemoryConfig(
        latent_dimensions=int(parts[1][1:]),
        neighbors=int(parts[2][1:]),
        temperature=float(parts[3][1:]),
        blend=float(parts[4][1:]),
        bank_mode=(parts[5][1:] if len(parts) > 5 else "posterior"),
    )


def _timed(function, repetitions: int) -> tuple[float, float]:
    for _ in range(2):
        function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    values = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append(1000.0 * (time.perf_counter() - started))
    return float(np.median(values)), float(np.std(values))


def run(
    config_path: Path,
    summary_path: Path,
    output: Path,
    device: str,
    compression_summary_path: Path | None = None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = json.loads(summary_path.read_text(encoding="utf-8"))
    compression = (
        json.loads(compression_summary_path.read_text(encoding="utf-8"))
        if compression_summary_path is not None
        else {}
    )
    budget = 2
    result = {}
    for domain, setting in config["domains"].items():
        archive = load_cross_domain_archive(Path(setting["dataset"]))
        operator_setting = {
            **setting,
            **setting.get("operator_by_budget", {}).get(str(budget), {}),
        }
        operator = GlobalConfig(
            rank=int(operator_setting["rank"]),
            strength=float(operator_setting["strength"]),
            prior="spectrum",
            normalization=str(operator_setting["normalization"]),
        )
        subspace = fit_global_subspace(
            archive, operator.rank, operator.normalization, device
        )
        evaluator = GlobalQ3Evaluator(
            archive,
            subspace,
            "test",
            budget,
            int(config["test_repeats"]),
            int(config["test_seeds"][1]),
            device,
        )
        memory = _memory_config(selected[domain][str(budget)]["memory"])
        threshold = selected[domain][str(budget)]["latent_gate_threshold"]
        use_threshold = threshold if setting.get("use_latent_gate", False) else None
        global_latency = _timed(
            lambda evaluator=evaluator, operator=operator: evaluator.predict(operator),
            5,
        )
        memory_latency = _timed(
            lambda evaluator=evaluator,
            operator=operator,
            memory=memory,
            use_threshold=use_threshold: evaluate_memory_config_batched(
                evaluator, operator, memory, use_threshold
            ),
            5,
        )
        systems = len(evaluator.raw)
        train_systems = len(evaluator.train_raw)
        receivers = evaluator.receiver_count
        observations = budget * receivers
        rank = operator.rank
        dimension = memory.latent_dimensions
        neighbors = memory.neighbors
        # Complex multiply-add proxies for one query event.  They expose scaling
        # without pretending that hardware-specific complex FLOPs are exact.
        global_proxy = (
            observations * rank**2
            + rank**3
            + systems * observations * rank
            + systems * rank**2
            + systems * receivers * rank
        )
        memory_extra_proxy = (
            observations * rank**2
            + rank**3
            + train_systems * observations * rank
            + train_systems * rank**2
            + systems * train_systems * dimension
            + systems * neighbors * receivers
        )
        response_bytes = int(
            archive.responses[archive.split == "train"].nbytes
        )
        domain_result = {
            "budget": budget,
            "test_systems": systems,
            "train_systems": train_systems,
            "query_events": len(evaluator.observed),
            "global_latency_ms_median": global_latency[0],
            "global_latency_ms_std": global_latency[1],
            "posterior_kernel_latency_ms_median": memory_latency[0],
            "posterior_kernel_latency_ms_std": memory_latency[1],
            "latency_ratio": memory_latency[0] / global_latency[0],
            "global_complex_mac_proxy_per_event": int(global_proxy),
            "memory_extra_complex_mac_proxy_per_event": int(memory_extra_proxy),
            "memory_response_bank_bytes": response_bytes,
            "memory_response_bank_mib": response_bytes / 2**20,
            "trainable_parameters_global": 0,
            "trainable_parameters_posterior_kernel": 0,
            "trainable_parameters_pbno_transport": 343809,
            "trainable_parameters_matched_cnp": 293250,
        }
        if domain in compression:
            coefficient_rank = int(compression[domain][str(budget)]["coefficient_rank"])
            coefficients = (evaluator.train_encoded @ evaluator.basis.conj())[
                :, :coefficient_rank
            ]
            compressed_latency = _timed(
                lambda evaluator=evaluator,
                operator=operator,
                memory=memory,
                use_threshold=use_threshold,
                coefficients=coefficients: evaluate_memory_config_batched(
                    evaluator,
                    operator,
                    memory,
                    use_threshold,
                    memory_coefficients=coefficients,
                ),
                5,
            )
            coefficient_bytes = int(coefficients.numel() * coefficients.element_size())
            domain_result.update(
                {
                    "compressed_coefficient_rank": coefficient_rank,
                    "compressed_kernel_latency_ms_median": compressed_latency[0],
                    "compressed_kernel_latency_ms_std": compressed_latency[1],
                    "compressed_latency_vs_full": compressed_latency[0] / memory_latency[0],
                    "compressed_value_bank_bytes": coefficient_bytes,
                    "compressed_value_bank_mib": coefficient_bytes / 2**20,
                    "value_bank_storage_fraction": coefficient_bytes / response_bytes,
                }
            )
        result[domain] = domain_result
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compression-summary", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.summary,
                args.output,
                args.device,
                args.compression_summary,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
