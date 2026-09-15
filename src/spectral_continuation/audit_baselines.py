"""Training-free and linear diagnostics for stage-1 identifiability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import (
    as_complex,
    load_ocean_npz,
    load_wavebench_development,
    spectral_carrier,
)
from .metrics import metric_vectors, paired_bootstrap


def to_two_channel(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((value.real, value.imag), dim=1).float()


def summarize(prediction: torch.Tensor, target: torch.Tensor) -> tuple[dict, dict[str, np.ndarray]]:
    vectors = metric_vectors(prediction, target)
    return {key: float(value.mean()) for key, value in vectors.items()}, vectors


def fit_global_complex_ridge(
    fields: dict[float, torch.Tensor],
    visible: tuple[float, ...],
    target_frequency: float,
    positions: np.ndarray,
    ridge: float = 1e-5,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Fit shared complex coefficients without treating coordinates as parameters."""
    count = len(visible)
    gram = torch.zeros((count, count), dtype=torch.complex128)
    rhs = torch.zeros(count, dtype=torch.complex128)
    for start in range(0, len(positions), chunk_size):
        current = positions[start : start + chunk_size]
        design = torch.stack([as_complex(fields[f][current]) for f in visible], dim=-1)
        target = as_complex(fields[target_frequency][current])
        design = design.reshape(-1, count).to(torch.complex128)
        target = target.reshape(-1).to(torch.complex128)
        gram += torch.conj(design).T @ design
        rhs += torch.conj(design).T @ target
    penalty = ridge * torch.trace(gram).real / max(count, 1)
    gram += penalty * torch.eye(count, dtype=gram.dtype)
    return torch.linalg.solve(gram, rhs).to(torch.complex64)


def apply_global_complex_ridge(
    fields: dict[float, torch.Tensor],
    visible: tuple[float, ...],
    positions: np.ndarray,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    prediction = sum(
        coefficients[index] * as_complex(fields[frequency][positions])
        for index, frequency in enumerate(visible)
    )
    return to_two_channel(prediction)


def run(args: argparse.Namespace) -> None:
    frequencies = tuple(map(float, args.frequencies.split(",")))
    conditions = [tuple(map(float, text.split(","))) for text in args.visible]
    required = sorted(set(frequencies).union(*conditions, {args.target}))
    if args.domain == "wavebench":
        bundle = load_wavebench_development(args.source_run, required, args.train_count)
    else:
        bundle = load_ocean_npz(args.dataset)
    train = np.flatnonzero(bundle.split == "train")
    validation = np.flatnonzero(bundle.split == "val")
    if args.train_count is not None and args.domain == "ocean":
        train = train[: args.train_count]
    target = bundle.fields[args.target][validation]
    result = {
        "schema_version": 1,
        "status": "development validation only",
        "domain": args.domain,
        "target_frequency_hz": args.target,
        "train_count": len(train),
        "validation_count": len(validation),
        "lineage": bundle.lineage,
        "conditions": {},
    }
    all_vectors = {}
    for visible in conditions:
        top = max(visible)
        top_complex = as_complex(bundle.fields[top][validation])
        top_amplitude = top_complex.abs()
        carrier = spectral_carrier(bundle.fields, visible, args.target, validation)
        carrier_complex = as_complex(carrier)
        predictions = {
            "copy_top": bundle.fields[top][validation],
            "amplitude_spectral_carrier": to_two_channel(top_amplitude * carrier_complex),
        }
        coefficients = fit_global_complex_ridge(bundle.fields, visible, args.target, train)
        predictions["global_complex_ridge"] = apply_global_complex_ridge(
            bundle.fields, visible, validation, coefficients
        )
        condition_key = "+".join(f"{frequency:g}" for frequency in visible)
        condition_result = {
            "visible_frequencies_hz": visible,
            "ridge_coefficients": [
                {"real": float(value.real), "imag": float(value.imag)} for value in coefficients
            ],
            "methods": {},
        }
        for method, prediction in predictions.items():
            summary, vectors = summarize(prediction, target)
            condition_result["methods"][method] = summary
            all_vectors[f"{condition_key}__{method}"] = vectors["complex_rel_l2"]
        result["conditions"][condition_key] = condition_result
    single_key = "+".join(f"{frequency:g}" for frequency in conditions[0])
    for visible in conditions[1:]:
        band_key = "+".join(f"{frequency:g}" for frequency in visible)
        for method in ("amplitude_spectral_carrier", "global_complex_ridge"):
            result.setdefault("paired_band_vs_single", {})[f"{band_key}__{method}"] = (
                paired_bootstrap(
                    all_vectors[f"{single_key}__{method}"], all_vectors[f"{band_key}__{method}"]
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("wavebench", "ocean"), required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--frequencies", default="10,15,20,40")
    parser.add_argument("--visible", action="append", required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
