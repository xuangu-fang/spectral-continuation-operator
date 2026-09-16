"""Oracle diagnostics for spectral representability and low-band identifiability.

The diagnostics deliberately fit each validation response independently.  They are
not deployable baselines: fitting through 100 Hz is a representation oracle, while
fitting only the visible band is an identifiability probe.  Keeping those two cases
separate prevents an oracle fit from being mistaken for learned generalization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_ocean_npz


def complex_rel_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        (prediction - target).abs().square().flatten(1).sum(1)
        / target.abs().square().flatten(1).sum(1).clamp_min(1e-12)
    ).sqrt()


def phase_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    difference = torch.angle(prediction * target.conj())
    weight = target.abs()
    denominator = weight.flatten(1).sum(1).clamp_min(1e-12)
    mae = (weight * difference.abs()).flatten(1).sum(1) / denominator
    coherence = (weight * difference.cos()).flatten(1).sum(1) / denominator
    return mae, coherence


def _ridge_ar_coefficients(
    sequence: torch.Tensor, order: int, ridge: float
) -> torch.Tensor:
    """Fit a per-pixel complex recurrence to ``[pixels, frequencies]``."""

    rows = []
    for index in range(order, sequence.shape[-1]):
        rows.append(
            torch.stack(
                [sequence[:, index - lag - 1] for lag in range(order)], dim=-1
            )
        )
    design = torch.stack(rows, dim=1)
    target = sequence[:, order:]
    adjoint = design.mH
    gram = adjoint @ design
    scale = gram.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-12)
    identity = torch.eye(order, dtype=sequence.dtype, device=sequence.device)
    return torch.linalg.solve(
        gram + ridge * scale[:, None, None] * identity[None],
        adjoint @ target[..., None],
    )[..., 0]


def ar_rollout(
    observed: torch.Tensor,
    order: int,
    ridge: float,
    steps: int,
    *,
    pixel_chunk: int = 16_384,
) -> torch.Tensor:
    """Fit and roll out independent per-pixel recurrences.

    Args:
        observed: complex tensor ``[batch, frequency, height, width]``.
    Returns:
        complex tensor ``[batch, steps, height, width]``.
    """

    batch, _, height, width = observed.shape
    flat = observed.permute(0, 2, 3, 1).reshape(-1, observed.shape[1])
    output = torch.empty(flat.shape[0], steps, dtype=observed.dtype, device=observed.device)
    for start in range(0, flat.shape[0], pixel_chunk):
        current = flat[start : start + pixel_chunk]
        coefficients = _ridge_ar_coefficients(current, order, ridge)
        history = [current[:, index] for index in range(current.shape[1])]
        predictions = []
        for _ in range(steps):
            prediction = sum(
                coefficients[:, lag] * history[-lag - 1] for lag in range(order)
            )
            history.append(prediction)
            predictions.append(prediction)
        output[start : start + len(current)] = torch.stack(predictions, dim=1)
    return output.reshape(batch, height, width, steps).permute(0, 3, 1, 2)


def polynomial_carrier(
    observed: torch.Tensor,
    observed_frequencies: torch.Tensor,
    query_frequencies: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Single-component log-amplitude/unwrapped-phase polynomial oracle."""

    reference = observed_frequencies[-1]
    delta = observed_frequencies - reference
    design = torch.stack([delta**power for power in range(degree + 1)], dim=1)
    pseudo_inverse = torch.linalg.pinv(design)
    amplitude = observed.abs().clamp_min(1e-12).log()
    increments = torch.angle(observed[:, 1:] * observed[:, :-1].conj())
    phase = torch.cat(
        (torch.zeros_like(increments[:, :1]), torch.cumsum(increments, dim=1)), dim=1
    )
    log_coefficients = torch.einsum("pf,bfhw->bphw", pseudo_inverse, amplitude)
    phase_coefficients = torch.einsum("pf,bfhw->bphw", pseudo_inverse, phase)
    query_delta = query_frequencies - reference
    query_design = torch.stack(
        [query_delta**power for power in range(degree + 1)], dim=1
    )
    query_log_amplitude = torch.einsum("qp,bphw->bqhw", query_design, log_coefficients)
    query_phase = torch.einsum("qp,bphw->bqhw", query_design, phase_coefficients)
    initial_unit = observed[:, 0] / observed[:, 0].abs().clamp_min(1e-12)
    return initial_unit[:, None] * torch.exp(
        torch.complex(query_log_amplitude, query_phase)
    )


def smoothed_endpoint_carrier(
    observed: torch.Tensor, query_delta: float, kernel: int
) -> torch.Tensor:
    cross = observed[:, -1] * observed[:, -2].conj()
    real = F.avg_pool2d(cross.real[:, None], kernel, stride=1, padding=kernel // 2)[:, 0]
    imag = F.avg_pool2d(cross.imag[:, None], kernel, stride=1, padding=kernel // 2)[:, 0]
    slope = torch.angle(torch.complex(real, imag))
    unit = observed[:, -1] / observed[:, -1].abs().clamp_min(1e-12)
    return observed[:, -1].abs() * unit * torch.polar(
        torch.ones_like(slope), query_delta * slope
    )


def summarize(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    rel = complex_rel_l2(prediction, target)
    phase_mae, coherence = phase_metrics(prediction, target)
    return {
        "complex_rel_l2_mean": float(rel.mean()),
        "complex_rel_l2_std": float(rel.std()),
        "weighted_phase_mae_rad_mean": float(phase_mae.mean()),
        "phase_coherence_mean": float(coherence.mean()),
    }


def _json_safe(value):
    """Keep divergent oracle fits explicit while emitting standards-compliant JSON."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def run(args: argparse.Namespace) -> None:
    bundle = load_ocean_npz(args.dataset)
    positions = np.flatnonzero(bundle.split == "val")
    frequencies = np.asarray(bundle.frequencies)
    frequency_to_index = {float(value): index for index, value in enumerate(frequencies)}
    requested = tuple(float(value) for value in args.queries.split(","))
    query_indices = [frequency_to_index[value] for value in requested]
    all_fields = torch.stack(
        [
            torch.complex(bundle.fields[float(frequency)][:, 0], bundle.fields[float(frequency)][:, 1])
            for frequency in frequencies
        ],
        dim=1,
    )[positions]
    device = torch.device(args.device)
    metrics: dict[str, dict] = {}
    per_method_vectors: dict[str, list[float]] = {}

    def record(name: str, prediction: torch.Tensor, target: torch.Tensor) -> None:
        current = summarize(prediction, target)
        metrics[name] = current
        per_method_vectors[name] = complex_rel_l2(prediction, target).cpu().tolist()
        print(name, json.dumps(current), flush=True)

    for fit_end, label in ((80, "visible"), (100, "representation_oracle")):
        observed_indices = np.flatnonzero((frequencies >= 74) & (frequencies <= fit_end))
        first_query = fit_end + 1
        max_query = int(max(requested))
        steps = max_query - fit_end
        for order in args.orders:
            if len(observed_indices) <= 2 * order:
                continue
            for ridge in args.ridges:
                predictions = []
                targets = []
                for start in range(0, len(positions), args.batch_size):
                    observed = all_fields[start : start + args.batch_size, observed_indices].to(device)
                    rollout = ar_rollout(observed, order, ridge, steps)
                    selected = torch.tensor(
                        [int(query - first_query) for query in requested], device=device
                    )
                    predictions.append(rollout[:, selected].cpu())
                    targets.append(all_fields[start : start + args.batch_size, query_indices])
                prediction = torch.cat(predictions).flatten(0, 1)
                target = torch.cat(targets).flatten(0, 1)
                record(f"ar_{label}_order{order}_ridge{ridge:g}", prediction, target)

        for degree in (1, 2, 3):
            predictions = []
            targets = []
            observed_frequencies = torch.tensor(
                frequencies[observed_indices], dtype=torch.float32, device=device
            )
            query_tensor = torch.tensor(requested, dtype=torch.float32, device=device)
            for start in range(0, len(positions), args.batch_size):
                observed = all_fields[start : start + args.batch_size, observed_indices].to(device)
                prediction = polynomial_carrier(
                    observed, observed_frequencies, query_tensor, degree
                )
                predictions.append(prediction.cpu())
                targets.append(all_fields[start : start + args.batch_size, query_indices])
            record(
                f"polynomial_carrier_{label}_degree{degree}",
                torch.cat(predictions).flatten(0, 1),
                torch.cat(targets).flatten(0, 1),
            )

    if len(requested) == 1:
        visible_indices = np.flatnonzero((frequencies >= 74) & (frequencies <= 80))
        observed = all_fields[:, visible_indices].to(device)
        target = all_fields[:, query_indices[0]].to(device)
        for kernel in (1, 3, 5, 9, 15):
            prediction = smoothed_endpoint_carrier(observed, requested[0] - 80.0, kernel)
            record(f"endpoint_carrier_kernel{kernel}", prediction.cpu(), target.cpu())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "status": "oracle diagnostic; not a deployable model",
        "dataset": str(args.dataset),
        "validation_environments": len(positions),
        "queries_hz": requested,
        "definitions": {
            "visible": "fit only 74--80 Hz; measures low-band identifiability",
            "representation_oracle": "fit through 100 Hz on each validation environment; measures function-class capacity and is not a valid inference baseline",
        },
        "metrics": metrics,
    }
    (args.output_dir / "oracle_summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "oracle_vectors.npz",
        **{key: np.asarray(value) for key, value in per_method_vectors.items()},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", default="110")
    parser.add_argument("--orders", type=int, nargs="+", default=(1, 2, 3, 4, 6, 8))
    parser.add_argument("--ridges", type=float, nargs="+", default=(1e-4, 1e-3, 1e-2))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
