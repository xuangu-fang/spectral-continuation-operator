"""Plot a median-improvement complex-field example from frozen checkpoints."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_ocean_npz
from .train_representation_poc import build_model, coordinates


def _summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_environment_error(paths: list[Path], frequencies: tuple[float, ...]) -> np.ndarray:
    values = []
    for path in paths:
        with np.load(path.parent / "validation_metrics.npz", allow_pickle=False) as archive:
            values.append(
                np.stack(
                    [archive[f"test_{frequency:g}_complex_rel_l2"] for frequency in frequencies]
                ).mean(axis=0)
            )
    return np.stack(values).mean(axis=0)


def _predict(summary_path: Path, fields, environment, query, environment_channels):
    summary = _summary(summary_path)
    checkpoint = torch.load(
        summary_path.parent / "best.pt", map_location="cpu", weights_only=False
    )
    model = build_model(
        Namespace(**summary["args"]),
        tuple(checkpoint["visible"]),
        environment_channels,
        checkpoint["normalization"].frequency_scale,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    context = coordinates(
        environment,
        query,
        checkpoint["normalization"].frequency_scale,
    )
    with torch.no_grad():
        prediction = model(fields, context, query)
    return prediction[0] * checkpoint["normalization"].field_scale


def run(args) -> None:
    candidate_summaries = sorted(Path().glob(args.candidate_glob))
    baseline_summaries = sorted(Path().glob(args.baseline_glob))
    if not candidate_summaries or not baseline_summaries:
        raise RuntimeError("checkpoint globs must match at least one summary")
    frequencies = tuple(map(float, args.horizon.split(",")))
    candidate_error = _mean_environment_error(candidate_summaries, frequencies)
    baseline_error = _mean_environment_error(baseline_summaries, frequencies)
    improvement = baseline_error - candidate_error
    order = np.argsort(improvement)
    validation_index = int(order[len(order) // 2])

    bundle = load_ocean_npz(args.dataset)
    validation_positions = np.flatnonzero(bundle.split == "val")
    position = int(validation_positions[validation_index])
    candidate_checkpoint = torch.load(
        candidate_summaries[0].parent / "best.pt", map_location="cpu", weights_only=False
    )
    normalization = candidate_checkpoint["normalization"]
    visible = tuple(map(float, candidate_checkpoint["visible"]))
    fields = torch.stack(
        [
            torch.complex(
                bundle.fields[frequency][position, 0],
                bundle.fields[frequency][position, 1],
            )
            for frequency in visible
        ]
    )[None] / normalization.field_scale
    environment = (
        bundle.environment[position : position + 1] - normalization.environment_mean
    ) / normalization.environment_std
    query = torch.tensor([args.query], dtype=fields.real.dtype)
    target_raw = bundle.fields[args.query][position]
    target = torch.complex(target_raw[0], target_raw[1])
    candidate = _predict(
        candidate_summaries[0], fields, environment, query, bundle.environment.shape[1]
    )
    baseline = _predict(
        baseline_summaries[0], fields, environment, query, bundle.environment.shape[1]
    )
    predictions = (target, candidate, baseline)
    errors = []
    for prediction in predictions:
        errors.append(
            float(
                torch.linalg.vector_norm(prediction - target)
                / torch.linalg.vector_norm(target).clamp_min(1e-12)
            )
        )

    with np.load(args.dataset, allow_pickle=False) as archive:
        ranges = np.asarray(archive["ranges_m"])
        depths = np.asarray(archive["depths_m"])
        environment_id = str(np.asarray(archive["environment_ids"])[position])
    extent = [float(ranges[0]), float(ranges[-1]), float(depths[-1]), float(depths[0])]
    rms = float(target.abs().square().mean().sqrt())
    real_limit = float(
        np.quantile(np.concatenate([value.real.abs().flatten().numpy() for value in predictions]), 0.995)
    )
    error_maps = [
        target.abs() / rms,
        (candidate - target).abs() / rms,
        (baseline - target).abs() / rms,
    ]
    error_limit = float(np.quantile(torch.cat([value.flatten() for value in error_maps]), 0.99))
    alpha = (target.abs() / target.abs().quantile(0.95).clamp_min(1e-12)).clamp(0.15, 1.0)

    figure, axes = plt.subplots(3, 3, figsize=(12.4, 6.6), constrained_layout=True)
    names = ("Target", args.candidate_label, args.baseline_label)
    for column, (name, value, rel_error) in enumerate(zip(names, predictions, errors, strict=True)):
        image = axes[0, column].imshow(
            value.real,
            cmap="RdBu_r",
            vmin=-real_limit,
            vmax=real_limit,
            extent=extent,
            aspect="auto",
        )
        axes[0, column].set_title(
            name if column == 0 else f"{name}  RelL2={rel_error:.3f}"
        )
        if column == 0:
            phase_value = torch.angle(value)
            phase_cmap, phase_min, phase_max = "twilight", -np.pi, np.pi
            axes[1, column].set_title("Target phase", fontsize=9)
        else:
            phase_value = torch.angle(value * torch.conj(target)).abs()
            phase_cmap, phase_min, phase_max = "viridis", 0.0, np.pi
            axes[1, column].set_title("Absolute circular phase error", fontsize=9)
        axes[1, column].imshow(
            phase_value,
            cmap=phase_cmap,
            vmin=phase_min,
            vmax=phase_max,
            alpha=alpha.numpy(),
            extent=extent,
            aspect="auto",
        )
        axes[2, column].imshow(
            error_maps[column],
            cmap="magma",
            vmin=0.0,
            vmax=error_limit,
            extent=extent,
            aspect="auto",
        )
        axes[2, column].set_title(
            "Target magnitude" if column == 0 else "Complex error magnitude",
            fontsize=9,
        )
        axes[2, column].set_xlabel("Range (m)")
    axes[0, 0].set_ylabel("Depth (m)\nReal field")
    axes[1, 0].set_ylabel("Depth (m)\nPhase")
    axes[2, 0].set_ylabel("Depth (m)\nNormalized magnitude/error")
    figure.colorbar(image, ax=axes[0], shrink=0.78, label="Real pressure")
    figure.suptitle(
        f"{args.query:g} Hz; median paired-improvement environment {environment_id}",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "selection_rule": "median validation environment after averaging seeds and all horizon queries",
        "validation_index": validation_index,
        "environment_id": environment_id,
        "query_hz": args.query,
        "candidate": args.candidate_label,
        "baseline": args.baseline_label,
        "candidate_rel_l2_first_seed": errors[1],
        "baseline_rel_l2_first_seed": errors[2],
        "paired_improvement_used_for_selection": float(improvement[validation_index]),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate-glob", required=True)
    parser.add_argument("--baseline-glob", required=True)
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--query", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
