"""Build the M0 reliable-horizon label atlas from frozen per-environment metrics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .horizon import (
    recovery_after_failure,
    reliable_horizon,
    safe_survival,
    seed_metric_cube,
)


def _parse_mapping(values: list[str], value_type=str) -> dict[str, object]:
    output = {}
    for item in values:
        label, value = item.split("=", 1)
        if label in output:
            raise ValueError(f"duplicate label {label!r}")
        output[label] = value_type(value)
    return output


def run(args: argparse.Namespace) -> None:
    patterns = _parse_mapping(args.group)
    references = _parse_mapping(args.reference_frequency, float)
    if set(patterns) != set(references):
        raise ValueError("group and reference-frequency labels must match")
    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    if args.primary_threshold not in thresholds:
        raise ValueError("primary threshold must be included in thresholds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "status": "M0 label atlas from frozen per-environment metric vectors",
        "metric": args.metric,
        "thresholds": thresholds,
        "primary_threshold": args.primary_threshold,
        "groups": {},
    }
    payload = {}
    rows = []

    figure, axes = plt.subplots(
        len(patterns), 3, figsize=(10.2, 3.0 * len(patterns)), squeeze=False, constrained_layout=True
    )
    for row_index, (label, pattern) in enumerate(patterns.items()):
        paths = sorted(Path(value) for value in glob.glob(str(pattern)))
        if not paths:
            raise RuntimeError(f"no metric files matched {pattern!r}")
        frequencies, cube = seed_metric_cube(paths, metric=args.metric)
        errors = cube.mean(axis=0)
        reference = float(references[label])
        current = {
            "metric_paths": [str(path) for path in paths],
            "seed_count": int(cube.shape[0]),
            "environment_count": int(cube.shape[1]),
            "query_frequencies_hz": frequencies.tolist(),
            "reference_frequency_hz": reference,
            "error_by_frequency": {},
            "thresholds": {},
        }
        for frequency_index, frequency in enumerate(frequencies):
            values = errors[:, frequency_index]
            current["error_by_frequency"][f"{frequency:g}"] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }

        payload[f"{label}_frequencies_hz"] = frequencies
        payload[f"{label}_seed_errors"] = cube
        for threshold in thresholds:
            horizon, first_unsafe, censored = reliable_horizon(
                errors, frequencies, reference, threshold
            )
            recovery = recovery_after_failure(errors, threshold)
            survival = safe_survival(errors, threshold)
            seed_horizons = np.stack(
                [
                    reliable_horizon(seed_errors, frequencies, reference, threshold)[0]
                    for seed_errors in cube
                ],
                axis=0,
            )
            key = f"{threshold:g}"
            current["thresholds"][key] = {
                "median_horizon_hz": float(np.median(horizon)),
                "mean_horizon_hz": float(horizon.mean()),
                "q25_horizon_hz": float(np.quantile(horizon, 0.25)),
                "q75_horizon_hz": float(np.quantile(horizon, 0.75)),
                "censored_fraction": float(censored.mean()),
                "recovery_after_first_failure_fraction": float(recovery.mean()),
                "mean_seed_horizon_std_hz": float(seed_horizons.std(axis=0).mean()),
                "failure_fraction_by_frequency": (errors > threshold).mean(axis=0).tolist(),
                "safe_prefix_survival_by_frequency": survival.tolist(),
            }
            payload[f"{label}_threshold_{key}_horizon_hz"] = horizon
            payload[f"{label}_threshold_{key}_first_unsafe_hz"] = first_unsafe
            payload[f"{label}_threshold_{key}_censored"] = censored
            for environment_index in range(errors.shape[0]):
                rows.append(
                    {
                        "group": label,
                        "environment_index": environment_index,
                        "threshold": threshold,
                        "horizon_hz": horizon[environment_index],
                        "first_unsafe_hz": first_unsafe[environment_index],
                        "censored": bool(censored[environment_index]),
                        "recovered_after_failure": bool(recovery[environment_index]),
                    }
                )

        summary["groups"][label] = current
        x = frequencies
        median = np.median(errors, axis=0)
        q25 = np.quantile(errors, 0.25, axis=0)
        q75 = np.quantile(errors, 0.75, axis=0)
        axes[row_index, 0].plot(x, median, marker="o", label="median error")
        axes[row_index, 0].fill_between(x, q25, q75, alpha=0.2, label="IQR")
        axes[row_index, 0].axhline(args.primary_threshold, color="tab:red", linestyle="--")
        axes[row_index, 0].set(title=f"{label}: error", ylabel=args.metric)
        axes[row_index, 0].legend(frameon=False, fontsize=8)

        for threshold in thresholds:
            axes[row_index, 1].plot(
                x, safe_survival(errors, threshold), marker="o", label=f"eta={threshold:g}"
            )
        axes[row_index, 1].set(title="Safe-prefix survival", ylim=(-0.02, 1.02))
        axes[row_index, 1].legend(frameon=False, fontsize=8)

        horizon = payload[f"{label}_threshold_{args.primary_threshold:g}_horizon_hz"]
        unique = np.unique(horizon)
        if len(unique) == 1:
            bins = np.asarray([unique[0] - 0.5, unique[0] + 0.5])
        else:
            step = np.min(np.diff(unique))
            bins = np.append(unique - step / 2, unique[-1] + step / 2)
        axes[row_index, 2].hist(horizon, bins=bins, edgecolor="white")
        axes[row_index, 2].set(title=f"Horizon distribution, eta={args.primary_threshold:g}")
        for axis in axes[row_index]:
            axis.set_xlabel("Frequency (Hz)" if axis is not axes[row_index, 2] else "Horizon (Hz)")
            axis.grid(alpha=0.22, linewidth=0.6)

    (args.output_dir / "horizon_atlas.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "horizon_labels.npz", **payload)
    with (args.output_dir / "horizon_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure.savefig(args.output_dir / "horizon_atlas.png", dpi=250)
    figure.savefig(args.output_dir / "horizon_atlas.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", action="append", required=True, help="LABEL=glob for seed metric files"
    )
    parser.add_argument(
        "--reference-frequency", action="append", required=True, help="LABEL=last observed Hz"
    )
    parser.add_argument("--metric", default="complex_rel_l2")
    parser.add_argument("--thresholds", default="0.4,0.5,0.6")
    parser.add_argument("--primary-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

