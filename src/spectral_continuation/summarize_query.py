"""Aggregate continuous-query runs and plot the extrapolation horizon."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import paired_bootstrap

COLORS = {
    "single_direct": "#777777",
    "band_direct": "#3274A1",
    "single_carrier": "#E1812C",
    "band_carrier": "#3A923A",
}
LABELS = {
    "single_direct": "single / direct",
    "band_direct": "fine band / direct",
    "single_carrier": "single / carrier",
    "band_carrier": "fine band / carrier",
}


def variant(summary: dict) -> str:
    condition = "single" if len(summary["visible_frequencies_hz"]) == 1 else "band"
    method = "carrier" if summary["parameterization"] == "carrier_relative" else "direct"
    return f"{condition}_{method}"


def run(args):
    paths = []
    for path in sorted(args.run_root.glob(args.run_glob)):
        content = json.loads(path.read_text(encoding="utf-8"))
        if "extrapolation" in content:
            paths.append((path, content))
    rows, vectors, sources = [], {}, []
    for path, summary in paths:
        track = "field_only" if summary.get("field_only") else "known_medium"
        name = variant(summary)
        sources.append(
            {
                "summary": str(path),
                "track": track,
                "variant": name,
                "seed": summary["seed"],
                "selected_epoch": summary["selected_epoch"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "lineage": summary["lineage"],
            }
        )
        with np.load(path.parent / "validation_metrics.npz", allow_pickle=False) as archive:
            for regime, frequencies in (
                ("selection", summary["selection_query_frequencies_hz"]),
                ("extrapolation", summary["extrapolation_query_frequencies_hz"]),
            ):
                for frequency in frequencies:
                    frequency_key = f"{float(frequency):g}"
                    metrics = summary[regime][frequency_key]
                    row = {
                        "track": track,
                        "variant": name,
                        "seed": summary["seed"],
                        "regime": regime,
                        "frequency_hz": float(frequency),
                        **metrics,
                    }
                    rows.append(row)
                    vectors[
                        (
                            track,
                            name,
                            summary["seed"],
                            float(frequency),
                        )
                    ] = np.asarray(archive[f"{regime}_{frequency_key}_complex_rel_l2"])
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["track"], row["variant"], row["frequency_hz"])].append(row)
    aggregates = {}
    for (track, name, frequency), current in grouped.items():
        aggregates[f"{track}/{name}/{frequency:g}"] = {
            "seeds": [row["seed"] for row in current],
            "complex_rel_l2_mean": float(np.mean([row["complex_rel_l2"] for row in current])),
            "complex_rel_l2_std": float(np.std([row["complex_rel_l2"] for row in current])),
            "amplitude_rel_l2_mean": float(np.mean([row["amplitude_rel_l2"] for row in current])),
            "weighted_phase_mae_rad_mean": float(
                np.mean([row["weighted_phase_mae_rad"] for row in current])
            ),
            "phase_coherence_mean": float(np.mean([row["phase_coherence"] for row in current])),
        }
    contrasts = {}
    tracks = sorted({row["track"] for row in rows})
    frequencies = sorted({row["frequency_hz"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    for track in tracks:
        for frequency in frequencies:
            for seed in seeds:
                for base, candidate, name in (
                    ("single_direct", "band_direct", "band_information"),
                    ("band_direct", "band_carrier", "carrier_parameterization"),
                    ("single_direct", "band_carrier", "full_method_vs_strong_single"),
                ):
                    base_key = (track, base, seed, frequency)
                    candidate_key = (track, candidate, seed, frequency)
                    if base_key in vectors and candidate_key in vectors:
                        contrasts[f"{track}/{name}/{frequency:g}/seed{seed}"] = paired_bootstrap(
                            vectors[base_key], vectors[candidate_key], seed=seed
                        )
    seed_aggregate_contrasts = {}
    for track in tracks:
        for frequency in frequencies:
            for base, candidate, name in (
                ("single_direct", "band_direct", "band_information"),
                ("band_direct", "band_carrier", "carrier_parameterization"),
                ("single_direct", "band_carrier", "full_method_vs_strong_single"),
            ):
                improvements = []
                for seed in seeds:
                    base_key = (track, base, seed, frequency)
                    candidate_key = (track, candidate, seed, frequency)
                    if base_key in vectors and candidate_key in vectors:
                        baseline = float(np.mean(vectors[base_key]))
                        candidate_value = float(np.mean(vectors[candidate_key]))
                        improvements.append(100 * (baseline - candidate_value) / baseline)
                if improvements:
                    seed_aggregate_contrasts[f"{track}/{name}/{frequency:g}"] = {
                        "seeds": len(improvements),
                        "relative_improvement_percent_mean": float(np.mean(improvements)),
                        "relative_improvement_percent_std": float(np.std(improvements)),
                        "fraction_seeds_positive": float(np.mean(np.asarray(improvements) > 0)),
                        "per_seed_percent": improvements,
                    }
    output = {
        "schema_version": 1,
        "sources": sources,
        "aggregates": aggregates,
        "paired_contrasts": contrasts,
        "seed_aggregate_contrasts": seed_aggregate_contrasts,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "query_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "query_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    if not rows:
        return
    figure, axes = plt.subplots(1, len(tracks), figsize=(5.4 * len(tracks), 4.2), squeeze=False)
    for axis, track in zip(axes[0], tracks, strict=True):
        for name, color in COLORS.items():
            active_frequencies = [
                frequency
                for frequency in frequencies
                if f"{track}/{name}/{frequency:g}" in aggregates
            ]
            if not active_frequencies:
                continue
            means = [
                aggregates[f"{track}/{name}/{frequency:g}"]["complex_rel_l2_mean"]
                for frequency in active_frequencies
            ]
            stds = [
                aggregates[f"{track}/{name}/{frequency:g}"]["complex_rel_l2_std"]
                for frequency in active_frequencies
            ]
            axis.errorbar(
                active_frequencies,
                means,
                yerr=stds,
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=color,
                label=LABELS[name],
            )
        axis.axvspan(min(frequencies), args.selection_max_hz, color="#EEEEEE", alpha=0.7)
        axis.axvline(args.selection_max_hz, color="#444444", linestyle="--", linewidth=1)
        axis.text(
            0.015,
            0.98,
            "fit: 90/100 Hz",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        axis.text(
            0.34,
            0.98,
            "label-blind extrapolation",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        axis.set_title(track.replace("_", " ").title())
        axis.set_xlabel("Query frequency (Hz)")
        axis.set_ylabel("Complex relative $L_2$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Spectral continuation beyond the selected frequency range")
    figure.tight_layout()
    figure.savefig(args.output_dir / "extrapolation_horizon.png", dpi=260, bbox_inches="tight")
    figure.savefig(args.output_dir / "extrapolation_horizon.pdf", bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-glob", default="**/summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-max-hz", type=float, default=100.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
