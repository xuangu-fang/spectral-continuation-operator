"""Aggregate matched runs and make deterministic stage-1 figures."""

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


def variant_name(summary: dict) -> str:
    visible = summary["visible_frequencies_hz"]
    condition = "single" if len(visible) == 1 else "band"
    parameterization = "carrier" if summary["parameterization"] == "carrier_relative" else "direct"
    if summary.get("field_only"):
        condition = f"field_only_{condition}"
    return f"{condition}_{parameterization}"


def load_fixed(paths: list[Path]) -> tuple[list[dict], dict]:
    rows, vectors = [], {}
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        variant = variant_name(summary)
        row = {
            "domain": summary["domain"],
            "variant": variant,
            "seed": summary["model_seed"],
            "visible": "+".join(map(str, summary["visible_frequencies_hz"])),
            **summary["best_validation"],
        }
        rows.append(row)
        with np.load(path.parent / "validation_metrics.npz", allow_pickle=False) as archive:
            vectors[(row["domain"], variant, row["seed"])] = {
                key: np.asarray(archive[key]) for key in archive.files
            }
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["variant"])].append(row)
    aggregates = {}
    metrics = (
        "complex_rel_l2",
        "amplitude_rel_l2",
        "weighted_phase_mae_rad",
        "phase_coherence",
    )
    for (domain, variant), current in grouped.items():
        aggregates[f"{domain}/{variant}"] = {
            "seeds": [row["seed"] for row in current],
            **{
                metric: {
                    "mean": float(np.mean([row[metric] for row in current])),
                    "std": float(np.std([row[metric] for row in current])),
                }
                for metric in metrics
            },
        }
    contrasts = {}
    for domain in sorted({row["domain"] for row in rows}):
        available_seeds = sorted({row["seed"] for row in rows if row["domain"] == domain})
        for seed in available_seeds:
            candidates = {
                variant: vectors[(domain_key, variant, seed)]["complex_rel_l2"]
                for domain_key, variant, current_seed in vectors
                if domain_key == domain and current_seed == seed
            }
            for baseline, candidate, name in (
                ("single_direct", "band_direct", "band_information"),
                ("band_direct", "band_carrier", "carrier_parameterization"),
                (
                    "field_only_single_direct",
                    "field_only_band_direct",
                    "field_only_band_information",
                ),
            ):
                if baseline in candidates and candidate in candidates:
                    contrasts[f"{domain}/{name}/seed{seed}"] = paired_bootstrap(
                        candidates[baseline], candidates[candidate], seed=seed
                    )
    return rows, {"aggregates": aggregates, "paired_contrasts": contrasts}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_fixed(summary: dict, output: Path) -> None:
    aggregates = summary["aggregates"]
    domains = sorted({key.split("/")[0] for key in aggregates})
    figure, axes = plt.subplots(1, len(domains), figsize=(5.2 * len(domains), 4.0), squeeze=False)
    for axis, domain in zip(axes[0], domains, strict=True):
        variants = [
            variant
            for variant in ("single_direct", "band_direct", "single_carrier", "band_carrier")
            if f"{domain}/{variant}" in aggregates
        ]
        values = [
            aggregates[f"{domain}/{variant}"]["complex_rel_l2"]["mean"] for variant in variants
        ]
        errors = [
            aggregates[f"{domain}/{variant}"]["complex_rel_l2"]["std"] for variant in variants
        ]
        axis.bar(
            np.arange(len(variants)),
            values,
            yerr=errors,
            color=[COLORS[variant] for variant in variants],
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_xticks(np.arange(len(variants)), [name.replace("_", "\n") for name in variants])
        axis.set_ylabel("Complex relative $L_2$")
        axis.set_title(domain.capitalize())
        axis.grid(axis="y", alpha=0.22)
        for index, value in enumerate(values):
            axis.text(index, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    figure.suptitle("Sparse-band information and spectral-carrier parameterization")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args):
    paths = sorted(args.run_root.glob("**/summary.json"))
    fixed = [path for path in paths if "best_validation" in json.loads(path.read_text())]
    rows, summary = load_fixed(fixed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fixed_runs.csv", rows)
    (args.output_dir / "fixed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_fixed(summary, args.output_dir / "fixed_comparison")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
