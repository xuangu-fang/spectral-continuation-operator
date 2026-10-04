"""Plot the predeclared, conditional fixed-code frequency continuation check."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(base, rank256, sweep, output):
    config = json.loads((sweep / "config.json").read_text())
    source = json.loads((base / "config.json").read_text())
    systems = config["systems"]
    frequencies = sorted(source["query_frequencies"] + config["query_frequencies_hz"])
    variants = source["source_variants"]
    base_rows = json.loads((base / "results.json").read_text())
    rank_rows = json.loads((rank256 / "results.json").read_text())
    sweep_rows = json.loads((sweep / "results.json").read_text())
    if len(base_rows) != 20 * len(source["systems"]):
        raise ValueError("complete original confirmation required")
    if len(rank_rows) != 8 * len(source["systems"]):
        raise ValueError("complete rank256 confirmation required")
    if len(sweep_rows) != 4 * len(systems) * len(config["query_frequencies_hz"]):
        raise ValueError("complete fixed-frequency supplement required")
    if set(systems) - set(source["systems"]):
        raise ValueError("unexpected systems")
    rows = {}
    for row in base_rows:
        if row["method"] == "prior32" and row["system"] in systems:
            rows[("frozen 128D", row["system"], row["frequency"], row["source"])] = row
    for row in rank_rows:
        if row["rank"] == 256 and row["system"] in systems:
            rows[("context-fit 256D", row["system"], row["frequency"], row["source"])] = row
    for row in sweep_rows:
        method = {"original_rank128": "frozen 128D",
                  "context_rank256": "context-fit 256D"}[row["method"]]
        key = (method, row["system"], row["frequency"], row["source"])
        if key in rows:
            raise ValueError(f"duplicate result {key}")
        rows[key] = row
    expected_keys = {
        (method, system, frequency, variant)
        for method in ["frozen 128D", "context-fit 256D"]
        for system in systems for frequency in frequencies for variant in variants
    }
    if set(rows) != expected_keys:
        raise ValueError("fixed-code frequency grid is incomplete or contains extra keys")
    metrics = ["complex_rel_l2", "energy_weighted_phase_rmse_rad"]
    methods = ["frozen 128D", "context-fit 256D"]
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharex=True)
    for mi, metric in enumerate(metrics):
        for vi, variant in enumerate(variants):
            ax = axes[mi, vi]
            for method, color, marker in zip(
                methods, ["#b9624a", "#287d8e"], ["o", "s"], strict=True
            ):
                means, q25s, q75s = [], [], []
                for frequency in frequencies:
                    values = np.array([
                        rows[(method, system, frequency, variant)][metric]
                        for system in systems
                    ])
                    means.append(float(values.mean()))
                    q25s.append(float(np.quantile(values, 0.25)))
                    q75s.append(float(np.quantile(values, 0.75)))
                    records.append({
                        "frequency_hz": frequency, "source": variant,
                        "method": method, "metric": metric, "systems": len(systems),
                        "mean": means[-1], "q25": q25s[-1], "q75": q75s[-1],
                        "max": float(values.max()),
                    })
                ax.plot(frequencies, means, color=color, marker=marker, label=method)
                ax.fill_between(frequencies, q25s, q75s, color=color, alpha=0.16)
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.set_title("Broad source" if variant == "original" else "Unseen narrow 2 m source")
            if vi == 0:
                ax.set_ylabel("Raw complex RelL2" if mi == 0 else "Weighted phase RMSE (rad)")
            if mi == 1:
                ax.set_xlabel("Query frequency (Hz)")
            ax.axvline(400, color="0.6", linestyle=":", linewidth=0.8)
            ax.axvline(800, color="0.6", linestyle=":", linewidth=0.8)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Fixed low-frequency code, 60/100/140 Hz context; first 8 new spectral systems\n"
        "400/800 Hz from independent confirmation; others conditional follow-up; no target observations"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{ext}"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    at_1000 = [r for r in records if r["frequency_hz"] == 1000
               and r["method"] == "context-fit 256D"]
    if len(at_1000) != 2 * len(variants):
        raise ValueError("1000 Hz gate cannot be evaluated")
    gate = {
        "both_sources_mean_raw_at_most_0.05": all(
            r["mean"] <= 0.05 for r in at_1000 if r["metric"] == metrics[0]
        ),
        "both_sources_mean_phase_at_most_0.05_rad": all(
            r["mean"] <= 0.05 for r in at_1000 if r["metric"] == metrics[1]
        ),
        "both_sources_max_system_raw_at_most_0.10": all(
            r["max"] <= 0.10 for r in at_1000 if r["metric"] == metrics[0]
        ),
    }
    output.with_suffix(".json").write_text(json.dumps({
        "systems": systems, "frequencies_hz": frequencies,
        "subset_note": "first 8 independent-confirmation systems; not a second independent cohort",
        "base_sha256": digest(base / "results.json"),
        "rank256_sha256": digest(rank256 / "results.json"),
        "sweep_sha256": digest(sweep / "results.json"),
        "records": records, "predeclared_1000hz_gate": gate,
        "gate_passed": all(gate.values()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rank256", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.rank256, args.sweep, args.output)


if __name__ == "__main__":
    main()
