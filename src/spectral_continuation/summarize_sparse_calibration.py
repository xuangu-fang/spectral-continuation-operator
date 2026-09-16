"""Aggregate sparse spectral calibration seeds and make paper-style figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from .metrics import paired_bootstrap

METRICS = (
    "complex_rel_l2",
    "amplitude_rel_l2",
    "weighted_phase_mae_rad",
    "phase_coherence",
)


def _read_records(paths):
    records = []
    for seed, path in zip((42, 43, 44), paths, strict=True):
        with (path / "records.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                current = {
                    "seed": seed,
                    "method": row["method"],
                    "budget": int(row["budget_per_frequency"]),
                    "mask_seed": int(row["mask_seed"]),
                    "query": float(row["query_hz"]),
                }
                current.update({name: float(row[name]) for name in METRICS})
                records.append(current)
    return records


def _select(records, method, *, budget=None, query_min=None, query_max=None):
    result = [row for row in records if row["method"] == method]
    if budget is not None:
        result = [row for row in result if row["budget"] == budget]
    if query_min is not None:
        result = [row for row in result if row["query"] >= query_min]
    if query_max is not None:
        result = [row for row in result if row["query"] <= query_max]
    return result


def _run_macros(rows, metric="complex_rel_l2"):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["seed"], row["mask_seed"]), []).append(row[metric])
    return np.asarray([np.mean(values) for values in grouped.values()])


def _query_means(rows, metric="complex_rel_l2"):
    queries = sorted({row["query"] for row in rows})
    means = np.asarray(
        [np.mean([row[metric] for row in rows if row["query"] == query]) for query in queries]
    )
    return np.asarray(queries), means


def _averaged_vectors(paths, method, budget, query, mask_seeds):
    values = []
    for path in paths:
        with np.load(path / "metric_vectors.npz") as archive:
            if mask_seeds:
                current = [
                    archive[f"{method}_k{budget}_mask{mask_seed}_q{query:g}_complex_rel_l2"]
                    for mask_seed in mask_seeds
                ]
                values.append(np.mean(current, axis=0))
            else:
                values.append(archive[f"{method}_k{budget}_mask-1_q{query:g}_complex_rel_l2"])
    return np.mean(values, axis=0)


def run(args):
    paths = tuple(args.run_dirs)
    records = _read_records(paths)
    source_evaluations = []
    for seed, path in zip((42, 43, 44), paths, strict=True):
        run_summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        source_evaluations.append(
            {
                "model_seed": seed,
                "path": str(path),
                "calibrator_checkpoint_sha256": run_summary[
                    "calibrator_checkpoint_sha256"
                ],
                "base_checkpoint_sha256": run_summary["base_checkpoint_sha256"],
            }
        )
    budgets = (23, 46, 92, 184, 369)
    mask_seeds = (4201, 4301, 4401)
    near_band = (107.0, 112.0)
    full_band = (107.0, 125.0)
    baseline_near = _run_macros(
        _select(records, "hyper_pole", query_min=near_band[0], query_max=near_band[1])
    )
    oracle_near = _run_macros(
        _select(records, "dense_ar1_oracle", query_min=near_band[0], query_max=near_band[1])
    )
    budget_summary = {}
    for budget in budgets:
        near = _run_macros(
            _select(
                records,
                "amortized_ar1",
                budget=budget,
                query_min=near_band[0],
                query_max=near_band[1],
            )
        )
        full = _run_macros(
            _select(
                records,
                "amortized_ar1",
                budget=budget,
                query_min=full_band[0],
                query_max=full_band[1],
            )
        )
        budget_summary[str(budget)] = {
            "near_macro_mean": float(near.mean()),
            "near_macro_std": float(near.std(ddof=1)),
            "full_macro_mean": float(full.mean()),
            "full_macro_std": float(full.std(ddof=1)),
            "near_relative_improvement_percent": float(
                100 * (baseline_near.mean() - near.mean()) / baseline_near.mean()
            ),
            "near_oracle_gap_closure_percent": float(
                100
                * (baseline_near.mean() - near.mean())
                / (baseline_near.mean() - oracle_near.mean())
            ),
        }

    baseline_vector = _averaged_vectors(paths, "hyper_pole", 0, 110.0, ())
    candidate_vector = _averaged_vectors(paths, "amortized_ar1", 92, 110.0, mask_seeds)
    q110_paired = paired_bootstrap(baseline_vector, candidate_vector, seed=20260916)
    q110_rows = _select(records, "amortized_ar1", budget=92)
    q110_rows = [row for row in q110_rows if row["query"] == 110.0]
    q110_base = _select(records, "hyper_pole")
    q110_base = [row for row in q110_base if row["query"] == 110.0]
    q110_oracle = _select(records, "dense_ar1_oracle")
    q110_oracle = [row for row in q110_oracle if row["query"] == 110.0]
    q110_shuffled = _select(records, "shuffled_instance", budget=92)
    q110_shuffled = [row for row in q110_shuffled if row["query"] == 110.0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)

    baseline_full = _run_macros(
        _select(records, "hyper_pole", query_min=full_band[0], query_max=full_band[1])
    )
    oracle_full = _run_macros(
        _select(
            records,
            "dense_ar1_oracle",
            query_min=full_band[0],
            query_max=full_band[1],
        )
    )
    summary = {
        "schema_version": 1,
        "status": "POC complete; near-horizon positive but preregistered gate not met",
        "source_evaluations": source_evaluations,
        "protocol": {
            "visible_range_hz": [90, 100],
            "calibration_range_hz": [101, 106],
            "zero_observation_query_range_hz": [107, 125],
            "environment_count": 32,
            "model_seeds": [42, 43, 44],
            "mask_seeds": list(mask_seeds),
        },
        "near_band_107_112": {
            "hyper_pole_mean": float(baseline_near.mean()),
            "hyper_pole_std": float(baseline_near.std(ddof=1)),
            "dense_ar1_oracle_mean": float(oracle_near.mean()),
            "budget_summary": budget_summary,
        },
        "full_band_107_125": {
            "hyper_pole_mean": float(baseline_full.mean()),
            "dense_ar1_oracle_mean": float(oracle_full.mean()),
            "amortized_k92_mean": budget_summary["92"]["full_macro_mean"],
        },
        "query_110": {
            "hyper_pole_complex_rel_l2": float(
                np.mean([row["complex_rel_l2"] for row in q110_base])
            ),
            "amortized_k92_complex_rel_l2": float(
                np.mean([row["complex_rel_l2"] for row in q110_rows])
            ),
            "dense_ar1_oracle_complex_rel_l2": float(
                np.mean([row["complex_rel_l2"] for row in q110_oracle])
            ),
            "hyper_pole_phase_coherence": float(
                np.mean([row["phase_coherence"] for row in q110_base])
            ),
            "amortized_k92_phase_coherence": float(
                np.mean([row["phase_coherence"] for row in q110_rows])
            ),
            "hyper_pole_amplitude_rel_l2": float(
                np.mean([row["amplitude_rel_l2"] for row in q110_base])
            ),
            "amortized_k92_amplitude_rel_l2": float(
                np.mean([row["amplitude_rel_l2"] for row in q110_rows])
            ),
            "shuffled_k92_complex_rel_l2": float(
                np.mean([row["complex_rel_l2"] for row in q110_shuffled])
            ),
            "paired_environment_statistics": q110_paired,
        },
        "gate": {
            "required_near_relative_improvement_percent": 20.0,
            "observed_near_relative_improvement_percent": budget_summary["92"][
                "near_relative_improvement_percent"
            ],
            "required_query_110_rel_l2_below": 0.35,
            "observed_query_110_rel_l2": float(
                np.mean([row["complex_rel_l2"] for row in q110_rows])
            ),
            "passed": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    near_means = np.asarray([budget_summary[str(k)]["near_macro_mean"] for k in budgets])
    near_stds = np.asarray([budget_summary[str(k)]["near_macro_std"] for k in budgets])
    axes[0, 0].errorbar(
        budgets, near_means, yerr=near_stds, marker="o", label="Sparse AR calibration"
    )
    axes[0, 0].axhline(
        baseline_near.mean(), color="#d95f02", linestyle="--", label="LF-only hyper-pole"
    )
    axes[0, 0].axhline(
        oracle_near.mean(), color="#1b9e77", linestyle=":", label="Dense calibration oracle"
    )
    axes[0, 0].set(
        xlabel="Sensors per calibration frequency",
        ylabel="107–112 Hz macro RelL2",
        xscale="log",
        title="A. Error–budget curve",
    )
    axes[0, 0].set_xticks(budgets, [str(value) for value in budgets])
    axes[0, 0].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].legend(frameon=False, fontsize=8)

    base_q, base_curve = _query_means(_select(records, "hyper_pole"))
    oracle_q, oracle_curve = _query_means(_select(records, "dense_ar1_oracle"))
    cal_q, cal_curve = _query_means(_select(records, "amortized_ar1", budget=92))
    axes[0, 1].plot(base_q, base_curve, marker="o", label="LF-only hyper-pole")
    axes[0, 1].plot(cal_q, cal_curve, marker="o", label="Sparse AR, K=92")
    axes[0, 1].plot(oracle_q, oracle_curve, linestyle=":", label="Dense AR oracle")
    axes[0, 1].axvspan(107, 112, color="0.9", zorder=-1)
    axes[0, 1].set(
        xlabel="Unobserved query frequency (Hz)",
        ylabel="Raw complex RelL2",
        title="B. Calibration horizon",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    improvements = []
    for budget in budgets:
        _, curve = _query_means(_select(records, "amortized_ar1", budget=budget))
        improvements.append(100 * (base_curve - curve) / base_curve)
    heat = axes[1, 0].imshow(
        np.asarray(improvements),
        aspect="auto",
        origin="lower",
        extent=(base_q[0] - 0.5, base_q[-1] + 0.5, -0.5, len(budgets) - 0.5),
        cmap="RdYlBu",
        vmin=-20,
        vmax=20,
    )
    axes[1, 0].set_yticks(range(len(budgets)), budgets)
    axes[1, 0].set(
        xlabel="Unobserved query frequency (Hz)",
        ylabel="Sensors per frequency",
        title="C. Relative improvement over LF-only (%)",
    )
    figure.colorbar(heat, ax=axes[1, 0], shrink=0.8)

    closures = [budget_summary[str(k)]["near_oracle_gap_closure_percent"] for k in budgets]
    phase_rows = _select(records, "amortized_ar1", budget=92, query_min=107, query_max=112)
    phase_macro = _run_macros(phase_rows, metric="phase_coherence")
    base_phase = _run_macros(
        _select(records, "hyper_pole", query_min=107, query_max=112),
        metric="phase_coherence",
    )
    axes[1, 1].plot(budgets, closures, marker="o", color="#7570b3")
    axes[1, 1].axhline(100, color="0.5", linestyle=":")
    axes[1, 1].set(
        xlabel="Sensors per calibration frequency",
        ylabel="Oracle gap closed (%)",
        xscale="log",
        title="D. Identifiability gap closure",
    )
    axes[1, 1].set_xticks(budgets, [str(value) for value in budgets])
    axes[1, 1].xaxis.set_minor_formatter(NullFormatter())
    axes[1, 1].text(
        0.03,
        0.05,
        f"Phase coherence, K=92: {phase_macro.mean():.3f}\nLF-only: {base_phase.mean():.3f}",
        transform=axes[1, 1].transAxes,
        fontsize=8,
    )
    figure.savefig(args.output_dir / "sparse_spectral_calibration.png", dpi=220)
    figure.savefig(args.output_dir / "sparse_spectral_calibration.pdf")
    plt.close(figure)

    mechanism, mechanism_axes = plt.subplots(1, 2, figsize=(8.8, 3.25), constrained_layout=True)
    method_rows = {
        "LF-only": q110_base,
        "Sparse AR\nK=92": q110_rows,
        "Shuffled\ncontrol": q110_shuffled,
        "Dense\noracle": q110_oracle,
    }
    labels = list(method_rows)
    colors = ["#d95f02", "#7570b3", "#999999", "#1b9e77"]
    amplitude = [
        np.mean([row["amplitude_rel_l2"] for row in method_rows[label]]) for label in labels
    ]
    coherence = [
        np.mean([row["phase_coherence"] for row in method_rows[label]]) for label in labels
    ]
    mechanism_axes[0].bar(labels, amplitude, color=colors)
    mechanism_axes[0].set(
        ylabel="Amplitude RelL2",
        title="A. Amplitude at unobserved 110 Hz",
        ylim=(0, max(amplitude) * 1.18),
    )
    mechanism_axes[1].bar(labels, coherence, color=colors)
    mechanism_axes[1].set(
        ylabel="Phase coherence (higher is better)",
        title="B. Phase at unobserved 110 Hz",
        ylim=(0.75, 0.92),
    )
    for axis in mechanism_axes:
        axis.tick_params(axis="x", labelsize=8)
    mechanism.savefig(args.output_dir / "sparse_calibration_mechanism.png", dpi=220)
    mechanism.savefig(args.output_dir / "sparse_calibration_mechanism.pdf")
    plt.close(mechanism)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
