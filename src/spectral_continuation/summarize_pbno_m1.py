"""Create paper-facing tables and figures for PBNO M0/M1."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .probabilistic_response_operator import paired_summary

BUDGETS = (1, 2, 4, 8)
NEURAL_RUNS = {
    "OpenFWI": ("openfwi_long", "openfwi_seed20260926", "openfwi_seed20260927"),
    "Deepwave": ("deepwave_long", "deepwave_seed20260926", "deepwave_seed20260927"),
}
DIRECTORIES = {"OpenFWI": "openfwi", "Deepwave": "deepwave", "Ocean": "ocean"}


def _system_columns(path: Path, split: str, budget: int) -> dict[str, np.ndarray]:
    rows = [
        row
        for row in csv.DictReader(path.open(encoding="utf-8"))
        if row["split"] == split and int(row["budget"]) == budget
    ]
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
        if key not in {"domain", "split", "budget", "system_id"}
    }


def _draw_method_figure(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 3.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.02, 0.30, 0.18, 0.42, "Sparse wave prompts\n$(s_i,\\omega_i,y_i)$", "#DCEAF7"),
        (0.27, 0.30, 0.20, 0.42, "Analytic Bayesian\nposterior $p(z_\\theta|D)$", "#E4DDF2"),
        (0.54, 0.30, 0.18, 0.42, "Whitened latent\ntransport $T_\\eta$", "#F8E2CC"),
        (0.79, 0.30, 0.19, 0.42, "Query response\n$H_\\theta(s^*,\\omega^*,r)$", "#DCEFD8"),
    ]
    for x, y, width, height, label, color in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.02", fc=color, ec="#333333"
            )
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)
    for left, right in ((0.20, 0.27), (0.47, 0.54), (0.72, 0.79)):
        axis.add_patch(
            FancyArrowPatch((left, 0.51), (right, 0.51), arrowstyle="-|>", mutation_scale=16)
        )
    axis.text(
        0.5,
        0.10,
        "Global complex response basis + prompt-only severe-shift abstention",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout()
    figure.savefig(path / "pbno_method.png", dpi=240, bbox_inches="tight")
    figure.savefig(path / "pbno_method.pdf", bbox_inches="tight")
    plt.close(figure)


def run(global_root: Path, neural_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    rows = []
    for domain, directory in DIRECTORIES.items():
        global_summary = json.loads(
            (global_root / directory / "summary.json").read_text(encoding="utf-8")
        )
        metric_path = global_root / directory / "system_metrics.csv"
        domain_result = {}
        for budget in BUDGETS:
            values = global_summary["results"][str(budget)]
            validation = values["validation"]
            pool = {
                "global_operator": validation["operator_mean"],
                "system_retrieval": validation["system_retrieval"],
                "train_mean": validation["train_mean"],
                "nearest_query": validation["nearest_query"],
            }
            strong_name = min(pool, key=pool.get)
            test_columns = _system_columns(metric_path, "test", budget)
            baseline = test_columns[strong_name]
            entry = {
                "strong_baseline": strong_name,
                "strong_baseline_mean": float(baseline.mean()),
                "global_operator_mean": float(test_columns["global_operator"].mean()),
            }
            if domain in NEURAL_RUNS:
                seed_arrays = []
                seed_means = []
                for run_name in NEURAL_RUNS[domain]:
                    archive = np.load(neural_root / run_name / "test_system_metrics.npz")
                    seed_arrays.append(np.asarray(archive[f"k{budget}"], dtype=np.float64))
                    seed_means.append(float(seed_arrays[-1].mean()))
                candidate = np.stack(seed_arrays).mean(axis=0)
                stats = paired_summary(baseline, candidate, 20261925 + budget)
                entry.update(
                    {
                        "pbno_mean": float(candidate.mean()),
                        "pbno_seed_std": float(np.std(seed_means)),
                        **stats,
                    }
                )
            else:
                entry.update(
                    {
                        "pbno_mean": float(test_columns["global_operator"].mean()),
                        "pbno_seed_std": 0.0,
                        "ratio_of_means_gain": float(
                            1.0 - test_columns["global_operator"].mean() / baseline.mean()
                        ),
                    }
                )
            if "ood" in values:
                entry.update(
                    {
                        "ood_fixed_operator": values["ood"]["operator_mean"],
                        "ood_safe_operator": values["ood"]["latent_gated_mean"],
                        "ood_operator_use_rate": values["ood"][
                            "latent_gate_operator_selection_rate"
                        ],
                    }
                )
                if domain in NEURAL_RUNS:
                    seed_ood = []
                    for run_name in NEURAL_RUNS[domain]:
                        archive = np.load(neural_root / run_name / "ood_system_metrics.npz")
                        seed_ood.append(np.asarray(archive[f"k{budget}"], dtype=np.float64))
                    entry["pbno_ood_mean"] = float(np.stack(seed_ood).mean(axis=0).mean())
            domain_result[str(budget)] = entry
            rows.append(
                {
                    "domain": domain,
                    "budget": budget,
                    **{key: value for key, value in entry.items() if not isinstance(value, dict)},
                }
            )
        result[domain] = domain_result

    with (output / "main_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for axis, domain in zip(axes, DIRECTORIES, strict=True):
        values = result[domain]
        baseline = [values[str(k)]["strong_baseline_mean"] for k in BUDGETS]
        pbno = [values[str(k)]["pbno_mean"] for k in BUDGETS]
        axis.plot(BUDGETS, baseline, "o--", label="validation-selected strong baseline")
        axis.plot(BUDGETS, pbno, "s-", label="PBNO")
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, labels=[str(k) for k in BUDGETS])
        axis.set_title(domain)
        axis.set_xlabel("Strict-Q3 prompts K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Raw complex relative L2")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "pbno_q3_main.png", dpi=240)
    figure.savefig(output / "pbno_q3_main.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    ocean = result["Ocean"]
    axis.plot(
        BUDGETS,
        [ocean[str(k)]["ood_fixed_operator"] for k in BUDGETS],
        "o-",
        label="fixed in-distribution prior",
    )
    axis.plot(
        BUDGETS,
        [ocean[str(k)]["ood_safe_operator"] for k in BUDGETS],
        "s-",
        label="prompt-only severe-shift gate",
    )
    axis.axhline(1.000858, color="#555555", linestyle="--", label="train-mean fallback")
    axis.set_yscale("log")
    axis.set_xscale("log", base=2)
    axis.set_xticks(BUDGETS, labels=[str(k) for k in BUDGETS])
    axis.set_xlabel("Strict-Q3 prompts K")
    axis.set_ylabel("Ocean OOD raw complex relative L2")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "pbno_ocean_ood_safety.png", dpi=240)
    figure.savefig(output / "pbno_ocean_ood_safety.pdf")
    plt.close(figure)
    _draw_method_figure(output)
    return result


if __name__ == "__main__":
    root = Path("artifacts")
    run(
        root / "pbno_m0_joint_query",
        root / "pbno_m1_semilinear",
        root / "pbno_m1_summary",
    )
