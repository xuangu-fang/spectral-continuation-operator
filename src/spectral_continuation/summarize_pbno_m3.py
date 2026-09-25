"""Build paper-facing M3 tables and figures for Wave-as-Prompt."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .probabilistic_response_operator import paired_summary

BUDGETS = (1, 2, 4, 8)
LATENT_RUNS = {
    "OpenFWI": ("openfwi_long", "openfwi_seed20260926", "openfwi_seed20260927"),
    "Deepwave": ("deepwave_long", "deepwave_seed20260926", "deepwave_seed20260927"),
}
SELECTED_BRANCH = {
    "OpenFWI": "posterior_kernel",
    "Deepwave": "latent_transport",
    "Ocean": "posterior_kernel",
}


def _latent_arrays(root: Path, domain: str, split: str, budget: int) -> np.ndarray:
    return np.stack(
        [
            np.load(root / run / f"{split}_system_metrics.npz")[f"k{budget}"]
            for run in LATENT_RUNS[domain]
        ]
    ).astype(np.float64)


def _draw_method(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.5, 4.2))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")

    def box(x, y, width, height, label, color):
        axis.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.018", fc=color, ec="#333333"
            )
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)

    def arrow(x0, y0, x1, y1):
        axis.add_patch(
            FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15)
        )

    box(0.01, 0.36, 0.16, 0.30, "Sparse complex prompts\n$(q_i,y_i)$", "#DCEAF7")
    box(0.23, 0.32, 0.20, 0.38, "Stable Bayesian tokenizer\nposterior $(m_D,\\Sigma_D)$", "#E4DDF2")
    box(0.50, 0.60, 0.20, 0.25, "Neural latent transport\n$T_\\eta(m_D,\\Sigma_D)$", "#F8E2CC")
    box(0.50, 0.16, 0.20, 0.25, "Posterior-kernel attention\nover response operators", "#F7E7B6")
    box(0.78, 0.34, 0.20, 0.34, "Functional query decoder\n$\\widehat H_\\theta(s,\\omega,r)$", "#DCEFD8")
    arrow(0.17, 0.51, 0.23, 0.51)
    arrow(0.43, 0.51, 0.50, 0.72)
    arrow(0.43, 0.51, 0.50, 0.28)
    arrow(0.70, 0.72, 0.78, 0.57)
    arrow(0.70, 0.28, 0.78, 0.44)
    axis.text(
        0.5,
        0.02,
        "Linear PCA–MAP is recovered by identity transport and zero kernel blend",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout()
    figure.savefig(path / "pbno_m3_method.png", dpi=260, bbox_inches="tight")
    figure.savefig(path / "pbno_m3_method.pdf", bbox_inches="tight")
    plt.close(figure)


def run(kernel_root: Path, latent_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    json.loads((kernel_root / "summary.json").read_text(encoding="utf-8"))
    rows = []
    result = {}
    arrays_for_scatter = {}
    for domain in ("OpenFWI", "Deepwave", "Ocean"):
        domain_result = {}
        for budget in BUDGETS:
            archive = np.load(kernel_root / f"{domain.lower()}_k{budget}_test_systems.npz")
            baseline = np.asarray(archive["baseline"], dtype=np.float64)
            kernel = np.asarray(archive["candidate"], dtype=np.float64)
            latent = None
            if domain in LATENT_RUNS:
                latent = _latent_arrays(latent_root, domain, "test", budget)
            branch = SELECTED_BRANCH[domain]
            selected = kernel if branch == "posterior_kernel" else latent
            assert selected is not None
            statistics = paired_summary(
                baseline.mean(axis=0), selected.mean(axis=0), 20260925 + budget
            )
            entry = {
                "domain": domain,
                "budget": budget,
                "selected_branch": branch,
                "baseline_mean": float(baseline.mean()),
                "posterior_kernel_mean": float(kernel.mean()),
                "latent_transport_mean": (
                    float(latent.mean()) if latent is not None else ""
                ),
                **statistics,
            }
            rows.append(entry)
            domain_result[str(budget)] = entry
            if budget == 2:
                arrays_for_scatter[domain] = (
                    baseline.mean(axis=0), selected.mean(axis=0)
                )
        result[domain] = domain_result

    with (output / "main_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    belief_runs = (
        latent_root / "deepwave_belief",
        output.parent / "pbno_m1_belief_ood_formal" / "deepwave_seed20260926",
        output.parent / "pbno_m1_belief_ood_formal" / "deepwave_seed20260927",
    )
    ood_rows = []
    ood_curves = {}
    for domain in ("Deepwave", "Ocean"):
        baseline_curve = []
        candidate_curve = []
        for budget in BUDGETS:
            archive = np.load(kernel_root / f"{domain.lower()}_k{budget}_ood_systems.npz")
            baseline = np.asarray(archive["baseline"], dtype=np.float64)
            if domain == "Deepwave":
                candidate = np.stack(
                    [
                        np.load(path / "ood_system_metrics.npz")[f"k{budget}"]
                        for path in belief_runs
                    ]
                ).astype(np.float64)
                branch = "belief_transport"
            else:
                candidate = np.asarray(archive["candidate"], dtype=np.float64)
                branch = "posterior_kernel"
            statistics = paired_summary(
                baseline.mean(axis=0), candidate.mean(axis=0), 20261925 + budget
            )
            ood_rows.append(
                {
                    "domain": domain,
                    "budget": budget,
                    "selected_branch": branch,
                    **statistics,
                }
            )
            baseline_curve.append(float(baseline.mean()))
            candidate_curve.append(float(candidate.mean()))
        ood_curves[domain] = (baseline_curve, candidate_curve)
    with (output / "ood_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ood_rows[0]))
        writer.writeheader()
        writer.writerows(ood_rows)

    colors = {"baseline": "#777777", "method": "#1768AC"}
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.65))
    for axis, domain in zip(axes, result, strict=True):
        baseline = [result[domain][str(k)]["baseline_mean"] for k in BUDGETS]
        selected = [result[domain][str(k)]["candidate_mean"] for k in BUDGETS]
        axis.plot(BUDGETS, baseline, "o--", color=colors["baseline"], label="strong baseline")
        axis.plot(BUDGETS, selected, "s-", color=colors["method"], label="PBNO selected branch")
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, labels=[str(value) for value in BUDGETS])
        axis.set_title(domain)
        axis.set_xlabel("Strict-Q3 prompts K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Raw complex relative L2")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "pbno_m3_budget_curves.png", dpi=260)
    figure.savefig(output / "pbno_m3_budget_curves.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.55))
    for axis, domain in zip(axes, arrays_for_scatter, strict=True):
        baseline, selected = arrays_for_scatter[domain]
        limit = 1.03 * max(float(baseline.max()), float(selected.max()))
        axis.scatter(baseline, selected, s=10, alpha=0.45, color=colors["method"], edgecolors="none")
        axis.plot([0, limit], [0, limit], "--", color="#777777", linewidth=1)
        axis.set(xlim=(0, limit), ylim=(0, limit), title=domain, xlabel="Strong baseline")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("PBNO selected branch")
    figure.suptitle("K=2 paired held-out-system errors (below diagonal is better)", fontsize=11)
    figure.tight_layout()
    figure.savefig(output / "pbno_m3_paired_systems.png", dpi=260)
    figure.savefig(output / "pbno_m3_paired_systems.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.55))
    for axis, (domain, curves) in zip(axes, ood_curves.items(), strict=True):
        baseline, candidate = curves
        axis.plot(BUDGETS, baseline, "o--", color=colors["baseline"], label="strong baseline")
        axis.plot(BUDGETS, candidate, "s-", color="#B33A3A", label="PBNO OOD branch")
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, labels=[str(value) for value in BUDGETS])
        axis.set_title(domain)
        axis.set_xlabel("Strict-Q3 prompts K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("OOD raw complex relative L2")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "pbno_m3_ood_curves.png", dpi=260)
    figure.savefig(output / "pbno_m3_ood_curves.pdf")
    plt.close(figure)
    _draw_method(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kernel_root, args.latent_root, args.output)


if __name__ == "__main__":
    main()
