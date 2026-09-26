"""Create paper-facing tables and figures for PBNO M4 risk resolution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BUDGETS = (1, 2, 4, 8)


def run(artifacts: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    m3 = json.loads((artifacts / "pbno_m3_summary/summary.json").read_text())
    structural = json.loads(
        (artifacts / "structural_lattice_posterior_memory_m4/summary.json").read_text()
    )["StructuralLattice"]
    compression = json.loads(
        (artifacts / "pbno_spectral_memory_compression_m4/summary.json").read_text()
    )
    noise = json.loads((artifacts / "pbno_noise_robustness_m4/summary.json").read_text())
    latency = json.loads((artifacts / "pbno_m4_summary/latency_storage.json").read_text())
    router_hard = json.loads(
        (artifacts / "pbno_prompt_router_poc_v2/summary.json").read_text()
    )
    router_risk = json.loads(
        (artifacts / "pbno_prompt_risk_router_decision_seed20260927/summary.json").read_text()
    )
    adapter_runs = {}
    for seed in (20260924, 20260925, 20260926):
        adapter_runs[seed] = json.loads(
            (artifacts / f"pbno_unified_adapter_formal_seed{seed}/summary.json").read_text()
        )

    all_results = {**m3, "StructuralLattice": {}}
    for budget in BUDGETS:
        entry = structural[str(budget)]["test"]
        all_results["StructuralLattice"][str(budget)] = {
            "baseline_mean": entry["baseline_mean"],
            "candidate_mean": entry["candidate_mean"],
            "ratio_of_means_gain": entry["ratio_of_means_gain"],
            "bootstrap_95_low": entry["bootstrap_95_low"],
            "bootstrap_95_high": entry["bootstrap_95_high"],
            "system_win_rate": entry["system_win_rate"],
            "selected_branch": "posterior_kernel",
        }

    rows = []
    for domain, values in all_results.items():
        entry = values["2"]
        rows.append(
            {
                "domain": domain,
                "budget": 2,
                "strong_baseline": entry["baseline_mean"],
                "pbno": entry["candidate_mean"],
                "relative_gain": entry["ratio_of_means_gain"],
                "bootstrap_95_low": entry["bootstrap_95_low"],
                "bootstrap_95_high": entry["bootstrap_95_high"],
                "system_win_rate": entry["system_win_rate"],
                "selected_branch": entry["selected_branch"],
            }
        )
    with (output / "m4_main_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))
    for axis, (domain, values) in zip(axes.flat, all_results.items(), strict=True):
        baseline = [values[str(budget)]["baseline_mean"] for budget in BUDGETS]
        method = [values[str(budget)]["candidate_mean"] for budget in BUDGETS]
        axis.plot(BUDGETS, baseline, "o--", color="#777777", label="strong baseline")
        axis.plot(BUDGETS, method, "s-", color="#1768AC", label="PBNO")
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
        axis.set_title("Structural lattice" if domain == "StructuralLattice" else domain)
        axis.set_xlabel("Strict-Q3 prompts K")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Raw complex relative L2")
    axes[1, 0].set_ylabel("Raw complex relative L2")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "pbno_m4_four_domain_q3.png", dpi=260)
    figure.savefig(output / "pbno_m4_four_domain_q3.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for axis, domain in zip(axes, ("OpenFWI", "Ocean"), strict=True):
        data = []
        with (artifacts / "pbno_spectral_memory_compression_m4/validation_pareto.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row["domain"] == domain and int(row["budget"]) == 2:
                    data.append(row)
        data.sort(key=lambda row: float(row["storage_fraction"]))
        x = [100.0 * float(row["storage_fraction"]) for row in data]
        y = [100.0 * float(row["relative_change"]) for row in data]
        axis.plot(x, y, "o-", color="#1768AC")
        axis.axhline(2.0, color="#B33A3A", linestyle="--", linewidth=1, label="2% error gate")
        axis.set_xscale("log", base=2)
        axis.set_title("Structural lattice" if domain == "StructuralLattice" else domain)
        axis.set_xlabel("Value-bank storage (%)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation error change (%)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "pbno_m4_storage_pareto.png", dpi=260)
    figure.savefig(output / "pbno_m4_storage_pareto.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.35))
    for axis, domain in zip(axes, noise, strict=True):
        labels = ["clean", "20 dB", "10 dB"]
        if domain in ("OpenFWI", "Ocean"):
            clean_base = m3[domain]["2"]["baseline_mean"]
            clean_method = m3[domain]["2"]["candidate_mean"]
        else:
            clean_base = structural["2"]["test"]["baseline_mean"]
            clean_method = structural["2"]["test"]["candidate_mean"]
        baseline = [clean_base, noise[domain]["20"]["baseline_mean"], noise[domain]["10"]["baseline_mean"]]
        method = [clean_method, noise[domain]["20"]["candidate_mean"], noise[domain]["10"]["candidate_mean"]]
        axis.plot(labels, baseline, "o--", color="#777777", label="global")
        axis.plot(labels, method, "s-", color="#B33A3A", label="kernel / safe fallback")
        axis.set_title(domain)
        axis.grid(alpha=0.25)
        if domain == "Ocean":
            axis.set_yscale("log")
    axes[0].set_ylabel("K=2 raw complex relative L2")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "pbno_m4_noise_robustness.png", dpi=260)
    figure.savefig(output / "pbno_m4_noise_robustness.pdf")
    plt.close(figure)

    router_rows = []
    for domain in ("OpenFWI", "Deepwave", "Ocean"):
        best = router_hard[domain]["2"]["test"]
        risk = router_risk[f"{domain.lower()}_k2"]
        router_rows.append(
            {
                "domain": domain,
                "best_branch": best["best_branch"],
                "best_branch_error": min(best[f"{name}_mean"] for name in ("global", "kernel", "neural")),
                "hard_loo_error": best["routed_mean"],
                "risk_head_error": risk["risk_head_mean"],
            }
        )
    with (output / "m4_router_no_go.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(router_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(router_rows)

    adapter_rows = []
    adapter_summary_rows = []
    router_roots = {
        seed: json.loads(
            (
                artifacts
                / f"pbno_prompt_adapter_formal_promptseed{seed}"
                / "summary.json"
            ).read_text()
        )
        for seed in adapter_runs
    }
    for domain in ("OpenFWI", "Deepwave", "Ocean"):
        for budget in BUDGETS:
            per_seed = []
            for seed, adapter in adapter_runs.items():
                entry = adapter[f"{domain.lower()}_k{budget}"]
                global_error = router_roots[seed][domain][str(budget)]["test"][
                    "global_mean"
                ]
                row = {
                    "domain": domain,
                    "budget": budget,
                    "prompt_seed": seed,
                    "best_branch": entry["best_branch"],
                    "global_error": global_error,
                    "best_branch_error": entry["best_branch_mean"],
                    "adapter_error": entry["adapter_mean"],
                    "gain_vs_global": 1.0 - entry["adapter_mean"] / global_error,
                    "gain_vs_best_branch": entry["ratio_of_means_gain"],
                    "system_win_rate_vs_best": entry["system_win_rate"],
                }
                adapter_rows.append(row)
                per_seed.append(row)
            adapter_summary_rows.append(
                {
                    "domain": domain,
                    "budget": budget,
                    "global_error_mean": float(np.mean([row["global_error"] for row in per_seed])),
                    "best_branch_error_mean": float(
                        np.mean([row["best_branch_error"] for row in per_seed])
                    ),
                    "adapter_error_mean": float(
                        np.mean([row["adapter_error"] for row in per_seed])
                    ),
                    "gain_vs_global_mean": float(
                        np.mean([row["gain_vs_global"] for row in per_seed])
                    ),
                    "gain_vs_best_mean": float(
                        np.mean([row["gain_vs_best_branch"] for row in per_seed])
                    ),
                    "gain_vs_best_min": float(
                        np.min([row["gain_vs_best_branch"] for row in per_seed])
                    ),
                    "gain_vs_best_max": float(
                        np.max([row["gain_vs_best_branch"] for row in per_seed])
                    ),
                }
            )
    for filename, table in (
        ("m4_unified_adapter_by_seed.csv", adapter_rows),
        ("m4_unified_adapter_summary.csv", adapter_summary_rows),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(table[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(table)

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), sharey=True)
    for axis, domain in zip(axes, ("OpenFWI", "Deepwave", "Ocean"), strict=True):
        domain_rows = [row for row in adapter_summary_rows if row["domain"] == domain]
        means = [-100.0 * row["gain_vs_best_mean"] for row in domain_rows]
        lower = [
            100.0 * (row["gain_vs_best_max"] - row["gain_vs_best_mean"])
            for row in domain_rows
        ]
        upper = [
            100.0 * (row["gain_vs_best_mean"] - row["gain_vs_best_min"])
            for row in domain_rows
        ]
        axis.errorbar(BUDGETS, means, yerr=(lower, upper), fmt="o-", color="#1768AC")
        axis.axhline(0.0, color="#333333", linewidth=1)
        axis.axhline(2.0, color="#B33A3A", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
        axis.set_title(domain)
        axis.set_xlabel("Prompt budget K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Error change vs best branch (%)\n(lower is better)")
    figure.tight_layout()
    figure.savefig(output / "pbno_m4_unified_adapter.png", dpi=260)
    figure.savefig(output / "pbno_m4_unified_adapter.pdf")
    plt.close(figure)

    summary = {
        "main_table": rows,
        "compression": compression,
        "latency_storage": latency,
        "noise": noise,
        "router_no_go": router_rows,
        "unified_adapter": adapter_summary_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifacts, args.output)


if __name__ == "__main__":
    main()
