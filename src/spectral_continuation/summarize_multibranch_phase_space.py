"""Summarize the gate-driven multibranch phase-space POC."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEEDS = (20260922, 20260923, 20260924)
BOUNDARIES = ("open", "partial", "closed")
SYNTHETIC_BUDGETS = ("s1_f1", "s2_f1", "s4_f1", "s6_f1", "s6_f4")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _only(paths: Iterable[Path], description: str) -> Path:
    selected = list(paths)
    if len(selected) != 1:
        raise RuntimeError(f"expected one {description}, found {selected}")
    return selected[0]


def _read_errors(
    path: Path, *, error_column: str = "contrast_rel_l2"
) -> dict[int, float]:
    with path.open(newline="") as handle:
        return {
            int(row["system"]): float(row[error_column])
            for row in csv.DictReader(handle)
        }


def _bootstrap_mean(
    values: np.ndarray, *, seed: int, draws: int = 20_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    replicates = values[indices].mean(axis=1)
    low, high = np.quantile(replicates, (0.025, 0.975))
    return float(low), float(high)


def _synthetic_rows(root: Path) -> list[dict]:
    rows = []
    for seed in SEEDS:
        control = _load_json(
            _only(
                (root / f"synthetic_continuation_control_seed{seed}").glob("*.json"),
                f"synthetic control JSON for seed {seed}",
            )
        )
        candidate = _load_json(
            _only(
                (root / f"synthetic_residual_multiscale_gram_seed{seed}").glob(
                    "*.json"
                ),
                f"synthetic candidate JSON for seed {seed}",
            )
        )
        for budget in SYNTHETIC_BUDGETS:
            control_error = control["low_group_budgets"][budget][
                "contrast_rel_l2_mean"
            ]
            candidate_error = candidate["low_group_budgets"][budget][
                "contrast_rel_l2_mean"
            ]
            rows.append(
                {
                    "seed": seed,
                    "budget": budget,
                    "control_rel_l2": control_error,
                    "candidate_rel_l2": candidate_error,
                    "relative_improvement": (control_error - candidate_error)
                    / control_error,
                }
            )
    return rows


def _boundary_rows(root: Path) -> tuple[list[dict], list[dict]]:
    raw_rows = []
    errors: dict[str, dict[int, dict[int, float]]] = {
        "control": {},
        "candidate": {},
    }
    eval_root = root / "boundary_eval"
    for seed in SEEDS:
        control_path = _only(
            eval_root.glob(f"*control*seed{seed}_records.csv"),
            f"boundary control records for seed {seed}",
        )
        candidate_path = _only(
            eval_root.glob(f"*multiscale*seed{seed}_records.csv"),
            f"boundary candidate records for seed {seed}",
        )
        errors["control"][seed] = _read_errors(control_path)
        errors["candidate"][seed] = _read_errors(candidate_path)
        if errors["control"][seed].keys() != errors["candidate"][seed].keys():
            raise RuntimeError(f"unpaired boundary systems for seed {seed}")
        for system in sorted(errors["control"][seed]):
            control = errors["control"][seed][system]
            candidate = errors["candidate"][seed][system]
            raw_rows.append(
                {
                    "seed": seed,
                    "system": system,
                    "boundary": BOUNDARIES[system % len(BOUNDARIES)],
                    "control_rel_l2": control,
                    "candidate_rel_l2": candidate,
                    "relative_improvement": (control - candidate) / control,
                }
            )

    # The inferential unit is the physical system, not a seed-system pair.
    # Average predictions over seeds first, then bootstrap systems.
    systems = sorted(errors["control"][SEEDS[0]])
    paired_rows = []
    for system in systems:
        control = float(np.mean([errors["control"][s][system] for s in SEEDS]))
        candidate = float(np.mean([errors["candidate"][s][system] for s in SEEDS]))
        paired_rows.append(
            {
                "system": system,
                "boundary": BOUNDARIES[system % len(BOUNDARIES)],
                "control_rel_l2": control,
                "candidate_rel_l2": candidate,
                "relative_improvement": (control - candidate) / control,
            }
        )
    return raw_rows, paired_rows


def _aggregate_boundary(paired_rows: list[dict]) -> list[dict]:
    summary = []
    for offset, boundary in enumerate((*BOUNDARIES, "all")):
        selected = [
            row
            for row in paired_rows
            if boundary == "all" or row["boundary"] == boundary
        ]
        control = np.asarray([row["control_rel_l2"] for row in selected])
        candidate = np.asarray([row["candidate_rel_l2"] for row in selected])
        improvement = np.asarray([row["relative_improvement"] for row in selected])
        ci_low, ci_high = _bootstrap_mean(improvement, seed=20260923 + offset)
        summary.append(
            {
                "boundary": boundary,
                "systems": len(selected),
                "control_rel_l2": float(control.mean()),
                "candidate_rel_l2": float(candidate.mean()),
                "ratio_of_means_improvement": float(
                    (control.mean() - candidate.mean()) / control.mean()
                ),
                "paired_relative_improvement": float(improvement.mean()),
                "paired_improvement_95ci": [ci_low, ci_high],
                "fraction_systems_improved": float(np.mean(candidate < control)),
            }
        )
    return summary


def _wavebench_summary(root: Path, phase_root: Path) -> dict:
    control = _load_json(
        phase_root / "amplitude_soft_phase_gradient_seed20260923.json"
    )
    packet = _load_json(
        root
        / "residual_wave_packets_seed20260923"
        / "amplitude_phase_wave_packets_seed20260923.json"
    )
    gram = _load_json(
        root
        / "residual_wave_packet_gram_seed20260923"
        / "amplitude_phase_wave_packet_gram_seed20260923.json"
    )
    budgets = {}
    for budget in ("1", "2", "3", "4"):
        values = {
            "control": control["frequency_budget"][budget]["contrast_rel_l2_mean"],
            "independent_packet_phase": packet["frequency_budget"][budget][
                "contrast_rel_l2_mean"
            ],
            "relative_phase_gram": gram["frequency_budget"][budget][
                "contrast_rel_l2_mean"
            ],
        }
        values["gram_improvement_vs_control"] = (
            values["control"] - values["relative_phase_gram"]
        ) / values["control"]
        budgets[budget] = values

    control_records = _read_errors(
        phase_root / "amplitude_soft_phase_gradient_seed20260923_records.csv",
        error_column="full_context_contrast_rel_l2",
    )
    gram_records = _read_errors(
        root
        / "residual_wave_packet_gram_seed20260923"
        / "amplitude_phase_wave_packet_gram_seed20260923_records.csv",
        error_column="full_context_contrast_rel_l2",
    )
    systems = sorted(control_records.keys() & gram_records.keys())
    improvement = np.asarray(
        [
            (control_records[system] - gram_records[system]) / control_records[system]
            for system in systems
        ]
    )
    ci_low, ci_high = _bootstrap_mean(improvement, seed=20260927)
    return {
        "status": "single-seed mechanism ablation; not a formal cross-seed result",
        "budgets": budgets,
        "full_context_paired_relative_improvement": float(improvement.mean()),
        "full_context_paired_improvement_95ci": [ci_low, ci_high],
        "full_context_fraction_systems_improved": float(np.mean(improvement > 0)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    oracle: dict,
    synthetic_rows: list[dict],
    boundary_summary: list[dict],
    wavebench: dict,
    output: Path,
) -> None:
    colors = {"control": "#6B7280", "candidate": "#0F766E"}
    figure, axes = plt.subplots(2, 2, figsize=(11.3, 7.6), constrained_layout=True)

    labels = ("iid", "open", "partial", "closed")
    x = np.arange(len(labels))
    for branches, color in ((1, colors["control"]), (4, colors["candidate"])):
        axes[0, 0].plot(
            x,
            [oracle["adaptive"][label][f"mean_error_k{branches}"] for label in labels],
            "o-",
            color=color,
            label=f"K={branches}",
        )
    axes[0, 0].set_xticks(x, ("IID", "Open", "Partial", "Closed"))
    axes[0, 0].set_ylabel("Held-out complex RelL2")
    axes[0, 0].set_title("A  Wavelength-scaled oracle", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    budget_labels = ("1 src/1 f", "2 src/1 f", "4 src/1 f", "6 src/1 f", "full")
    for name, color in colors.items():
        means = []
        stds = []
        for budget in SYNTHETIC_BUDGETS:
            key = f"{name}_rel_l2"
            values = np.asarray(
                [row[key] for row in synthetic_rows if row["budget"] == budget]
            )
            means.append(values.mean())
            stds.append(values.std())
        axes[0, 1].errorbar(
            np.arange(len(SYNTHETIC_BUDGETS)),
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            color=color,
            label="Phase token" if name == "control" else "Multiscale Gram",
        )
    axes[0, 1].set_xticks(np.arange(len(SYNTHETIC_BUDGETS)), budget_labels, rotation=18)
    axes[0, 1].set_ylabel("Contrast RelL2")
    axes[0, 1].set_title("B  Synthetic measurement budget", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    own = {row["boundary"]: row for row in boundary_summary}
    x = np.arange(3)
    width = 0.34
    axes[1, 0].bar(
        x - width / 2,
        [own[label]["control_rel_l2"] for label in BOUNDARIES],
        width,
        color=colors["control"],
        label="Phase token",
    )
    axes[1, 0].bar(
        x + width / 2,
        [own[label]["candidate_rel_l2"] for label in BOUNDARIES],
        width,
        color=colors["candidate"],
        label="Multiscale Gram",
    )
    for position, label in enumerate(BOUNDARIES):
        gain = 100 * own[label]["paired_relative_improvement"]
        axes[1, 0].text(position, own[label]["control_rel_l2"] + 0.012, f"{gain:.1f}%", ha="center", fontsize=9)
    axes[1, 0].axhline(0, color="black", linewidth=0.5)
    axes[1, 0].set_xticks(x, ("Open", "Partial", "Closed"))
    axes[1, 0].set_ylabel("Contrast RelL2")
    axes[1, 0].set_title("C  64→128 boundary shift (3 seeds)", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)

    budgets = ("1", "2", "3", "4")
    x = np.arange(4)
    for key, label, color, style in (
        ("control", "Phase token", colors["control"], "--"),
        ("independent_packet_phase", "Independent packet phase", "#D97706", ":"),
        ("relative_phase_gram", "Relative-phase Gram", colors["candidate"], "-"),
    ):
        axes[1, 1].plot(
            x,
            [wavebench["budgets"][budget][key] for budget in budgets],
            marker="o",
            linestyle=style,
            color=color,
            label=label,
        )
    axes[1, 1].set_xticks(x, ("1", "2", "3", "4"))
    axes[1, 1].set_xlabel("Observed frequencies")
    axes[1, 1].set_ylabel("Contrast RelL2")
    axes[1, 1].set_title("D  WaveBench mechanism ablation", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/multibranch_phase_space_mps1")
    )
    parser.add_argument(
        "--oracle-summary",
        type=Path,
        default=Path("artifacts/multibranch_phase_space_mps0b/summary.json"),
    )
    parser.add_argument(
        "--wavebench-phase-root",
        type=Path,
        default=Path("artifacts/wavebench_representation_m1/development_seed20260923"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/multibranch_phase_space_mps1/summary"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    oracle = _load_json(args.oracle_summary)
    synthetic_rows = _synthetic_rows(args.root)
    boundary_seed_rows, boundary_paired_rows = _boundary_rows(args.root)
    boundary_summary = _aggregate_boundary(boundary_paired_rows)
    wavebench = _wavebench_summary(args.root, args.wavebench_phase_root)

    hard = {
        row["boundary"]: row for row in boundary_summary if row["boundary"] in {"partial", "closed"}
    }
    gate_passed = any(
        row["paired_relative_improvement"] >= 0.15
        and row["paired_improvement_95ci"][0] > 0
        for row in hard.values()
    )
    iid_full = [row for row in synthetic_rows if row["budget"] == "s6_f4"]
    iid_relative_change = float(
        np.mean(
            [
                (row["candidate_rel_l2"] - row["control_rel_l2"])
                / row["control_rel_l2"]
                for row in iid_full
            ]
        )
    )
    safety_passed = iid_relative_change <= 0.02
    summary = {
        "schema_version": 1,
        "oracle": {
            "gate_passed": oracle["promotion_gate"]["passed"],
            "adaptive": oracle["adaptive"],
            "interpretation": "multi-component local headroom, not semantic path recovery",
        },
        "synthetic_budget": synthetic_rows,
        "boundary_shift": boundary_summary,
        "wavebench": wavebench,
        "promotion_gate": {
            "required_hard_condition_paired_gain": 0.15,
            "method_gate_passed": gate_passed,
            "iid_safety_passed": safety_passed,
            "iid_full_context_relative_change": iid_relative_change,
            "decision": "stop_front_end_enumeration" if not gate_passed else "promote",
        },
        "claim_boundary": (
            "The oracle establishes missing local branch information. The fixed multiscale "
            "Gram representation improves frequency-poor and reflective settings, but does "
            "not pass the preregistered 15% three-seed method gate."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(args.output / "synthetic_budget.csv", synthetic_rows)
    _write_csv(args.output / "boundary_seed_system.csv", boundary_seed_rows)
    _write_csv(args.output / "boundary_system_paired.csv", boundary_paired_rows)
    _write_csv(args.output / "boundary_summary.csv", boundary_summary)
    _plot(
        oracle,
        synthetic_rows,
        boundary_summary,
        wavebench,
        args.output / "multibranch_phase_space_summary.png",
    )
    print(json.dumps(summary["promotion_gate"], indent=2))


if __name__ == "__main__":
    main()
