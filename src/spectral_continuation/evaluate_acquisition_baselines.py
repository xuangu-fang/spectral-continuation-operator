"""Matched source--frequency acquisition benchmark for wave interrogation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .cross_domain_response_data import load_cross_domain_archive
from .evaluate_sequential_active_wave_query import (
    adapter_error,
    attach_evaluation_archive,
    greedy_variance_schedule,
)
from .probabilistic_response_operator import paired_summary
from .train_posterior_operator_composer import DomainState, _build_state


def legal_candidates(state: DomainState, target: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (source, frequency)
            for source in range(state.source_count)
            for frequency in range(state.frequency_count)
            if source != int(target[0]) and frequency != int(target[1])
        ],
        dtype=np.int64,
    )


def random_order(state: DomainState, target: np.ndarray, seed: int) -> np.ndarray:
    candidates = legal_candidates(state, target)
    event_seed = seed + 1_000_003 * int(target[0]) + 10_007 * int(target[1])
    return candidates[np.random.default_rng(event_seed).permutation(len(candidates))]


def action_coordinates(state: DomainState) -> np.ndarray:
    source = np.asarray(state.archive.sources, dtype=np.float64)
    source = (source - source.min(axis=0, keepdims=True)) / np.ptp(
        source, axis=0, keepdims=True
    ).clip(min=1e-12)
    frequency = np.log(np.asarray(state.archive.frequencies, dtype=np.float64))
    frequency = (frequency - frequency.min()) / max(float(np.ptp(frequency)), 1e-12)
    return np.asarray(
        [
            np.concatenate((source[source_index], [frequency[frequency_index]]))
            for source_index in range(state.source_count)
            for frequency_index in range(state.frequency_count)
        ]
    )


def nearest_target_schedule(
    state: DomainState,
    target: np.ndarray,
    ordered: np.ndarray,
    maximum_budget: int,
) -> np.ndarray:
    coordinate = action_coordinates(state)
    target_index = int(target[0]) * state.frequency_count + int(target[1])
    remaining = ordered[1:]
    indices = remaining[:, 0] * state.frequency_count + remaining[:, 1]
    distance = np.square(coordinate[indices] - coordinate[target_index]).sum(axis=1)
    order = np.argsort(distance, kind="stable")
    return np.concatenate((ordered[:1], remaining[order][: maximum_budget - 1]))


def farthest_schedule(
    state: DomainState,
    ordered: np.ndarray,
    maximum_budget: int,
) -> np.ndarray:
    coordinate = action_coordinates(state)
    selected = [ordered[0]]
    remaining = [value for value in ordered[1:]]
    while len(selected) < maximum_budget:
        selected_indices = np.asarray(
            [value[0] * state.frequency_count + value[1] for value in selected]
        )
        scores = []
        for candidate in remaining:
            index = candidate[0] * state.frequency_count + candidate[1]
            distance = np.square(
                coordinate[index][None] - coordinate[selected_indices]
            ).sum(axis=1)
            scores.append(float(distance.min()))
        selected.append(remaining.pop(int(np.argmax(scores))))
    return np.stack(selected)


def d_optimal_schedule(
    state: DomainState,
    ordered: np.ndarray,
    maximum_budget: int,
) -> np.ndarray:
    selected = [ordered[0]]
    remaining = [value for value in ordered[1:]]
    receiver = torch.arange(state.receiver_count, device=state.basis.device)
    while len(selected) < maximum_budget:
        scores = []
        for candidate in remaining:
            context = np.stack((*selected, candidate))
            query = torch.as_tensor(
                context[:, 0] * state.frequency_count + context[:, 1],
                dtype=torch.long,
                device=state.basis.device,
            )
            observed = (query[:, None] * state.receiver_count + receiver[None]).reshape(-1)
            basis = state.basis[observed].to(torch.complex128)
            precision = basis.mH @ basis + torch.diag(state.penalty.double()).to(
                torch.complex128
            )
            scores.append(float(torch.linalg.slogdet(precision)[1].real.cpu()))
        selected.append(remaining.pop(int(np.argmax(scores))))
    return np.stack(selected)


def schedule_cost(schedule: np.ndarray, receivers: int) -> dict[str, float]:
    return {
        "source_activations": float(len(schedule)),
        "unique_sources": float(len(np.unique(schedule[:, 0]))),
        "source_switches": float(np.sum(schedule[1:, 0] != schedule[:-1, 0])),
        "frequency_switches": float(np.sum(schedule[1:, 1] != schedule[:-1, 1])),
        "receiver_samples": float(len(schedule) * receivers),
    }


def _stored_schedule(
    payload: dict,
    domain: str,
    seed: int,
    target: np.ndarray,
) -> np.ndarray | None:
    key = f"{int(target[0])}_{int(target[1])}"
    values = payload.get("domains", {}).get(domain, {}).get(str(seed), {}).get(key)
    return None if values is None else np.asarray(values, dtype=np.int64)


@torch.inference_mode()
def run(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "base_config" in config:
        base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
        base.update(config)
        config = base
    device = torch.device(device_name)
    budgets = tuple(int(value) for value in config["active_budgets"])
    maximum_budget = max(budgets)
    schedule_files = dict(config.get("schedule_files", {}))
    if "population_schedule" in config:
        schedule_files.setdefault("population", config["population_schedule"])
    stored_schedules = {
        name: json.loads(Path(path).read_text(encoding="utf-8"))
        for name, path in schedule_files.items()
    }
    states: dict[tuple[str, int], DomainState] = {}
    for domain, setting in config["domains"].items():
        evaluation_archive = load_cross_domain_archive(
            Path(setting.get("evaluation_dataset", setting["dataset"]))
        )
        for budget in budgets:
            state = _build_state(
                domain,
                budget,
                setting,
                setting["budgets"][str(budget)],
                int(config["rank"]),
                int(config["value_rank"]),
                Path(config["adapter_checkpoint"]),
                device,
            )
            if "evaluation_dataset" in setting:
                attach_evaluation_archive(
                    state,
                    evaluation_archive,
                    str(setting.get("evaluation_source_split", "test")),
                )
            states[(domain, budget)] = state

    methods = ["random", "nearest", "farthest", "d_optimal", "target_risk"]
    methods.extend(stored_schedules)
    accumulators: dict[tuple[str, str, int], list[torch.Tensor]] = {}
    costs: dict[tuple[str, str, int], list[dict[str, float]]] = {}
    planning: dict[tuple[str, str], list[float]] = {}
    oracle_k2: dict[str, list[torch.Tensor]] = {}
    for seed in config["formal_test_seeds"]:
        for domain in config["domains"]:
            selection = states[(domain, int(config["selection_budget"]))]
            system_count = len(selection.split_raw["test"])
            totals = {
                (method, budget): torch.zeros(system_count, device=device)
                for method in methods
                for budget in budgets
            }
            oracle_total = torch.zeros(system_count, device=device)
            events = 0
            for target_source in range(selection.source_count):
                for target_frequency in range(selection.frequency_count):
                    target = np.asarray([target_source, target_frequency], dtype=np.int64)
                    ordered = random_order(selection, target, int(seed))
                    schedules = {"random": ordered[:maximum_budget]}
                    built = {
                        "nearest": nearest_target_schedule(
                            selection, target, ordered, maximum_budget
                        ),
                        "farthest": farthest_schedule(
                            selection, ordered, maximum_budget
                        ),
                    }
                    for method, schedule in built.items():
                        schedules[method] = schedule
                        planning.setdefault((domain, method), []).append(0.0)
                    builders = (
                        ("d_optimal", d_optimal_schedule, (selection, ordered, maximum_budget)),
                        (
                            "target_risk",
                            greedy_variance_schedule,
                            (selection, target, ordered, maximum_budget),
                        ),
                    )
                    for method, builder, arguments in builders:
                        start = time.perf_counter()
                        schedules[method] = builder(*arguments)
                        planning.setdefault((domain, method), []).append(
                            1e3 * (time.perf_counter() - start)
                        )
                    for method, payload in stored_schedules.items():
                        stored = _stored_schedule(payload, domain, int(seed), target)
                        if stored is None:
                            raise ValueError(
                                f"stored schedule {method!r} is missing {domain} "
                                f"seed={seed} target={target.tolist()}"
                            )
                        schedules[method] = stored
                    for method, schedule in schedules.items():
                        for budget in budgets:
                            state = states[(domain, budget)]
                            totals[(method, budget)] += adapter_error(
                                state, "test", schedule[:budget], target
                            )
                            costs.setdefault((domain, method, budget), []).append(
                                schedule_cost(schedule[:budget], state.receiver_count)
                            )
                    if 2 in budgets and bool(config.get("evaluate_k2_truth_oracle", True)):
                        candidate_errors = []
                        state = states[(domain, 2)]
                        for candidate in ordered[1:]:
                            candidate_errors.append(
                                adapter_error(
                                    state,
                                    "test",
                                    np.stack((ordered[0], candidate)),
                                    target,
                                )
                            )
                        oracle_total += torch.stack(candidate_errors).amin(dim=0)
                    events += 1
            for (method, budget), total in totals.items():
                accumulators.setdefault((domain, method, budget), []).append(
                    total / events
                )
            if bool(config.get("evaluate_k2_truth_oracle", True)):
                oracle_k2.setdefault(domain, []).append(oracle_total / events)

    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: torch.stack(values).mean(dim=0).cpu().numpy().astype(np.float64)
        for key, values in accumulators.items()
    }
    rows = []
    summary: dict = {
        "evaluation_datasets": {
            domain: setting.get("evaluation_dataset", setting["dataset"])
            for domain, setting in config["domains"].items()
        },
        "domains": {},
    }
    for domain in config["domains"]:
        domain_summary = {"budgets": {}}
        for budget in budgets:
            random = arrays[(domain, "random", budget)]
            budget_rows = []
            for method in methods:
                value = arrays[(domain, method, budget)]
                statistics = paired_summary(random, value, 20260928 + budget)
                cost_values = costs[(domain, method, budget)]
                row = {
                    "domain": domain,
                    "method": method,
                    "budget": budget,
                    "error": float(value.mean()),
                    "gain_vs_random": float(1.0 - value.mean() / random.mean()),
                    "bootstrap_95_low": float(statistics["bootstrap_95_low"]),
                    "bootstrap_95_high": float(statistics["bootstrap_95_high"]),
                    "system_win_rate": float(statistics["system_win_rate"]),
                    **{
                        name: float(np.mean([item[name] for item in cost_values]))
                        for name in cost_values[0]
                    },
                    "planning_ms": float(
                        np.mean(planning.get((domain, method), [0.0]))
                    ),
                }
                rows.append(row)
                budget_rows.append(row)
                np.savez_compressed(output / f"{domain.lower()}_{method}_k{budget}.npz", error=value)
            domain_summary["budgets"][str(budget)] = budget_rows
        x = np.log2(np.asarray(budgets, dtype=np.float64))
        domain_summary["auc"] = {
            method: float(
                np.trapezoid(
                    [arrays[(domain, method, budget)].mean() for budget in budgets], x
                )
            )
            for method in methods
        }
        if domain in oracle_k2:
            value = torch.stack(oracle_k2[domain]).mean(dim=0).cpu().numpy()
            domain_summary["k2_truth_oracle"] = {
                "error": float(value.mean()),
                "gain_vs_random": float(
                    1.0 - value.mean() / arrays[(domain, "random", 2)].mean()
                ),
            }
            np.savez_compressed(output / f"{domain.lower()}_truth_oracle_k2.npz", error=value)
        summary["domains"][domain] = domain_summary

    with (output / "baseline_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, len(config["domains"]), figsize=(11.5, 3.8))
    if len(config["domains"]) == 1:
        axes = [axes]
    for axis, domain in zip(axes, config["domains"], strict=True):
        for method in methods:
            axis.plot(
                budgets,
                [arrays[(domain, method, budget)].mean() for budget in budgets],
                marker="o",
                label=method.replace("_", " "),
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(budgets, labels=[str(value) for value in budgets])
        axis.set_title(domain)
        axis.set_xlabel("probe budget K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("raw complex relative L2")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "acquisition_baselines.png", dpi=220)
    figure.savefig(output / "acquisition_baselines.pdf")
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
