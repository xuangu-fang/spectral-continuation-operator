"""Evaluate repeated risk-triggered calibration against fixed and random schedules."""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .horizon import reliable_horizon
from .metrics import paired_bootstrap


def schedule_trajectory(
    base_error: np.ndarray,
    calibrated_error: np.ndarray,
    frequencies: np.ndarray,
    endpoints: np.ndarray,
    schedules: Sequence[Sequence[float]],
) -> np.ndarray:
    """Apply each environment's ordered reset schedule to its error trajectory."""
    output = np.asarray(base_error, dtype=float).copy()
    endpoint_lookup = {float(value): index for index, value in enumerate(endpoints)}
    for environment, schedule in enumerate(schedules):
        for endpoint in sorted(schedule):
            endpoint_index = endpoint_lookup[float(endpoint)]
            future = frequencies >= endpoint
            output[environment, future] = calibrated_error[environment, endpoint_index, future]
    return output


def adaptive_schedules(
    base_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    frequencies: np.ndarray,
    endpoints: np.ndarray,
    base_threshold: float,
    calibrated_threshold: float,
    maximum_actions: int,
) -> list[tuple[float, ...]]:
    """Trigger a reset when the currently active state's risk first crosses threshold."""
    endpoint_lookup = {float(value): index for index, value in enumerate(endpoints)}
    schedules: list[tuple[float, ...]] = []
    for environment in range(len(base_probability)):
        selected: list[float] = []
        active_endpoint_index: int | None = None
        for query_index, frequency in enumerate(frequencies):
            if float(frequency) not in endpoint_lookup:
                continue
            if active_endpoint_index is None:
                probability = base_probability[environment, query_index]
            else:
                probability = calibrated_probability[
                    environment, active_endpoint_index, query_index
                ]
            threshold = base_threshold if active_endpoint_index is None else calibrated_threshold
            if probability > threshold and len(selected) < maximum_actions:
                selected.append(float(frequency))
                active_endpoint_index = endpoint_lookup[float(frequency)]
        schedules.append(tuple(selected))
    return schedules


def _unique_frequency_cost(
    schedules: Sequence[Sequence[float]], window: int, sensors: int
) -> np.ndarray:
    costs = np.zeros(len(schedules), dtype=int)
    for index, schedule in enumerate(schedules):
        observed = set()
        for endpoint in schedule:
            observed.update(endpoint - offset for offset in range(window))
        costs[index] = len(observed) * sensors
    return costs


def _policy_record(
    trajectory: np.ndarray,
    schedules: Sequence[Sequence[float]],
    window: int,
    sensors: int,
) -> dict[str, object]:
    environment_error = trajectory.mean(axis=1)
    worst_count = max(1, int(np.ceil(0.25 * len(environment_error))))
    actions = np.asarray([len(schedule) for schedule in schedules], dtype=int)
    unique_cost = _unique_frequency_cost(schedules, window, sensors)
    histogram: dict[str, int] = {}
    for schedule in schedules:
        for endpoint in schedule:
            key = f"{endpoint:g}"
            histogram[key] = histogram.get(key, 0) + 1
    return {
        "macro_complex_rel_l2": float(trajectory.mean()),
        "mean_frequency_curve": trajectory.mean(axis=0).tolist(),
        "worst_quartile_mean_environment_rel_l2": float(
            np.sort(environment_error)[-worst_count:].mean()
        ),
        "mean_actions": float(actions.mean()),
        "action_count_histogram": {
            str(value): int(count)
            for value, count in zip(*np.unique(actions, return_counts=True), strict=True)
        },
        "mean_gross_complex_observations": float(actions.mean() * window * sensors),
        "mean_unique_complex_observations": float(unique_cost.mean()),
        "trigger_histogram": histogram,
    }


def _candidate_schedules(endpoints: np.ndarray, maximum_actions: int):
    yield ()
    for action_count in range(1, maximum_actions + 1):
        yield from itertools.combinations(map(float, endpoints), action_count)


def run(args: argparse.Namespace) -> None:
    with np.load(args.sequential_errors, allow_pickle=False) as archive:
        environment_ids = archive["environment_ids"].astype(str)
        frequencies = np.asarray(archive["query_frequencies_hz"], dtype=float)
        endpoints = np.asarray(archive["calibration_endpoints_hz"], dtype=float)
        base_error = np.asarray(archive["base_error"], dtype=float)
        calibrated_error = np.asarray(archive["calibrated_error"], dtype=float)
        shuffled_error = np.asarray(archive["shuffled_error"], dtype=float)
    with np.load(args.base_risk, allow_pickle=False) as archive:
        if not np.array_equal(environment_ids, archive["environment_ids"].astype(str)):
            raise RuntimeError("base-risk environment ids differ from sequential replay")
        base_probability = np.asarray(
            archive[f"{args.base_risk_method}_failure_probability"], dtype=float
        )
    with np.load(args.calibrated_risk, allow_pickle=False) as archive:
        if not np.array_equal(environment_ids, archive["environment_ids"].astype(str)):
            raise RuntimeError("calibrated-risk environment ids differ from sequential replay")
        calibrated_probability = np.asarray(
            archive[f"{args.calibrated_risk_method}_failure_probability"], dtype=float
        )

    candidate_endpoints = endpoints
    empty_schedules = [()] * len(environment_ids)
    policies: dict[str, dict[str, object]] = {
        "no_calibration": _policy_record(
            base_error, empty_schedules, args.window, args.sensor_budget
        )
    }
    trajectories = {"no_calibration": base_error}
    schedule_payload: dict[str, list[tuple[float, ...]]] = {"no_calibration": empty_schedules}
    rng = np.random.default_rng(args.random_seed)
    fixed_frontier = []
    adaptive_frontier = []
    random_frontier = []
    oracle_frontier = []
    gate_audit = []

    for maximum_actions in range(1, args.maximum_actions + 1):
        candidates = list(_candidate_schedules(candidate_endpoints, maximum_actions))
        candidate_trajectories = []
        candidate_environment_errors = []
        for schedule in candidates:
            repeated = [schedule] * len(environment_ids)
            trajectory = schedule_trajectory(
                base_error, calibrated_error, frequencies, endpoints, repeated
            )
            candidate_trajectories.append(trajectory)
            candidate_environment_errors.append(trajectory.mean(axis=1))
        candidate_macro = np.asarray([value.mean() for value in candidate_trajectories])
        candidate_unique_cost = np.asarray(
            [
                _unique_frequency_cost([schedule], args.window, args.sensor_budget)[0]
                for schedule in candidates
            ],
            dtype=float,
        )
        fixed_index = int(np.argmin(candidate_macro))
        fixed_schedule = candidates[fixed_index]
        fixed_schedules = [fixed_schedule] * len(environment_ids)
        fixed_trajectory = candidate_trajectories[fixed_index]

        environment_matrix = np.stack(candidate_environment_errors, axis=1)
        oracle_indices = np.argmin(environment_matrix, axis=1)
        oracle_schedules = [candidates[index] for index in oracle_indices]
        oracle_trajectory = schedule_trajectory(
            base_error, calibrated_error, frequencies, endpoints, oracle_schedules
        )

        current_adaptive_schedules = adaptive_schedules(
            base_probability,
            calibrated_probability,
            frequencies,
            endpoints,
            args.failure_probability,
            args.calibrated_failure_probability,
            maximum_actions,
        )
        adaptive_trajectory = schedule_trajectory(
            base_error,
            calibrated_error,
            frequencies,
            endpoints,
            current_adaptive_schedules,
        )
        shuffled_trajectory = schedule_trajectory(
            base_error,
            shuffled_error,
            frequencies,
            endpoints,
            current_adaptive_schedules,
        )

        random_errors = np.empty(args.random_replicates, dtype=float)
        random_trajectory_sum = np.zeros_like(base_error)
        random_unique_cost_sum = 0.0
        for replicate in range(args.random_replicates):
            random_schedules = [
                tuple(
                    sorted(
                        rng.choice(
                            candidate_endpoints,
                            size=maximum_actions,
                            replace=False,
                        ).tolist()
                    )
                )
                for _ in environment_ids
            ]
            random_trajectory = schedule_trajectory(
                base_error, calibrated_error, frequencies, endpoints, random_schedules
            )
            random_errors[replicate] = random_trajectory.mean()
            random_trajectory_sum += random_trajectory
            random_unique_cost_sum += _unique_frequency_cost(
                random_schedules, args.window, args.sensor_budget
            ).mean()
        mean_random_trajectory = random_trajectory_sum / args.random_replicates
        representative_random_schedules = [()] * len(environment_ids)

        names_and_values = {
            f"fixed_up_to_{maximum_actions}": (fixed_trajectory, fixed_schedules),
            f"adaptive_up_to_{maximum_actions}": (
                adaptive_trajectory,
                current_adaptive_schedules,
            ),
            f"adaptive_shuffled_up_to_{maximum_actions}": (
                shuffled_trajectory,
                current_adaptive_schedules,
            ),
            f"oracle_up_to_{maximum_actions}": (oracle_trajectory, oracle_schedules),
        }
        for name, (trajectory, schedules) in names_and_values.items():
            policies[name] = _policy_record(trajectory, schedules, args.window, args.sensor_budget)
            trajectories[name] = trajectory
            schedule_payload[name] = schedules
        random_record = _policy_record(
            mean_random_trajectory,
            representative_random_schedules,
            args.window,
            args.sensor_budget,
        )
        random_record.update(
            {
                "mean_actions": float(maximum_actions),
                "mean_gross_complex_observations": float(
                    maximum_actions * args.window * args.sensor_budget
                ),
                "mean_unique_complex_observations": float(
                    random_unique_cost_sum / args.random_replicates
                ),
                "action_count_histogram": {str(maximum_actions): len(environment_ids)},
                "trigger_histogram": {},
                "macro_error_replicate_mean": float(random_errors.mean()),
                "macro_error_replicate_ci95_low": float(np.quantile(random_errors, 0.025)),
                "macro_error_replicate_ci95_high": float(np.quantile(random_errors, 0.975)),
            }
        )
        random_name = f"random_exactly_{maximum_actions}"
        policies[random_name] = random_record
        trajectories[random_name] = mean_random_trajectory

        fixed_record = policies[f"fixed_up_to_{maximum_actions}"]
        adaptive_record = policies[f"adaptive_up_to_{maximum_actions}"]
        fixed_error = float(fixed_record["macro_complex_rel_l2"])
        adaptive_error = float(adaptive_record["macro_complex_rel_l2"])
        adaptive_unique_cost = float(adaptive_record["mean_unique_complex_observations"])
        fixed_worst = float(fixed_record["worst_quartile_mean_environment_rel_l2"])
        adaptive_worst = float(adaptive_record["worst_quartile_mean_environment_rel_l2"])
        comparison = paired_bootstrap(
            fixed_trajectory.mean(axis=1),
            adaptive_trajectory.mean(axis=1),
            seed=args.random_seed + maximum_actions,
        )
        cost_matched = np.flatnonzero(candidate_unique_cost <= adaptive_unique_cost + 1e-9)
        cost_matched_index = int(cost_matched[np.argmin(candidate_macro[cost_matched])])
        error_matched = np.flatnonzero(candidate_macro <= adaptive_error + 1e-12)
        if error_matched.size:
            error_matched_index = int(
                error_matched[np.argmin(candidate_unique_cost[error_matched])]
            )
            error_matched_cost = float(candidate_unique_cost[error_matched_index])
            cost_reduction = float(
                100 * (error_matched_cost - adaptive_unique_cost) / error_matched_cost
            )
            error_matched_schedule: list[float] | None = list(candidates[error_matched_index])
        else:
            error_matched_cost = None
            cost_reduction = None
            error_matched_schedule = None
        current_gate = {
            "maximum_actions": maximum_actions,
            "best_fixed_schedule_hz": list(fixed_schedule),
            "relative_improvement_over_best_fixed_percent": float(
                100 * (fixed_error - adaptive_error) / fixed_error
            ),
            "adaptive_mean_unique_complex_observations": adaptive_unique_cost,
            "cost_matched_fixed_schedule_hz": list(candidates[cost_matched_index]),
            "cost_matched_fixed_unique_complex_observations": float(
                candidate_unique_cost[cost_matched_index]
            ),
            "cost_matched_fixed_macro_complex_rel_l2": float(candidate_macro[cost_matched_index]),
            "relative_improvement_over_cost_matched_fixed_percent": float(
                100
                * (candidate_macro[cost_matched_index] - adaptive_error)
                / candidate_macro[cost_matched_index]
            ),
            "minimum_fixed_unique_cost_at_matched_or_lower_error": error_matched_cost,
            "error_matched_fixed_schedule_hz": error_matched_schedule,
            "unique_observation_reduction_at_matched_error_percent": cost_reduction,
            "adaptive_error_below_every_fixed_schedule": not error_matched.size,
            "worst_quartile_regression_percent": float(
                100 * (adaptive_worst - fixed_worst) / fixed_worst
            ),
            "paired_vs_fixed": comparison,
            "shuffled_instance_macro_complex_rel_l2": float(
                policies[f"adaptive_shuffled_up_to_{maximum_actions}"]["macro_complex_rel_l2"]
            ),
        }
        current_gate["matched_cost_error_gate_pass"] = (
            current_gate["relative_improvement_over_cost_matched_fixed_percent"]
            >= args.required_improvement_percent
        )
        current_gate["matched_error_cost_gate_pass"] = (
            cost_reduction is not None
            and cost_reduction >= args.required_observation_reduction_percent
        )
        current_gate["worst_quartile_gate_pass"] = (
            current_gate["worst_quartile_regression_percent"]
            <= args.maximum_worst_quartile_regression_percent
        )
        gate_audit.append(current_gate)

        fixed_frontier.append(fixed_record)
        adaptive_frontier.append(adaptive_record)
        random_frontier.append(random_record)
        oracle_frontier.append(policies[f"oracle_up_to_{maximum_actions}"])

    output = {
        "schema_version": 1,
        "status": "M2b repeated-reset sequential calibration frontier",
        "source_sequential_errors": str(args.sequential_errors),
        "source_base_risk": str(args.base_risk),
        "source_calibrated_risk": str(args.calibrated_risk),
        "environment_count": len(environment_ids),
        "query_hz": frequencies.tolist(),
        "calibration_endpoints_hz": endpoints.tolist(),
        "maximum_actions": args.maximum_actions,
        "failure_probability": args.failure_probability,
        "calibrated_failure_probability": args.calibrated_failure_probability,
        "sensor_budget_per_frequency": args.sensor_budget,
        "window": args.window,
        "base_risk_method": args.base_risk_method,
        "calibrated_risk_method": args.calibrated_risk_method,
        "full_field_query_coverage": 1.0,
        "cost_note": (
            "gross cost counts every sweep; unique cost reuses overlapping sensor-frequency "
            "observations. Full-field error is scored everywhere, including sensor pixels."
        ),
        "policies": policies,
        "gate_audit": gate_audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "repeated_calibration_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "repeated_calibration_trajectories.npz",
        environment_ids=environment_ids,
        query_frequencies_hz=frequencies,
        **{name: value for name, value in trajectories.items()},
    )

    figure, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=True)
    base_record = policies["no_calibration"]
    series = {
        "Best fixed": [base_record, *fixed_frontier],
        "Risk-adaptive": [base_record, *adaptive_frontier],
        "Random": [base_record, *random_frontier],
        "Instance oracle": [base_record, *oracle_frontier],
    }
    for label, records in series.items():
        axes[0, 0].plot(
            [record["mean_gross_complex_observations"] for record in records],
            [record["macro_complex_rel_l2"] for record in records],
            marker="o",
            label=label,
        )
        axes[0, 1].plot(
            [record["mean_unique_complex_observations"] for record in records],
            [record["macro_complex_rel_l2"] for record in records],
            marker="o",
            label=label,
        )
    axes[0, 0].set(
        xlabel="Gross complex observations / env",
        ylabel="Macro complex RelL2",
        title="Action-cost frontier",
    )
    axes[0, 1].set(
        xlabel="Unique complex observations / env",
        ylabel="Macro complex RelL2",
        title="Reuse-aware cost frontier",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    final_adaptive_name = f"adaptive_up_to_{args.maximum_actions}"
    final_fixed_name = f"fixed_up_to_{args.maximum_actions}"
    axes[1, 0].plot(
        frequencies,
        base_error.mean(axis=0),
        marker="o",
        label="No calibration",
    )
    axes[1, 0].plot(
        frequencies,
        trajectories[final_fixed_name].mean(axis=0),
        marker="o",
        label="Best fixed",
    )
    axes[1, 0].plot(
        frequencies,
        trajectories[final_adaptive_name].mean(axis=0),
        marker="o",
        label="Risk-adaptive",
    )
    axes[1, 0].set(
        xlabel="Query frequency (Hz)",
        ylabel="Complex RelL2",
        title=f"Up to {args.maximum_actions} resets",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    horizon = reliable_horizon(
        base_error, frequencies, args.reference_frequency, args.error_threshold
    )[0]
    adaptive_environment = trajectories[final_adaptive_name].mean(axis=1)
    fixed_environment = trajectories[final_fixed_name].mean(axis=1)
    axes[1, 1].scatter(horizon, fixed_environment - adaptive_environment, alpha=0.75)
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set(
        xlabel="Initial true reliable horizon (Hz)",
        ylabel="Fixed minus adaptive error",
        title="Instance-level adaptive gain",
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.22, linewidth=0.6)
    figure.savefig(args.output_dir / "repeated_calibration_frontier.png", dpi=250)
    figure.savefig(args.output_dir / "repeated_calibration_frontier.pdf")
    plt.close(figure)

    schedule_json = {
        name: [list(schedule) for schedule in schedules]
        for name, schedules in schedule_payload.items()
    }
    (args.output_dir / "repeated_calibration_schedules.json").write_text(
        json.dumps(schedule_json, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-errors", type=Path, required=True)
    parser.add_argument("--base-risk", type=Path, required=True)
    parser.add_argument("--calibrated-risk", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-risk-method", default="disagreement_only")
    parser.add_argument("--calibrated-risk-method", default="disagreement_only")
    parser.add_argument("--failure-probability", type=float, default=0.1)
    parser.add_argument("--calibrated-failure-probability", type=float, default=0.1)
    parser.add_argument("--maximum-actions", type=int, default=3)
    parser.add_argument("--sensor-budget", type=int, default=92)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--random-replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--reference-frequency", type=float, default=106.0)
    parser.add_argument("--error-threshold", type=float, default=0.5)
    parser.add_argument("--required-improvement-percent", type=float, default=10.0)
    parser.add_argument("--required-observation-reduction-percent", type=float, default=20.0)
    parser.add_argument("--maximum-worst-quartile-regression-percent", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
