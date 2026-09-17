"""Summarize matched-cost one-trigger sequential calibration policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .horizon import reliable_horizon
from .metrics import paired_bootstrap


def _trajectory(
    base_error: np.ndarray,
    calibrated_error: np.ndarray,
    frequencies: np.ndarray,
    endpoints: np.ndarray,
    selected_endpoints: np.ndarray,
) -> np.ndarray:
    output = base_error.copy()
    for environment, endpoint in enumerate(selected_endpoints):
        endpoint_index = int(np.flatnonzero(endpoints == endpoint)[0])
        future = frequencies >= endpoint
        output[environment, future] = calibrated_error[environment, endpoint_index, future]
    return output


def _first_trigger(
    probability: np.ndarray,
    frequencies: np.ndarray,
    endpoints: np.ndarray,
    threshold: float,
) -> np.ndarray:
    selected = np.full(probability.shape[0], endpoints[-1])
    for environment, row in enumerate(probability):
        positions = np.flatnonzero(row > threshold)
        if positions.size:
            selected[environment] = np.clip(
                frequencies[int(positions[0])], endpoints[0], endpoints[-1]
            )
    return selected


def run(args: argparse.Namespace) -> None:
    with np.load(args.sequential_errors, allow_pickle=False) as archive:
        environment_ids = archive["environment_ids"].astype(str)
        frequencies = np.asarray(archive["query_frequencies_hz"], dtype=float)
        endpoints = np.asarray(archive["calibration_endpoints_hz"], dtype=float)
        base_error = np.asarray(archive["base_error"], dtype=float)
        calibrated_error = np.asarray(archive["calibrated_error"], dtype=float)
        shuffled_error = np.asarray(archive["shuffled_error"], dtype=float)
        dense_oracle_error = np.asarray(archive["dense_oracle_error"], dtype=float)
    with np.load(args.risk_predictions, allow_pickle=False) as archive:
        risk_ids = archive["environment_ids"].astype(str)
        risk_frequencies = np.asarray(archive["query_frequencies_hz"], dtype=float)
        risk_methods = {
            key.removesuffix("_failure_probability"): np.asarray(archive[key], dtype=float)
            for key in archive.files
            if key.endswith("_failure_probability")
        }
    if not np.array_equal(environment_ids, risk_ids) or not np.array_equal(
        frequencies, risk_frequencies
    ):
        raise RuntimeError("risk and sequential replay environments or queries differ")

    fixed = {}
    fixed_trajectories = {}
    for endpoint_index, endpoint in enumerate(endpoints):
        selected = np.full(len(environment_ids), endpoint)
        trajectory = _trajectory(base_error, calibrated_error, frequencies, endpoints, selected)
        fixed[f"{endpoint:g}"] = float(trajectory.mean())
        fixed_trajectories[endpoint] = trajectory
    best_fixed_endpoint = min(endpoints, key=lambda value: fixed[f"{value:g}"])
    best_fixed = fixed_trajectories[best_fixed_endpoint]

    policies = {
        "no_calibration": {
            "trajectory": base_error,
            "selected_endpoints_hz": None,
            "cost": 0,
        }
    }
    for method, probability in risk_methods.items():
        selected = _first_trigger(probability, frequencies, endpoints, args.failure_probability)
        policies[f"risk_{method}"] = {
            "trajectory": _trajectory(
                base_error, calibrated_error, frequencies, endpoints, selected
            ),
            "selected_endpoints_hz": selected,
            "cost": args.sensor_budget * args.window,
        }
    selected = policies[f"risk_{args.primary_risk_method}"]["selected_endpoints_hz"]
    assert selected is not None
    policies["best_fixed_development"] = {
        "trajectory": best_fixed,
        "selected_endpoints_hz": np.full(len(environment_ids), best_fixed_endpoint),
        "cost": args.sensor_budget * args.window,
    }
    policies["risk_trigger_shuffled_instance"] = {
        "trajectory": _trajectory(base_error, shuffled_error, frequencies, endpoints, selected),
        "selected_endpoints_hz": selected,
        "cost": args.sensor_budget * args.window,
    }
    policies["risk_trigger_dense_oracle_calibration"] = {
        "trajectory": _trajectory(base_error, dense_oracle_error, frequencies, endpoints, selected),
        "selected_endpoints_hz": selected,
        "cost": args.window * args.full_field_points,
    }

    candidate_trajectories = np.stack(
        [fixed_trajectories[endpoint].mean(axis=1) for endpoint in endpoints], axis=1
    )
    oracle_indices = np.argmin(candidate_trajectories, axis=1)
    oracle_selected = endpoints[oracle_indices]
    oracle_trajectory = _trajectory(
        base_error, calibrated_error, frequencies, endpoints, oracle_selected
    )
    policies["oracle_instance_trigger"] = {
        "trajectory": oracle_trajectory,
        "selected_endpoints_hz": oracle_selected,
        "cost": args.sensor_budget * args.window,
    }

    primary_name = f"risk_{args.primary_risk_method}"
    primary = policies[primary_name]["trajectory"]
    rng = np.random.default_rng(args.permutation_seed)
    permutation_errors = np.empty(args.permutations)
    for index in range(args.permutations):
        permuted = rng.permutation(selected)
        permutation_errors[index] = _trajectory(
            base_error, calibrated_error, frequencies, endpoints, permuted
        ).mean()

    true_horizon = reliable_horizon(
        base_error, frequencies, args.reference_frequency, args.error_threshold
    )[0]
    output = {
        "schema_version": 1,
        "status": "M2a matched-cost one-trigger sequential calibration summary",
        "source_sequential_errors": str(args.sequential_errors),
        "source_risk_predictions": str(args.risk_predictions),
        "environment_count": len(environment_ids),
        "query_hz": frequencies.tolist(),
        "calibration_endpoints_hz": endpoints.tolist(),
        "sensor_budget_per_frequency": args.sensor_budget,
        "window": args.window,
        "complex_observations_per_trigger": args.sensor_budget * args.window,
        "full_field_query_coverage": 1.0,
        "fixed_endpoint_macro_complex_rel_l2": fixed,
        "best_fixed_endpoint_hz": float(best_fixed_endpoint),
        "policies": {},
        "primary_risk_method": args.primary_risk_method,
        "primary_vs_endpoint_permutation": {
            "actual_macro_complex_rel_l2": float(primary.mean()),
            "permuted_mean": float(permutation_errors.mean()),
            "permuted_ci95_low": float(np.quantile(permutation_errors, 0.025)),
            "permuted_ci95_high": float(np.quantile(permutation_errors, 0.975)),
            "one_sided_p": float(
                (1 + np.sum(permutation_errors <= primary.mean())) / (1 + len(permutation_errors))
            ),
        },
    }
    for name, current in policies.items():
        trajectory = current["trajectory"]
        endpoint_values = current["selected_endpoints_hz"]
        record = {
            "macro_complex_rel_l2": float(trajectory.mean()),
            "mean_environment_macro_complex_rel_l2": float(trajectory.mean(axis=1).mean()),
            "worst_quartile_environment_macro_complex_rel_l2": float(
                np.quantile(trajectory.mean(axis=1), 0.75)
            ),
            "mean_frequency_curve": trajectory.mean(axis=0).tolist(),
            "complex_observations_per_environment": int(current["cost"]),
        }
        if endpoint_values is not None:
            values, counts = np.unique(endpoint_values, return_counts=True)
            record["trigger_histogram"] = {
                f"{value:g}": int(count) for value, count in zip(values, counts, strict=True)
            }
        output["policies"][name] = record

    base_macro = output["policies"]["no_calibration"]["macro_complex_rel_l2"]
    fixed_macro = output["policies"]["best_fixed_development"]["macro_complex_rel_l2"]
    primary_macro = output["policies"][primary_name]["macro_complex_rel_l2"]
    oracle_macro = output["policies"]["oracle_instance_trigger"]["macro_complex_rel_l2"]
    output["primary_comparison"] = {
        "relative_improvement_over_no_calibration_percent": float(
            100 * (base_macro - primary_macro) / base_macro
        ),
        "relative_improvement_over_best_fixed_percent": float(
            100 * (fixed_macro - primary_macro) / fixed_macro
        ),
        "adaptive_oracle_gap_closure_percent": float(
            100 * (fixed_macro - primary_macro) / max(fixed_macro - oracle_macro, 1e-12)
        ),
        "paired_vs_best_fixed": paired_bootstrap(
            best_fixed.mean(axis=1), primary.mean(axis=1), seed=args.permutation_seed
        ),
    }
    output["true_base_horizon_hz"] = {
        "median": float(np.median(true_horizon)),
        "q25": float(np.quantile(true_horizon, 0.25)),
        "q75": float(np.quantile(true_horizon, 0.75)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sequential_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.7), constrained_layout=True)
    display = [
        "no_calibration",
        "best_fixed_development",
        primary_name,
        "risk_trigger_shuffled_instance",
        "oracle_instance_trigger",
    ]
    labels = [
        "No cal.",
        f"Fixed {best_fixed_endpoint:g} Hz",
        "Risk trigger",
        "Shuffled",
        "Oracle trigger",
    ]
    values = [output["policies"][name]["macro_complex_rel_l2"] for name in display]
    axes[0, 0].bar(labels, values)
    axes[0, 0].tick_params(axis="x", rotation=28)
    axes[0, 0].set(ylabel="Macro complex RelL2", title="Matched-cost policies")

    for name, label in zip(display, labels, strict=True):
        axes[0, 1].plot(
            frequencies,
            output["policies"][name]["mean_frequency_curve"],
            marker="o",
            markersize=3,
            label=label,
        )
    axes[0, 1].set(
        xlabel="Query frequency (Hz)", ylabel="Complex RelL2", title="Sequential trajectory"
    )
    axes[0, 1].legend(frameon=False, fontsize=7)

    endpoint_values, endpoint_counts = np.unique(selected, return_counts=True)
    axes[1, 0].bar(endpoint_values, endpoint_counts, width=0.8)
    axes[1, 0].axvline(best_fixed_endpoint, color="tab:orange", linestyle="--", label="best fixed")
    axes[1, 0].set(
        xlabel="Triggered sweep endpoint (Hz)",
        ylabel="Environments",
        title="Adaptive trigger times",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    primary_environment = primary.mean(axis=1)
    fixed_environment = best_fixed.mean(axis=1)
    axes[1, 1].scatter(true_horizon, fixed_environment - primary_environment, alpha=0.75)
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set(
        xlabel="Base-model true horizon (Hz)",
        ylabel="Fixed minus risk-trigger error",
        title="Who benefits from adaptation?",
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.22, linewidth=0.6)
    figure.savefig(args.output_dir / "sequential_calibration.png", dpi=250)
    figure.savefig(args.output_dir / "sequential_calibration.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-errors", type=Path, required=True)
    parser.add_argument("--risk-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-risk-method", default="disagreement_only")
    parser.add_argument("--failure-probability", type=float, default=0.1)
    parser.add_argument("--error-threshold", type=float, default=0.5)
    parser.add_argument("--reference-frequency", type=float, default=106.0)
    parser.add_argument("--sensor-budget", type=int, default=92)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--full-field-points", type=int, default=48 * 384)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--permutation-seed", type=int, default=20260917)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
