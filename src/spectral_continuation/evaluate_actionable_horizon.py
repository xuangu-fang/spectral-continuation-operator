"""Evaluate target-free value prediction over counterfactual calibration actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .horizon import reliable_horizon
from .metrics import paired_bootstrap
from .risk import RidgeLinearRegression, spearman_correlation


def action_trajectories(
    base_error: np.ndarray,
    calibrated_error: np.ndarray,
    frequencies: np.ndarray,
    endpoints: np.ndarray,
) -> np.ndarray:
    """Return [environment, action, query] deployment trajectories."""
    output = np.broadcast_to(base_error[:, None], calibrated_error.shape).copy()
    for action, endpoint in enumerate(endpoints):
        future = frequencies >= endpoint
        output[:, action, future] = calibrated_error[:, action, future]
    return output


def _frequency_positions(frequencies: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            int(np.argmin(np.abs(frequencies - max(endpoint, frequencies[0]))))
            for endpoint in endpoints
        ]
    )


def _sensor_patch_statistics(
    spatial_risk: np.ndarray,
    sensor_indices: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Summarize the candidate sensor footprint on a patch-level risk map."""
    environments, rows, columns = spatial_risk.shape
    row = sensor_indices // width
    column = sensor_indices % width
    patch_row = np.minimum(row * rows // height, rows - 1)
    patch_column = np.minimum(column * columns // width, columns - 1)
    sampled = spatial_risk[np.arange(environments)[:, None], patch_row, patch_column]
    unique_fraction = np.empty(environments)
    for environment in range(environments):
        patches = patch_row[environment] * columns + patch_column[environment]
        unique_fraction[environment] = len(np.unique(patches)) / (rows * columns)
    return np.column_stack(
        (
            sampled.mean(axis=1),
            sampled.max(axis=1),
            sampled.std(axis=1),
            unique_fraction,
            row.mean(axis=1) / max(height - 1, 1),
            row.std(axis=1) / max(height - 1, 1),
            column.mean(axis=1) / max(width - 1, 1),
            column.std(axis=1) / max(width - 1, 1),
        )
    )


def build_action_features(
    base_features: np.ndarray,
    base_feature_names: tuple[str, ...],
    risk_probability: np.ndarray,
    spatial_risk_probability: np.ndarray,
    frequencies: np.ndarray,
    endpoints: np.ndarray,
    sensor_indices: np.ndarray,
    field_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct pre-action state--action features without query targets."""
    positions = _frequency_positions(frequencies, endpoints)
    environments, actions = len(base_features), len(endpoints)
    selected_base = base_features[:, positions]
    selected_risk = risk_probability[:, positions]
    future_mean = np.empty((environments, actions))
    future_max = np.empty_like(future_mean)
    future_slope = np.empty_like(future_mean)
    patch_global_mean = np.empty_like(future_mean)
    patch_global_max = np.empty_like(future_mean)
    mask_statistics = np.empty((environments, actions, 8))
    for action, position in enumerate(positions):
        future = risk_probability[:, position:]
        future_mean[:, action] = future.mean(axis=1)
        future_max[:, action] = future.max(axis=1)
        future_slope[:, action] = future[:, -1] - future[:, 0]
        current_spatial = spatial_risk_probability[:, position]
        patch_global_mean[:, action] = current_spatial.mean(axis=(1, 2))
        patch_global_max[:, action] = current_spatial.max(axis=(1, 2))
        mask_statistics[:, action] = _sensor_patch_statistics(
            current_spatial,
            sensor_indices,
            field_shape[0],
            field_shape[1],
        )
    endpoint_feature = np.broadcast_to(endpoints[None], (environments, actions))
    remaining = np.stack([(frequencies >= endpoint).mean() for endpoint in endpoints])
    remaining = np.broadcast_to(remaining[None], (environments, actions))
    features = np.concatenate(
        (
            endpoint_feature[..., None],
            remaining[..., None],
            selected_base,
            selected_risk[..., None],
            future_mean[..., None],
            future_max[..., None],
            future_slope[..., None],
            patch_global_mean[..., None],
            patch_global_max[..., None],
            mask_statistics,
        ),
        axis=2,
    )
    names = (
        "action_endpoint_hz",
        "remaining_query_fraction",
        *base_feature_names,
        "failure_risk_at_action",
        "future_failure_risk_mean",
        "future_failure_risk_max",
        "future_failure_risk_slope",
        "spatial_risk_mean",
        "spatial_risk_max",
        "mask_sampled_risk_mean",
        "mask_sampled_risk_max",
        "mask_sampled_risk_std",
        "mask_patch_coverage",
        "mask_row_mean",
        "mask_row_std",
        "mask_column_mean",
        "mask_column_std",
    )
    return features, names


def _nested_ridge(
    features: np.ndarray,
    utility: np.ndarray,
    train_environment: np.ndarray,
    ridge_grid: tuple[float, ...],
    folds: int,
    center_targets: bool,
) -> float:
    candidates = []
    train_indices = np.flatnonzero(train_environment)
    for ridge in ridge_grid:
        losses = []
        for inner in range(folds):
            inner_test = train_indices[np.arange(len(train_indices)) % folds == inner]
            inner_train = np.setdiff1d(train_indices, inner_test)
            if not len(inner_test) or not len(inner_train):
                continue
            target = utility[inner_train]
            if center_targets:
                target = target - target.mean(axis=1, keepdims=True)
            model = RidgeLinearRegression(ridge=ridge).fit(
                features[inner_train].reshape(-1, features.shape[-1]), target.reshape(-1)
            )
            prediction = model.predict(
                features[inner_test].reshape(-1, features.shape[-1])
            ).reshape(len(inner_test), features.shape[1])
            truth = utility[inner_test]
            if center_targets:
                truth = truth - truth.mean(axis=1, keepdims=True)
            losses.append(float(np.mean((prediction - truth) ** 2)))
        candidates.append((np.mean(losses), ridge))
    return float(min(candidates)[1])


def cross_validated_value_prediction(
    features: np.ndarray,
    utility: np.ndarray,
    fold_ids: np.ndarray,
    ridge_grid: tuple[float, ...],
    center_targets: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """OOF utility rankings, with all actions from one environment kept together."""
    prediction = np.empty_like(utility)
    selected_ridges = []
    for fold in np.unique(fold_ids):
        train_environment = fold_ids != fold
        test_environment = fold_ids == fold
        ridge = _nested_ridge(
            features,
            utility,
            train_environment,
            ridge_grid,
            folds=3,
            center_targets=center_targets,
        )
        target = utility[train_environment]
        if center_targets:
            target = target - target.mean(axis=1, keepdims=True)
        model = RidgeLinearRegression(ridge=ridge).fit(
            features[train_environment].reshape(-1, features.shape[-1]), target.reshape(-1)
        )
        prediction[test_environment] = model.predict(
            features[test_environment].reshape(-1, features.shape[-1])
        ).reshape(test_environment.sum(), features.shape[1])
        selected_ridges.append(ridge)
    return prediction, selected_ridges


def _policy_record(
    name: str,
    selected: np.ndarray,
    trajectories: np.ndarray,
    base_error: np.ndarray,
    oracle_utility: np.ndarray,
    frequencies: np.ndarray,
    reference_frequency: float,
    error_threshold: float,
) -> dict[str, object]:
    no_action = selected < 0
    chosen_trajectory = base_error.copy()
    if (~no_action).any():
        chosen_trajectory[~no_action] = trajectories[
            np.flatnonzero(~no_action), selected[~no_action]
        ]
    base_macro = base_error.mean(axis=1)
    policy_macro = chosen_trajectory.mean(axis=1)
    utility = base_macro - policy_macro
    base_horizon = reliable_horizon(base_error, frequencies, reference_frequency, error_threshold)[
        0
    ]
    policy_horizon = reliable_horizon(
        chosen_trajectory, frequencies, reference_frequency, error_threshold
    )[0]
    return {
        "name": name,
        "macro_complex_rel_l2": float(policy_macro.mean()),
        "mean_utility": float(utility.mean()),
        "median_utility": float(np.median(utility)),
        "harmful_calibration_rate": float(np.mean((utility <= 0) & (~no_action))),
        "no_action_rate": float(no_action.mean()),
        "mean_action_regret": float(np.mean(oracle_utility - utility)),
        "median_reliable_horizon_hz": float(np.median(policy_horizon)),
        "mean_reliable_horizon_hz": float(np.mean(policy_horizon)),
        "mean_horizon_extension_hz": float(np.mean(policy_horizon - base_horizon)),
        "selected_action_indices": selected.tolist(),
        "environment_macro_complex_rel_l2": policy_macro.tolist(),
        "environment_utility": utility.tolist(),
    }


def _fold_specific_fixed(utility: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    selected = np.empty(len(utility), dtype=int)
    for fold in np.unique(fold_ids):
        train = fold_ids != fold
        action = int(np.argmax(utility[train].mean(axis=0)))
        selected[fold_ids == fold] = action
    return selected


def _risk_trigger_policy(
    risk_probability: np.ndarray,
    frequencies: np.ndarray,
    endpoints: np.ndarray,
    mask_ids: np.ndarray,
    utility: np.ndarray,
    fold_ids: np.ndarray,
    threshold: float,
) -> np.ndarray:
    trigger = np.full(len(risk_probability), endpoints.max())
    for environment, row in enumerate(risk_probability):
        crossings = np.flatnonzero(row > threshold)
        if len(crossings):
            trigger[environment] = np.clip(
                frequencies[crossings[0]], endpoints.min(), endpoints.max()
            )
    selected = np.empty(len(utility), dtype=int)
    for fold in np.unique(fold_ids):
        train = fold_ids != fold
        test = fold_ids == fold
        mask_values = np.unique(mask_ids)
        scores = []
        for mask in mask_values:
            action = np.asarray(
                [np.flatnonzero((endpoints == value) & (mask_ids == mask))[0] for value in trigger]
            )
            scores.append(utility[train, action[train]].mean())
        best_mask = mask_values[int(np.argmax(scores))]
        selected[test] = [
            np.flatnonzero((endpoints == value) & (mask_ids == best_mask))[0]
            for value in trigger[test]
        ]
    return selected


def _risk_trigger_fixed_mask(
    risk_probability: np.ndarray,
    frequencies: np.ndarray,
    endpoints: np.ndarray,
    mask_ids: np.ndarray,
    threshold: float,
    mask_id: int,
) -> np.ndarray:
    """Use the frozen risk trigger with one pre-specified sensor mask."""
    selected = np.empty(len(risk_probability), dtype=int)
    for environment, row in enumerate(risk_probability):
        crossings = np.flatnonzero(row > threshold)
        endpoint = endpoints.max()
        if len(crossings):
            endpoint = np.clip(frequencies[crossings[0]], endpoints.min(), endpoints.max())
        selected[environment] = np.flatnonzero((endpoints == endpoint) & (mask_ids == mask_id))[0]
    return selected


def run(args: argparse.Namespace) -> None:
    replay_paths = args.sequential_errors
    replay_archives = [np.load(path, allow_pickle=False) for path in replay_paths]
    first = replay_archives[0]
    environment_ids = first["environment_ids"].astype(str)
    frequencies = np.asarray(first["query_frequencies_hz"], dtype=float)
    base_error = np.asarray(first["base_error"], dtype=float)
    action_endpoints, mask_ids, calibrated, sensors = [], [], [], []
    for path, archive, mask_id in zip(replay_paths, replay_archives, args.mask_ids, strict=True):
        if not np.array_equal(environment_ids, archive["environment_ids"].astype(str)):
            raise RuntimeError(f"environment mismatch in {path}")
        if not np.allclose(base_error, archive["base_error"]):
            raise RuntimeError(f"base prediction mismatch in {path}")
        current_endpoints = np.asarray(archive["calibration_endpoints_hz"], dtype=float)
        calibrated.append(np.asarray(archive["calibrated_error"], dtype=float))
        action_endpoints.extend(current_endpoints)
        mask_ids.extend([mask_id] * len(current_endpoints))
        sensors.extend([np.asarray(archive["sensor_indices"], dtype=int)] * len(current_endpoints))
    for archive in replay_archives:
        archive.close()
    endpoints = np.asarray(action_endpoints, dtype=float)
    mask_ids_array = np.asarray(mask_ids, dtype=int)
    calibrated_error = np.concatenate(calibrated, axis=1)
    sensor_by_action = np.stack(sensors, axis=1)

    trajectories = action_trajectories(base_error, calibrated_error, frequencies, endpoints)
    environment_base = base_error.mean(axis=1)
    utility = environment_base[:, None] - trajectories.mean(axis=2)
    local_utility = np.empty_like(utility)
    for action, endpoint in enumerate(endpoints):
        positions = np.flatnonzero(frequencies >= endpoint)[: args.local_window]
        local_utility[:, action] = (
            base_error[:, positions] - calibrated_error[:, action, positions]
        ).mean(axis=1)

    with np.load(args.features, allow_pickle=False) as archive:
        feature_ids = archive["environment_ids"].astype(str)
        feature_names = tuple(archive["feature_names"].astype(str))
        base_features = np.asarray(archive["features"], dtype=float)
    with np.load(args.risk_predictions, allow_pickle=False) as archive:
        risk_ids = archive["environment_ids"].astype(str)
        fold_ids = np.asarray(archive["fold_ids"], dtype=int)
        risk_probability = np.asarray(
            archive[args.risk_method + "_failure_probability"], dtype=float
        )
    with np.load(args.spatial_risk_predictions, allow_pickle=False) as archive:
        spatial_ids = archive["environment_ids"].astype(str)
        spatial_risk = np.asarray(
            archive[args.spatial_risk_method + "_failure_probability"], dtype=float
        )
    if not (
        np.array_equal(environment_ids, feature_ids)
        and np.array_equal(environment_ids, risk_ids)
        and np.array_equal(environment_ids, spatial_ids)
    ):
        raise RuntimeError("feature, risk, spatial-risk, and replay environments differ")

    action_features, action_feature_names = build_action_features(
        base_features,
        feature_names,
        risk_probability,
        spatial_risk,
        frequencies,
        endpoints,
        sensor_by_action[:, 0],
        tuple(args.field_shape),
    )
    # Replace mask statistics action by action; the base builder accepts one mask for convenience.
    positions = _frequency_positions(frequencies, endpoints)
    mask_offset = len(action_feature_names) - 8
    for action, position in enumerate(positions):
        action_features[:, action, mask_offset:] = _sensor_patch_statistics(
            spatial_risk[:, position],
            sensor_by_action[:, action],
            args.field_shape[0],
            args.field_shape[1],
        )

    feature_sets = {
        "endpoint_only": (0, 1),
        "risk_state": tuple(range(2 + len(feature_names) + 6)),
        "state_action": tuple(range(len(action_feature_names))),
    }
    predictions = {}
    absolute_predictions = {}
    selected_ridges = {}
    selected_absolute_ridges = {}
    for name, indices in feature_sets.items():
        predictions[name], selected_ridges[name] = cross_validated_value_prediction(
            action_features[..., indices], utility, fold_ids, tuple(args.ridge_grid)
        )
        absolute_predictions[name], selected_absolute_ridges[name] = (
            cross_validated_value_prediction(
                action_features[..., indices],
                utility,
                fold_ids,
                tuple(args.ridge_grid),
                center_targets=False,
            )
        )

    oracle_selected = utility.argmax(axis=1)
    fixed_selected = _fold_specific_fixed(utility, fold_ids)
    learned_selected = predictions[args.primary_model].argmax(axis=1)
    learned_absolute_value = absolute_predictions[args.primary_model][
        np.arange(len(utility)), learned_selected
    ]
    learned_selected[learned_absolute_value <= 0] = -1
    risk_selected = _risk_trigger_policy(
        risk_probability,
        frequencies,
        endpoints,
        mask_ids_array,
        utility,
        fold_ids,
        args.failure_probability,
    )
    fixed_mask_risk_selected = _risk_trigger_fixed_mask(
        risk_probability,
        frequencies,
        endpoints,
        mask_ids_array,
        args.failure_probability,
        args.primary_mask_id,
    )
    rng = np.random.default_rng(args.seed)
    random_selected = rng.integers(0, len(endpoints), size=len(environment_ids))
    policies = {
        "no_calibration": np.full(len(environment_ids), -1, dtype=int),
        "random_matched_cost": random_selected,
        "fold_specific_best_fixed": fixed_selected,
        "failure_risk_trigger_fold_mask": risk_selected,
        "failure_risk_trigger_primary_mask": fixed_mask_risk_selected,
        f"learned_{args.primary_model}": learned_selected,
        "counterfactual_oracle": oracle_selected,
    }
    oracle_utility = utility[np.arange(len(utility)), oracle_selected]
    records = {
        name: _policy_record(
            name,
            selected,
            trajectories,
            base_error,
            oracle_utility,
            frequencies,
            args.reference_frequency,
            args.error_threshold,
        )
        for name, selected in policies.items()
    }
    fixed_error = np.asarray(
        records["fold_specific_best_fixed"]["environment_macro_complex_rel_l2"]
    )
    learned_error = np.asarray(
        records[f"learned_{args.primary_model}"]["environment_macro_complex_rel_l2"]
    )
    risk_error = np.asarray(
        records["failure_risk_trigger_primary_mask"]["environment_macro_complex_rel_l2"]
    )
    fixed_macro = fixed_error.mean()
    development_fixed_action = int(np.argmax(utility.mean(axis=0)))
    development_fixed_macro = float(trajectories[:, development_fixed_action].mean(axis=1).mean())
    oracle_macro = records["counterfactual_oracle"]["macro_complex_rel_l2"]
    learned_macro = learned_error.mean()
    paper_headroom = 100 * (development_fixed_macro - oracle_macro) / development_fixed_macro
    oof_gap_closure = 100 * (fixed_macro - learned_macro) / max(fixed_macro - oracle_macro, 1e-12)
    shuffled_macro = np.empty(args.shuffled_replicates)
    for replicate in range(args.shuffled_replicates):
        permutation = rng.permutation(len(utility))
        shuffled_selected = predictions[args.primary_model][permutation].argmax(axis=1)
        shuffled_absolute = absolute_predictions[args.primary_model][permutation, shuffled_selected]
        shuffled_value = utility[np.arange(len(utility)), shuffled_selected]
        shuffled_value[shuffled_absolute <= 0] = 0.0
        shuffled_macro[replicate] = base_error.mean() - shuffled_value.mean()
    correlations = []
    for environment in range(len(utility)):
        value = spearman_correlation(
            utility[environment], predictions[args.primary_model][environment]
        )
        if np.isfinite(value):
            correlations.append(value)

    output = {
        "schema_version": 1,
        "status": "environment-OOF actionable-horizon value POC",
        "environment_count": len(environment_ids),
        "action_count": len(endpoints),
        "mask_ids": args.mask_ids,
        "sensor_budget_per_frequency": args.sensor_budget,
        "window": args.window,
        "complex_observations_per_nonempty_action": args.sensor_budget * args.window,
        "primary_utility": "fixed-interval trajectory raw complex RelL2 reduction",
        "feature_names": action_feature_names,
        "selected_ridges": selected_ridges,
        "selected_absolute_ridges": selected_absolute_ridges,
        "policies": records,
        "task_headroom": {
            "best_development_fixed_action": (
                f"{endpoints[development_fixed_action]:g}@{mask_ids_array[development_fixed_action]}"
            ),
            "best_development_fixed_macro_complex_rel_l2": development_fixed_macro,
            "fold_specific_fixed_macro_complex_rel_l2": float(fixed_macro),
            "oracle_macro_complex_rel_l2": float(oracle_macro),
            "oracle_absolute_improvement_over_best_development_fixed": float(
                development_fixed_macro - oracle_macro
            ),
            "oracle_relative_improvement_over_best_development_fixed_percent": float(
                paper_headroom
            ),
            "fraction_oracle_differs_from_best_development_fixed": float(
                np.mean(oracle_selected != development_fixed_action)
            ),
            "oracle_action_histogram": {
                f"{endpoints[index]:g}@{mask_ids_array[index]}": int(count)
                for index, count in zip(
                    *np.unique(oracle_selected, return_counts=True), strict=True
                )
            },
        },
        "learned_policy": {
            "primary_model": args.primary_model,
            "within_environment_utility_spearman": float(np.mean(correlations)),
            "relative_improvement_over_fixed_percent": float(
                100 * (fixed_macro - learned_macro) / fixed_macro
            ),
            "fixed_to_oracle_gap_closure_percent": float(oof_gap_closure),
            "paired_vs_fold_specific_fixed": paired_bootstrap(
                fixed_error, learned_error, seed=args.seed
            ),
            "relative_improvement_over_primary_risk_trigger_percent": float(
                100 * (risk_error.mean() - learned_macro) / risk_error.mean()
            ),
            "paired_vs_primary_risk_trigger": paired_bootstrap(
                risk_error, learned_error, seed=args.seed
            ),
            "shuffled_instance_control": {
                "replicates": args.shuffled_replicates,
                "macro_complex_rel_l2_mean": float(shuffled_macro.mean()),
                "macro_complex_rel_l2_ci95_low": float(np.quantile(shuffled_macro, 0.025)),
                "macro_complex_rel_l2_ci95_high": float(np.quantile(shuffled_macro, 0.975)),
                "one_sided_p": float(
                    (1 + np.sum(shuffled_macro <= learned_macro)) / (1 + args.shuffled_replicates)
                ),
            },
        },
        "local_utility_summary": {
            "mean": float(local_utility.mean()),
            "oracle_mean": float(local_utility.max(axis=1).mean()),
            "positive_action_fraction": float((local_utility > 0).mean()),
        },
        "gate_decisions": {
            "paper_level_task_gate_passed": bool(paper_headroom >= args.paper_headroom_percent),
            "registered_m1_weak_baseline_gate_passed": bool(
                learned_macro < fixed_macro
                and oof_gap_closure >= args.gap_closure_percent
                and records[f"learned_{args.primary_model}"]["harmful_calibration_rate"]
                <= records["fold_specific_best_fixed"]["harmful_calibration_rate"]
                and (1 + np.sum(shuffled_macro <= learned_macro)) / (1 + args.shuffled_replicates)
                < 0.05
            ),
            "strong_risk_baseline_gate_passed": bool(learned_macro < risk_error.mean()),
            "decision": "stop selector scaling; redesign action space",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "actionable_horizon_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "actionable_horizon_predictions.npz",
        environment_ids=environment_ids,
        query_frequencies_hz=frequencies,
        action_endpoints_hz=endpoints,
        action_mask_ids=mask_ids_array,
        utility=utility,
        local_utility=local_utility,
        feature_names=np.asarray(action_feature_names),
        action_features=action_features,
        fold_ids=fold_ids,
        **{f"{name}_predicted_utility": value for name, value in predictions.items()},
        **{
            f"{name}_absolute_predicted_utility": value
            for name, value in absolute_predictions.items()
        },
    )

    figure, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=True)
    display = [
        "no_calibration",
        "fold_specific_best_fixed",
        "failure_risk_trigger_primary_mask",
        f"learned_{args.primary_model}",
        "counterfactual_oracle",
    ]
    labels = ["No cal.", "Fold fixed", "Risk trigger", "Value probe", "Oracle"]
    axes[0, 0].bar(labels, [records[name]["macro_complex_rel_l2"] for name in display])
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].set(ylabel="Macro complex RelL2", title="OOF matched-cost policies")
    centered_utility = utility - utility.mean(axis=1, keepdims=True)
    centered_prediction = predictions[args.primary_model] - predictions[args.primary_model].mean(
        axis=1, keepdims=True
    )
    axes[0, 1].scatter(centered_utility.ravel(), centered_prediction.ravel(), s=8, alpha=0.25)
    low = min(centered_utility.min(), centered_prediction.min())
    high = max(centered_utility.max(), centered_prediction.max())
    axes[0, 1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0, 1].set(
        xlabel="Centered true utility",
        ylabel="Centered OOF score",
        title="Within-environment ranking",
    )
    axes[1, 0].hist(utility.ravel(), bins=30, alpha=0.8)
    axes[1, 0].axvline(0, color="black", linewidth=1)
    axes[1, 0].set(xlabel="Action utility", ylabel="Count", title="Helpful vs harmful actions")
    fixed_utility = utility[np.arange(len(utility)), fixed_selected]
    learned_utility = np.where(
        learned_selected < 0,
        0.0,
        utility[np.arange(len(utility)), np.maximum(learned_selected, 0)],
    )
    axes[1, 1].scatter(fixed_utility, learned_utility, alpha=0.75)
    low = min(fixed_utility.min(), learned_utility.min())
    high = max(fixed_utility.max(), learned_utility.max())
    axes[1, 1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1, 1].set(
        xlabel="Best-fixed utility",
        ylabel="Learned-policy utility",
        title="Per-environment benefit",
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.22, linewidth=0.6)
    figure.savefig(args.output_dir / "actionable_horizon_poc.png", dpi=250)
    figure.savefig(args.output_dir / "actionable_horizon_poc.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-errors", type=Path, action="append", required=True)
    parser.add_argument("--mask-ids", type=int, nargs="+", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--risk-predictions", type=Path, required=True)
    parser.add_argument("--spatial-risk-predictions", type=Path, required=True)
    parser.add_argument("--risk-method", default="disagreement_only")
    parser.add_argument("--spatial-risk-method", default="local_combo")
    parser.add_argument(
        "--primary-model",
        choices=("endpoint_only", "risk_state", "state_action"),
        default="state_action",
    )
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--failure-probability", type=float, default=0.1)
    parser.add_argument("--primary-mask-id", type=int, default=4201)
    parser.add_argument("--reference-frequency", type=float, default=106.0)
    parser.add_argument("--error-threshold", type=float, default=0.5)
    parser.add_argument("--shuffled-replicates", type=int, default=20_000)
    parser.add_argument("--paper-headroom-percent", type=float, default=10.0)
    parser.add_argument("--gap-closure-percent", type=float, default=30.0)
    parser.add_argument("--local-window", type=int, default=6)
    parser.add_argument("--sensor-budget", type=int, default=92)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--field-shape", type=int, nargs=2, default=(48, 384))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.sequential_errors) != len(args.mask_ids):
        parser.error("--mask-ids must match the number of --sequential-errors inputs")
    return args


if __name__ == "__main__":
    run(parse_args())
