"""Training-free paired-fidelity continuation oracle audit.

Predictions may consume the cheap-solver field at each requested query and
paired cheap/high-fidelity sensors at calibration frequencies.  High-fidelity
query fields are evaluation targets only and never enter a predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .paired_fidelity_protocol import (
    ARCHIVE_SCHEMA,
    FidelityArchive,
    validate_paired_fidelity_archives,
)

EPS = 1e-12
METRICS = (
    "complex_rel_l2",
    "amplitude_rel_l2",
    "weighted_phase_mae_rad",
    "phase_coherence",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_frequencies(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(",") if value.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("frequencies must be non-empty, unique, and increasing")
    return values


def _to_complex(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != 2:
        raise ValueError("complex field must have a final real/imaginary axis of length two")
    return values[..., 0].astype(np.float32) + 1j * values[..., 1].astype(np.float32)


@dataclass(frozen=True)
class ArchiveSlice:
    path: Path
    sha256: str
    ids: np.ndarray
    frequencies: tuple[float, ...]
    fields: np.ndarray  # complex64 [environment, frequency, height, width]


@dataclass(frozen=True)
class PairedAuditInputs:
    ids: np.ndarray
    calibration_frequencies: tuple[float, ...]
    query_frequencies: tuple[float, ...]
    low_calibration: np.ndarray
    high_calibration: np.ndarray
    low_query: np.ndarray
    high_query_target: np.ndarray
    low_archive: ArchiveSlice
    high_archive: ArchiveSlice
    fidelity_independent: bool
    identity_guard: dict[str, Any]


def _resolve_split(splits: np.ndarray, requested: str) -> str:
    available = set(np.asarray(splits).astype(str).tolist())
    aliases = {"val": "validation", "validation": "val"}
    if requested in available:
        return requested
    if aliases.get(requested) in available:
        return aliases[requested]
    raise ValueError(f"split {requested!r} is absent; available splits are {sorted(available)}")


def _load_archive_slice(
    path: Path,
    required_frequencies: tuple[float, ...],
    split: str,
) -> ArchiveSlice:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as archive:
        required_keys = {"complex_field", "frequencies_hz", "environment_ids", "split"}
        missing = required_keys.difference(archive.files)
        if missing:
            raise ValueError(f"archive is missing required keys {sorted(missing)}: {path}")
        frequencies = np.asarray(archive["frequencies_hz"], dtype=np.float64)
        ids = np.asarray(archive["environment_ids"]).astype(str)
        splits = np.asarray(archive["split"]).astype(str)
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"environment IDs are not unique: {path}")
        selected_split = _resolve_split(splits, split)
        positions = np.flatnonzero(splits == selected_split)
        lookup = {float(value): index for index, value in enumerate(frequencies)}
        absent = [value for value in required_frequencies if value not in lookup]
        if absent:
            raise ValueError(f"archive lacks frequencies {absent}: {path}")
        frequency_positions = np.asarray([lookup[value] for value in required_frequencies])
        raw = np.asarray(archive["complex_field"])
        if raw.ndim != 5 or raw.shape[0] != len(ids) or raw.shape[1] != len(frequencies):
            raise ValueError(f"unexpected complex_field shape {raw.shape}: {path}")
        # Advanced indexing extracts only the selected split/frequencies before
        # the archive is closed; downstream predictors cannot recover any other
        # frequency from this object.
        selected = raw[positions][:, frequency_positions].copy()
        if "source_ids" in archive.files:
            source_ids = np.asarray(archive["source_ids"]).astype(str)
            if source_ids.shape != ids.shape:
                raise ValueError(f"source_ids shape differs from environment_ids: {path}")
            ids = np.asarray(
                [
                    f"{environment}::{source}"
                    for environment, source in zip(ids, source_ids, strict=True)
                ]
            )
        selected_ids = ids[positions].copy()
    return ArchiveSlice(
        path=path,
        sha256=sha256_file(path),
        ids=selected_ids,
        frequencies=required_frequencies,
        fields=_to_complex(selected),
    )


def _slice_validated_archive(
    archive: FidelityArchive,
    required_frequencies: tuple[float, ...],
    split: str,
) -> ArchiveSlice:
    selected_split = _resolve_split(archive.split, split)
    positions = np.flatnonzero(archive.split == selected_split)
    lookup = {float(value): index for index, value in enumerate(archive.frequencies_hz)}
    absent = [value for value in required_frequencies if value not in lookup]
    if absent:
        raise ValueError(f"archive lacks frequencies {absent}: {archive.path}")
    frequency_positions = np.asarray([lookup[value] for value in required_frequencies])
    selected = archive.fields[positions][:, frequency_positions].copy()
    all_ids = np.asarray(
        [
            f"{environment}::{source}"
            for environment, source in zip(
                archive.environment_ids.astype(str), archive.source_ids.astype(str), strict=True
            )
        ]
    )
    return ArchiveSlice(
        path=archive.path,
        sha256=archive.archive_sha256,
        ids=all_ids[positions],
        frequencies=required_frequencies,
        fields=_to_complex(selected),
    )


def _archive_schema(path: Path) -> str | None:
    with np.load(path, allow_pickle=False) as archive:
        if "schema_version" not in archive.files:
            return None
        value = np.asarray(archive["schema_version"])
        return str(value.item()) if value.ndim == 0 else None


def load_paired_inputs(
    low_archive: Path,
    high_archive: Path,
    calibration_frequencies: tuple[float, ...],
    query_frequencies: tuple[float, ...],
    *,
    split: str = "val",
    allow_same_archive_smoke: bool = False,
) -> PairedAuditInputs:
    overlap = set(calibration_frequencies).intersection(query_frequencies)
    if overlap:
        raise ValueError(
            f"query-frequency sensor leakage: calibration overlaps queries {sorted(overlap)}"
        )
    if calibration_frequencies[-1] >= query_frequencies[0]:
        raise ValueError("calibration sensors must end strictly before the first query")
    combined = calibration_frequencies + query_frequencies
    gaps = np.diff(np.asarray(combined))
    if not np.allclose(gaps, gaps[0], rtol=0.0, atol=1e-9):
        raise ValueError("AR oracle requires one uniformly spaced continuation grid")

    low_path, high_path = low_archive.resolve(), high_archive.resolve()
    same_path = low_path == high_path
    if same_path and not allow_same_archive_smoke:
        raise ValueError("same-archive input is legacy smoke only; pass --allow-same-archive-smoke")
    low_schema = _archive_schema(low_path)
    high_schema = low_schema if same_path else _archive_schema(high_path)
    formal_protocol = low_schema == ARCHIVE_SCHEMA or high_schema == ARCHIVE_SCHEMA
    paired_identity_sha256 = None
    if formal_protocol:
        if same_path:
            raise ValueError("formal paired-fidelity protocol always requires two archives")
        if low_schema != ARCHIVE_SCHEMA or high_schema != ARCHIVE_SCHEMA:
            raise ValueError("low/high archive schema mismatch; both must use the formal protocol")
        pair = validate_paired_fidelity_archives(low_path, high_path)
        low = _slice_validated_archive(pair.low, combined, split)
        high = _slice_validated_archive(pair.high, combined, split)
        paired_identity_sha256 = pair.identity_sha256
    else:
        low = _load_archive_slice(low_path, combined, split)
        high = low if same_path else _load_archive_slice(high_path, combined, split)
    same_sha256 = low.sha256 == high.sha256
    if same_sha256 and not allow_same_archive_smoke:
        raise ValueError(
            "low/high archives have identical SHA-256; independent fidelity is required"
        )
    if set(low.ids.tolist()) != set(high.ids.tolist()):
        raise ValueError(
            "low/high archives do not contain the same environment IDs in the selected split"
        )
    high_position = {value: index for index, value in enumerate(high.ids.tolist())}
    high_order = np.asarray([high_position[value] for value in low.ids.tolist()])
    high_fields = high.fields[high_order]
    if low.fields.shape != high_fields.shape:
        raise ValueError(
            f"paired low/high field shapes differ: {low.fields.shape} versus {high_fields.shape}"
        )
    calibration_count = len(calibration_frequencies)
    independent = not same_path and not same_sha256
    return PairedAuditInputs(
        ids=low.ids,
        calibration_frequencies=calibration_frequencies,
        query_frequencies=query_frequencies,
        low_calibration=low.fields[:, :calibration_count],
        high_calibration=high_fields[:, :calibration_count],
        low_query=low.fields[:, calibration_count:],
        high_query_target=high_fields[:, calibration_count:],
        low_archive=low,
        high_archive=high,
        fidelity_independent=independent,
        identity_guard={
            "same_resolved_path": same_path,
            "same_sha256": same_sha256,
            "fidelity_independent": independent,
            "legacy_same_archive_smoke": not independent and allow_same_archive_smoke,
            "formal_protocol_validated": formal_protocol,
            "paired_identity_sha256": paired_identity_sha256,
        },
    )


def _fit_complex_ar1(sequence: np.ndarray, ridge: float, max_pole: float) -> np.ndarray:
    previous, following = sequence[:, :-1], sequence[:, 1:]
    numerator = np.sum(np.conj(previous) * following, axis=1)
    power = np.sum(np.abs(previous) ** 2, axis=1)
    coefficient = numerator / (power * (1.0 + ridge) + EPS)
    magnitude = np.abs(coefficient)
    return coefficient * np.minimum(1.0, max_pole / np.maximum(magnitude, EPS))


def dense_calibration_ar1_residual(
    low_calibration: np.ndarray,
    high_calibration: np.ndarray,
    low_query: np.ndarray,
    *,
    ridge: float = 1e-5,
    max_pole: float = 1.05,
) -> np.ndarray:
    """Per-environment, per-location AR(1) oracle for the fidelity discrepancy."""
    residual = high_calibration - low_calibration
    coefficient = _fit_complex_ar1(residual, ridge, max_pole)
    state = residual[:, -1]
    predictions = []
    for query_index in range(low_query.shape[1]):
        state = coefficient * state
        predictions.append(low_query[:, query_index] + state)
    return np.stack(predictions, axis=1)


def damped_velocity_linear_state_residual(
    low_calibration: np.ndarray,
    high_calibration: np.ndarray,
    low_query: np.ndarray,
    *,
    ridge: float = 1e-5,
    max_velocity_pole: float = 0.995,
) -> np.ndarray:
    """Controlled two-state discrepancy oracle: residual plus damped velocity."""
    residual = high_calibration - low_calibration
    velocity = np.diff(residual, axis=1)
    coefficient = _fit_complex_ar1(velocity, ridge, max_velocity_pole)
    state = residual[:, -1]
    speed = velocity[:, -1]
    predictions = []
    for query_index in range(low_query.shape[1]):
        speed = coefficient * speed
        state = state + speed
        predictions.append(low_query[:, query_index] + state)
    return np.stack(predictions, axis=1)


def global_rational_residual(
    low_calibration: np.ndarray,
    high_calibration: np.ndarray,
    low_query: np.ndarray,
    *,
    order: int,
    ridge: float = 1e-5,
) -> np.ndarray:
    """Fit an environment-level complex AR(order) rational discrepancy model.

    Coefficients are shared over the dense spatial field but fitted separately
    for each environment.  This is a deliberately training-free low-order
    rational-state oracle, not a learned population model.
    """
    residual = high_calibration - low_calibration
    calibration_count = residual.shape[1]
    if not 1 <= order < calibration_count:
        raise ValueError("order must be positive and smaller than the calibration width")
    outputs = []
    for environment in residual:
        sequence = environment.reshape(calibration_count, -1)
        design_rows, targets = [], []
        for index in range(order, calibration_count):
            design_rows.append(
                np.stack([sequence[index - lag - 1] for lag in range(order)], axis=1)
            )
            targets.append(sequence[index])
        design = np.concatenate(design_rows, axis=0)
        target = np.concatenate(targets, axis=0)
        gram = design.conj().T @ design
        scale = max(float(np.trace(gram).real / order), EPS)
        coefficient = np.linalg.solve(
            gram + ridge * scale * np.eye(order, dtype=gram.dtype),
            design.conj().T @ target,
        )
        history = [sequence[index].copy() for index in range(calibration_count)]
        forecast = []
        for _ in range(low_query.shape[1]):
            state = sum(coefficient[lag] * history[-lag - 1] for lag in range(order))
            history.append(state)
            forecast.append(state.reshape(environment.shape[1:]))
        outputs.append(np.stack(forecast, axis=0))
    return low_query + np.stack(outputs, axis=0)


def load_population_prior(
    path: Path,
    ids: np.ndarray,
    query_frequencies: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as archive:
        if "environment_ids" not in archive.files:
            raise ValueError("population-prior archive requires environment_ids")
        prior_ids = np.asarray(archive["environment_ids"]).astype(str)
        if len(set(prior_ids.tolist())) != len(prior_ids):
            raise ValueError("population-prior environment IDs are not unique")
        if "prediction" in archive.files:
            raw = np.asarray(archive["prediction"])
            frequency_key = "query_frequencies_hz"
        elif "complex_field" in archive.files:
            raw = np.asarray(archive["complex_field"])
            frequency_key = "frequencies_hz"
        else:
            raise ValueError("population-prior archive requires prediction or complex_field")
        if frequency_key not in archive.files:
            raise ValueError(f"population-prior archive requires {frequency_key}")
        frequencies = np.asarray(archive[frequency_key], dtype=np.float64)
        lookup = {float(value): index for index, value in enumerate(frequencies)}
        if any(value not in lookup for value in query_frequencies):
            raise ValueError("population-prior archive does not cover every query")
        id_lookup = {value: index for index, value in enumerate(prior_ids.tolist())}
        if any(value not in id_lookup for value in ids.tolist()):
            raise ValueError("population-prior archive does not cover every paired environment")
        positions = np.asarray([id_lookup[value] for value in ids.tolist()])
        frequency_positions = np.asarray([lookup[value] for value in query_frequencies])
        selected = raw[positions][:, frequency_positions].copy()
    return _to_complex(selected), {"path": str(path), "sha256": sha256_file(path)}


def metric_vectors(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    axes = tuple(range(2, prediction.ndim))
    target_energy = np.sum(np.abs(target) ** 2, axis=axes).clip(min=EPS)
    complex_rel = np.sqrt(np.sum(np.abs(prediction - target) ** 2, axis=axes) / target_energy)
    amplitude_rel = np.sqrt(
        np.sum((np.abs(prediction) - np.abs(target)) ** 2, axis=axes) / target_energy
    )
    phase_difference = np.angle(prediction * np.conj(target))
    weight = np.abs(target)
    weight_sum = np.sum(weight, axis=axes).clip(min=EPS)
    phase_mae = np.sum(weight * np.abs(phase_difference), axis=axes) / weight_sum
    coherence = np.sum(weight * np.cos(phase_difference), axis=axes) / weight_sum
    return {
        "complex_rel_l2": complex_rel,
        "amplitude_rel_l2": amplitude_rel,
        "weighted_phase_mae_rad": phase_mae,
        "phase_coherence": coherence,
    }


def summarize_method(
    prediction: np.ndarray,
    target: np.ndarray,
    query_frequencies: tuple[float, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    vectors = metric_vectors(prediction, target)
    per_query = []
    for index, frequency in enumerate(query_frequencies):
        per_query.append(
            {
                "query_hz": frequency,
                **{metric: float(values[:, index].mean()) for metric, values in vectors.items()},
            }
        )
    macro = {metric: float(values.mean()) for metric, values in vectors.items()}
    return {"macro_107_112": macro, "per_query": per_query}, vectors


def _solver_cost(
    method: str,
    calibration_count: int,
    query_count: int,
) -> dict[str, Any]:
    calibrated = method in {
        "dense_calibration_ar1",
        "damped_velocity_linear_state",
        "global_rational_ar2",
        "global_rational_ar4",
    }
    state_dimension = {
        "dense_calibration_ar1": 1,
        "damped_velocity_linear_state": 2,
        "global_rational_ar2": 2,
        "global_rational_ar4": 4,
    }.get(method, 0)
    return {
        "training_steps": 0,
        "trainable_parameters": 0,
        "cheap_solver_query_calls_per_environment": query_count
        if method != "population_prior"
        else 0,
        "high_fidelity_calibration_sensor_calls_per_environment": calibration_count
        if calibrated
        else 0,
        "high_fidelity_query_sensor_calls_per_environment": 0,
        "evaluation_only_high_fidelity_target_fields_per_environment": query_count,
        "closed_form_complex_fits_per_spatial_location": 1 if calibrated else 0,
        "linear_state_dimension": state_dimension,
    }


def evaluate(
    inputs: PairedAuditInputs, population_prior: np.ndarray | None = None
) -> dict[str, Any]:
    # Predictor calls are deliberately completed before `target` is bound.
    started = time.perf_counter()
    predictions = {
        "cheap_query": inputs.low_query.copy(),
        "dense_calibration_ar1": dense_calibration_ar1_residual(
            inputs.low_calibration, inputs.high_calibration, inputs.low_query
        ),
        "damped_velocity_linear_state": damped_velocity_linear_state_residual(
            inputs.low_calibration, inputs.high_calibration, inputs.low_query
        ),
        "global_rational_ar2": global_rational_residual(
            inputs.low_calibration, inputs.high_calibration, inputs.low_query, order=2
        ),
        "global_rational_ar4": global_rational_residual(
            inputs.low_calibration, inputs.high_calibration, inputs.low_query, order=4
        ),
    }
    if population_prior is not None:
        if population_prior.shape != inputs.low_query.shape:
            raise ValueError("population-prior prediction shape differs from paired queries")
        predictions["population_prior"] = population_prior.copy()
    prediction_seconds = time.perf_counter() - started
    target = inputs.high_query_target

    methods: dict[str, Any] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for method, prediction in predictions.items():
        summary, current_vectors = summarize_method(prediction, target, inputs.query_frequencies)
        summary["solver_cost"] = _solver_cost(
            method, len(inputs.calibration_frequencies), len(inputs.query_frequencies)
        )
        methods[method] = summary
        vectors[method] = current_vectors

    uncalibrated = ["cheap_query"] + (["population_prior"] if population_prior is not None else [])
    best_uncalibrated = min(
        uncalibrated, key=lambda name: methods[name]["macro_107_112"]["complex_rel_l2"]
    )
    calibrated = [
        "dense_calibration_ar1",
        "damped_velocity_linear_state",
        "global_rational_ar2",
        "global_rational_ar4",
    ]
    best_calibrated = min(
        calibrated, key=lambda name: methods[name]["macro_107_112"]["complex_rel_l2"]
    )
    baseline_macro = methods[best_uncalibrated]["macro_107_112"]["complex_rel_l2"]
    dense_macro = methods["dense_calibration_ar1"]["macro_107_112"]["complex_rel_l2"]
    for summary in methods.values():
        macro = summary["macro_107_112"]["complex_rel_l2"]
        summary["oracle_gap"] = {
            "absolute_rel_l2_gap_to_dense_calibration_ar1": float(macro - dense_macro),
            "relative_improvement_vs_best_uncalibrated_percent": float(
                100.0 * (baseline_macro - macro) / max(baseline_macro, EPS)
            ),
        }

    coverage = (107.0, 108.0, 109.0, 110.0)
    query_lookup = {value: index for index, value in enumerate(inputs.query_frequencies)}
    coverage_present = all(value in query_lookup for value in coverage)
    baseline_curve = vectors[best_uncalibrated]["complex_rel_l2"].mean(axis=0)
    candidate_curve = vectors[best_calibrated]["complex_rel_l2"].mean(axis=0)
    per_query_improvement = {
        f"{value:g}": float(
            100.0
            * (baseline_curve[query_lookup[value]] - candidate_curve[query_lookup[value]])
            / max(float(baseline_curve[query_lookup[value]]), EPS)
        )
        for value in coverage
        if value in query_lookup
    }
    macro_improvement = float(
        100.0
        * (baseline_macro - methods[best_calibrated]["macro_107_112"]["complex_rel_l2"])
        / max(baseline_macro, EPS)
    )
    gate = {
        "threshold_relative_improvement_percent": 30.0,
        "required_query_coverage_hz": coverage,
        "best_uncalibrated_method": best_uncalibrated,
        "best_calibrated_method": best_calibrated,
        "macro_relative_improvement_percent": macro_improvement,
        "per_query_relative_improvement_percent": per_query_improvement,
        "coverage_present": coverage_present,
        "every_required_query_meets_threshold": coverage_present
        and all(value >= 30.0 for value in per_query_improvement.values()),
        "fidelity_independent": inputs.fidelity_independent,
    }
    gate["passed"] = bool(
        gate["fidelity_independent"]
        and gate["coverage_present"]
        and gate["macro_relative_improvement_percent"] >= 30.0
        and gate["every_required_query_meets_threshold"]
    )
    gate["forced_failure_reason"] = (
        None
        if inputs.fidelity_independent
        else "same/identical archive is smoke, not multi-fidelity evidence"
    )
    return {
        "methods": methods,
        "task_gate": gate,
        "prediction_wall_seconds_all_methods": prediction_seconds,
    }


def write_outputs(
    output_dir: Path,
    inputs: PairedAuditInputs,
    evaluation: dict[str, Any],
    population_lineage: dict[str, Any] | None,
    command: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    output = {
        "schema_version": 1,
        "status": "training-free paired-fidelity oracle audit",
        "command": command,
        "lineage": {
            "low_archive": {
                "path": str(inputs.low_archive.path),
                "sha256": inputs.low_archive.sha256,
            },
            "high_archive": {
                "path": str(inputs.high_archive.path),
                "sha256": inputs.high_archive.sha256,
            },
            "population_prior": population_lineage,
            "environment_count": len(inputs.ids),
            "environment_ids_sha256": hashlib.sha256(
                "\n".join(inputs.ids.tolist()).encode("utf-8")
            ).hexdigest(),
            "identity_guard": inputs.identity_guard,
        },
        "sensor_access_contract": {
            "high_fidelity_predictor_input_frequencies_hz": inputs.calibration_frequencies,
            "low_fidelity_query_input_frequencies_hz": inputs.query_frequencies,
            "high_fidelity_query_target_frequencies_hz": inputs.query_frequencies,
            "high_fidelity_query_sensor_access": False,
            "prediction_constructed_before_target_binding": True,
        },
        "calibration_frequencies_hz": inputs.calibration_frequencies,
        "query_frequencies_hz": inputs.query_frequencies,
        **evaluation,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    curve_path = output_dir / "query_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "query_hz", *METRICS))
        writer.writeheader()
        for method, method_result in output["methods"].items():
            for row in method_result["per_query"]:
                writer.writerow({"method": method, **row})
            writer.writerow(
                {
                    "method": method,
                    "query_hz": "macro_107_112",
                    **method_result["macro_107_112"],
                }
            )
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "query_curve": str(curve_path),
        "query_curve_sha256": sha256_file(curve_path),
        "task_gate": output["task_gate"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_paired_inputs(
        args.low_archive,
        args.high_archive,
        args.calibration_frequencies,
        args.queries,
        split=args.split,
        allow_same_archive_smoke=args.allow_same_archive_smoke,
    )
    population_prediction = None
    population_lineage = None
    if args.population_prior is not None:
        population_prediction, population_lineage = load_population_prior(
            args.population_prior, inputs.ids, inputs.query_frequencies
        )
    evaluation = evaluate(inputs, population_prediction)
    artifacts = write_outputs(
        args.output_dir,
        inputs,
        evaluation,
        population_lineage,
        command=" ".join(args.command),
    )
    print(json.dumps(artifacts, indent=2), flush=True)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-archive", type=Path, required=True)
    parser.add_argument("--high-archive", type=Path, required=True)
    parser.add_argument("--population-prior", type=Path)
    parser.add_argument(
        "--calibration-frequencies",
        type=parse_frequencies,
        default=parse_frequencies("101,102,103,104,105,106"),
    )
    parser.add_argument(
        "--queries", type=parse_frequencies, default=parse_frequencies("107,108,109,110,111,112")
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--allow-same-archive-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import sys

    args.command = tuple(sys.argv)
    return args


if __name__ == "__main__":
    run(parse_args())
