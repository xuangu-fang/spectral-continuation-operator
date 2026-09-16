"""Benchmark independently solved Pekeris low/high-fidelity field pairs."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from .ocean_multifidelity_solver import (
    EnvironmentCase,
    GridSpec,
    JuliaPekerisRunner,
    atomic_write_json,
    field_metrics,
    load_config,
    parse_environments,
    parse_grid,
    parse_solver,
    sha256_file,
)


def aggregate(rows: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row["candidate_id"] == candidate_id]
    keys = (
        "complex_rel_l2",
        "phase_coherence",
        "weighted_phase_mae_rad",
        "amplitude_rel_l2",
        "wall_time_speedup",
        "allocation_reduction",
    )
    return {
        "candidate_id": candidate_id,
        "environment_count": len(selected),
        **{f"median_{key}": float(np.median([row[key] for row in selected])) for key in keys},
        **{f"worst_{key}": float(max(row[key] for row in selected)) for key in keys[:1]},
        "minimum_phase_coherence": float(min(row["phase_coherence"] for row in selected)),
        "minimum_wall_time_speedup": float(min(row["wall_time_speedup"] for row in selected)),
        "minimum_allocation_reduction": float(min(row["allocation_reduction"] for row in selected)),
    }


def passes_gate(summary: dict[str, Any], gate: dict[str, float]) -> bool:
    return bool(
        summary["median_wall_time_speedup"] >= gate["median_wall_time_speedup_min"]
        and summary["median_allocation_reduction"] >= gate["median_allocation_reduction_min"]
        and summary["median_complex_rel_l2"] <= gate["median_complex_rel_l2_max"]
        and summary["median_phase_coherence"] >= gate["median_phase_coherence_min"]
    )


def reproduce_legacy_high_fidelity(
    runner: JuliaPekerisRunner,
    high,
    legacy: dict[str, Any],
) -> dict[str, Any]:
    """Recompute one published HF field with the same solver and native grid."""
    shard_path = Path(legacy["shard"]).resolve()
    frequency_hz = float(legacy["frequency_hz"])
    with np.load(shard_path, allow_pickle=False) as archive:
        names = np.asarray(archive["environment_parameter_names"]).astype(str)
        parameters = np.asarray(archive["environment_parameters"], dtype=np.float64)
        lookup = dict(zip(names.tolist(), parameters.tolist()))
        frequencies = np.asarray(archive["frequencies_hz"], dtype=np.float64)
        matches = np.flatnonzero(np.isclose(frequencies, frequency_hz))
        if len(matches) != 1:
            raise RuntimeError("legacy frequency is missing or ambiguous")
        archived_tensor = np.asarray(archive["complex_field"][int(matches[0])], dtype=np.float32)
        ranges = np.asarray(archive["ranges_m"], dtype=np.float64)
        depths = np.asarray(archive["depths_m"], dtype=np.float64)
        case_id = str(archive["environment_id"])
    case = EnvironmentCase(
        case_id=case_id,
        water_speed_mps=lookup["water_speed_mps"],
        water_depth_m=lookup["water_depth_m"],
        seabed_speed_mps=lookup["seabed_speed_mps"],
        seabed_density_kgm3=lookup["seabed_density_kgm3"],
        source_depth_m=lookup["source_depth_m"],
    )
    grid = GridSpec(
        range_start_m=float(ranges[0]),
        range_end_m=float(ranges[-1]),
        range_steps=len(ranges),
        depth_start_m=float(depths[0]),
        depth_end_m=float(depths[-1]),
        depth_steps=len(depths),
    )
    fresh = runner.solve(high, case, frequency_hz, grid)
    archived = archived_tensor[..., 0].astype(np.float64) + 1j * archived_tensor[..., 1]
    fresh_float32 = np.stack((fresh.field.real, fresh.field.imag), axis=-1).astype(np.float32)
    metrics = field_metrics(fresh.field, archived)
    passed = bool(
        metrics["complex_rel_l2"] <= float(legacy["complex_rel_l2_max"])
        and metrics["phase_coherence"] >= float(legacy["phase_coherence_min"])
    )
    return {
        "passed": passed,
        "published_shard": str(shard_path),
        "published_shard_sha256": sha256_file(shard_path),
        "case_id": case_id,
        "frequency_hz": frequency_hz,
        "solver": high.__dict__,
        "grid_shape": list(fresh.field.shape),
        "float32_bitwise_equal": bool(np.array_equal(fresh_float32, archived_tensor)),
        "maximum_float32_absolute_difference": float(
            np.max(np.abs(fresh_float32 - archived_tensor))
        ),
        **metrics,
        "tolerances": {
            "complex_rel_l2_max": float(legacy["complex_rel_l2_max"]),
            "phase_coherence_min": float(legacy["phase_coherence_min"]),
        },
    }


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    grid = parse_grid(config)
    cases = parse_environments(config, "audit_environments")
    high = parse_solver(config["high_fidelity"])
    candidates = [parse_solver(item) for item in config["low_fidelity_candidates"]]
    repetitions = int(config["benchmark"]["repetitions"])
    if repetitions < 2:
        raise ValueError("at least two timed repetitions are required")
    runner = JuliaPekerisRunner()
    legacy_reproduction = reproduce_legacy_high_fidelity(
        runner, high, config["legacy_high_fidelity_reproduction"]
    )

    # Compile every solver family/configuration on a deliberately tiny grid.
    warm_grid = type(grid)(
        range_start_m=grid.range_start_m,
        range_end_m=min(grid.range_end_m, grid.range_start_m + 500.0),
        range_steps=4,
        depth_start_m=grid.depth_start_m,
        depth_steps=4,
        depth_end_fraction=grid.depth_end_fraction,
        depth_end_m=grid.depth_end_m,
    )
    for solver in [high, *candidates]:
        runner.solve(solver, cases[0], float(config["frequency_hz"]), warm_grid)

    rows: list[dict[str, Any]] = []
    for case in cases:
        high_runs = [
            runner.solve(high, case, float(config["frequency_hz"]), grid)
            for _ in range(repetitions)
        ]
        high_time = float(np.median([item.elapsed_seconds for item in high_runs]))
        high_bytes = float(np.median([item.allocated_bytes for item in high_runs]))
        reference = high_runs[-1].field
        for candidate in candidates:
            low_runs = [
                runner.solve(candidate, case, float(config["frequency_hz"]), grid)
                for _ in range(repetitions)
            ]
            low_time = float(np.median([item.elapsed_seconds for item in low_runs]))
            low_bytes = float(np.median([item.allocated_bytes for item in low_runs]))
            rows.append(
                {
                    "case_id": case.case_id,
                    "water_depth_m": case.water_depth_m,
                    "candidate_id": candidate.solver_id,
                    "candidate_family": candidate.family,
                    "high_component_count": high_runs[-1].component_count,
                    "low_component_count": low_runs[-1].component_count,
                    "high_elapsed_seconds_median": high_time,
                    "low_elapsed_seconds_median": low_time,
                    "wall_time_speedup": high_time / max(low_time, np.finfo(float).eps),
                    "high_allocated_bytes_median": high_bytes,
                    "low_allocated_bytes_median": low_bytes,
                    "allocation_reduction": high_bytes / max(low_bytes, 1.0),
                    **field_metrics(low_runs[-1].field, reference),
                }
            )

    summaries = [aggregate(rows, candidate.solver_id) for candidate in candidates]
    gate = config["go_gate"]
    for summary in summaries:
        summary["passes_go_gate"] = passes_gate(summary, gate)
    eligible = [item for item in summaries if item["passes_go_gate"]]
    recommended = (
        min(eligible, key=lambda item: item["median_complex_rel_l2"])["candidate_id"]
        if eligible
        else None
    )
    payload = {
        "schema_version": 1,
        "status": "GO" if recommended and legacy_reproduction["passed"] else "NO-GO",
        "scope": "development-only same-instance same-frequency microbenchmark",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__)),
        "frequency_hz": float(config["frequency_hz"]),
        "receiver_grid": config["receiver_grid"],
        "benchmark_repetitions": repetitions,
        "measurement_contract": {
            "wall_time": "Julia @timed around fresh model construction and complex-field solve; JIT warmed separately",
            "memory": "Julia heap bytes allocated by the same timed solve; this is allocation volume, not peak RSS",
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "lf_independence": "every LF field is a fresh propagation solve on the native HF receiver grid; no HF array is read by LF",
        },
        "high_fidelity": config["high_fidelity"],
        "legacy_high_fidelity_reproduction": legacy_reproduction,
        "go_gate": gate,
        "candidate_summaries": summaries,
        "recommended_candidate": recommended if legacy_reproduction["passed"] else None,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "pid": os.getpid(),
    }
    atomic_write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
