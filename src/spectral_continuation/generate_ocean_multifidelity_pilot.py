"""Generate tiny independent formal Pekeris LF/HF frequency-band archives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .ocean_multifidelity_solver import (
    JuliaPekerisRunner,
    SolveResult,
    atomic_write_json,
    field_metrics,
    load_config,
    parse_environments,
    parse_grid,
    parse_solver,
    sha256_file,
)
from .paired_fidelity_protocol import ARCHIVE_SCHEMA, validate_paired_fidelity_archives


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _complex_tensor(results: list[list[SolveResult]]) -> np.ndarray:
    values = np.asarray([[item.field for item in row] for row in results], dtype=np.complex64)
    return np.stack((values.real, values.imag), axis=-1)


def _provenance(
    role: str,
    solver: dict,
    config_path: Path,
    config_sha256: str,
    script_sha256: str,
) -> dict:
    common = {
        "fidelity_level": role,
        "solver_family": "UnderwaterAcoustics.jl/PekerisModeSolver",
        "solver_version": "UnderwaterAcoustics.jl 0.8.1",
        "solver_configuration": solver,
        "generator_script": str(Path(__file__).resolve()),
        "generator_script_sha256": script_sha256,
        "derivation_operations": ["fresh_propagation_solve_on_native_receiver_grid"],
        "parents": [
            {
                "role": "generator_config",
                "path": str(config_path.resolve()),
                "sha256": config_sha256,
            }
        ],
    }
    if role == "low":
        return {
            **common,
            "field_origin": "independent_reduced_order_solver",
            "independent_of_high_target": True,
            "derived_from_high_target": False,
            "independence_detail": (
                "Each field is a fresh nmodes-truncated modal solve. The low pass runs "
                "before the high pass and never receives a high field or archive."
            ),
        }
    return {
        **common,
        "field_origin": "independent_high_fidelity_solver",
    }


def run(config_path: Path, output_dir: Path, candidate_id: str | None) -> dict:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    config = load_config(config_path)
    grid = parse_grid({"receiver_grid": config["pilot"]["receiver_grid"]})
    cases = parse_environments(config, "pilot_environments")
    high = parse_solver(config["high_fidelity"])
    candidates = {
        value["solver_id"]: parse_solver(value) for value in config["low_fidelity_candidates"]
    }
    selected_id = candidate_id or config["pilot"]["selected_candidate"]
    if selected_id not in candidates:
        raise ValueError(f"unknown candidate {selected_id!r}")
    if any(case.source_id is None or case.split is None for case in cases):
        raise ValueError("formal pilot cases require source_id and instance-level split")
    low = candidates[selected_id]
    frequencies = np.asarray(config["pilot"]["frequencies_hz"], dtype=np.float64)
    if not np.array_equal(frequencies, np.arange(101.0, 113.0)):
        raise ValueError("formal tiny pilot is frozen to the 101--112 Hz lattice")

    runner = JuliaPekerisRunner()
    # The low pass is completed independently before any high target is solved.
    low_results = [
        [runner.solve(low, case, float(frequency), grid) for frequency in frequencies]
        for case in cases
    ]
    high_results = [
        [runner.solve(high, case, float(frequency), grid) for frequency in frequencies]
        for case in cases
    ]
    low_tensor = _complex_tensor(low_results)
    high_tensor = _complex_tensor(high_results)

    environment_parameters = np.asarray(
        [
            [
                case.water_speed_mps,
                case.water_depth_m,
                case.seabed_speed_mps,
                case.seabed_density_kgm3,
            ]
            for case in cases
        ],
        dtype=np.float64,
    )
    source_parameters = np.asarray([[case.source_depth_m] for case in cases], dtype=np.float64)
    identity_arrays = {
        "environment_ids": np.asarray([case.case_id for case in cases]),
        "source_ids": np.asarray([case.source_id for case in cases]),
        "split": np.asarray([case.split for case in cases]),
        "frequencies_hz": frequencies,
        "depths_m": np.linspace(
            grid.depth_start_m, float(grid.depth_end_m), grid.depth_steps, dtype=np.float64
        ),
        "ranges_m": np.linspace(
            grid.range_start_m, grid.range_end_m, grid.range_steps, dtype=np.float64
        ),
        "environment_parameters": environment_parameters,
        "environment_parameter_names": np.asarray(
            [
                "water_speed_mps",
                "water_depth_m",
                "seabed_speed_mps",
                "seabed_density_kgm3",
            ]
        ),
        "source_parameters": source_parameters,
        "source_parameter_names": np.asarray(["source_depth_m"]),
        "source_coordinates": np.stack(
            (np.zeros(len(cases), dtype=np.float64), -source_parameters[:, 0]), axis=1
        ),
    }
    output_dir.mkdir(parents=True)
    low_path, high_path = output_dir / "low.npz", output_dir / "high.npz"
    config_sha = sha256_file(config_path)
    script_sha = sha256_file(Path(__file__))
    atomic_savez(
        low_path,
        schema_version=np.asarray(ARCHIVE_SCHEMA),
        fidelity_role=np.asarray("low"),
        complex_field=low_tensor,
        provenance_json=np.asarray(
            json.dumps(_provenance("low", low.__dict__, config_path, config_sha, script_sha))
        ),
        **identity_arrays,
    )
    atomic_savez(
        high_path,
        schema_version=np.asarray(ARCHIVE_SCHEMA),
        fidelity_role=np.asarray("high"),
        complex_field=high_tensor,
        provenance_json=np.asarray(
            json.dumps(_provenance("high", high.__dict__, config_path, config_sha, script_sha))
        ),
        **identity_arrays,
    )
    pair = validate_paired_fidelity_archives(low_path, high_path)

    rows = []
    for case_index, case in enumerate(cases):
        for frequency_index, frequency in enumerate(frequencies):
            low_result = low_results[case_index][frequency_index]
            high_result = high_results[case_index][frequency_index]
            rows.append(
                {
                    "case_id": case.case_id,
                    "source_id": case.source_id,
                    "split": case.split,
                    "frequency_hz": float(frequency),
                    "high_elapsed_seconds": high_result.elapsed_seconds,
                    "low_elapsed_seconds": low_result.elapsed_seconds,
                    "high_allocated_bytes": high_result.allocated_bytes,
                    "low_allocated_bytes": low_result.allocated_bytes,
                    "high_component_count": high_result.component_count,
                    "low_component_count": low_result.component_count,
                    **field_metrics(low_result.field, high_result.field),
                }
            )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "scope": "tiny formal development pilot; not a paper-scale dataset",
        "low_archive": str(low_path.resolve()),
        "low_archive_sha256": pair.low.archive_sha256,
        "high_archive": str(high_path.resolve()),
        "high_archive_sha256": pair.high.archive_sha256,
        "paired_identity_sha256": pair.identity_sha256,
        "formal_schema": ARCHIVE_SCHEMA,
        "formal_pair_validation": "passed validate_paired_fidelity_archives",
        "config": str(config_path.resolve()),
        "config_sha256": config_sha,
        "script_sha256": script_sha,
        "frequencies_hz": frequencies.tolist(),
        "case_count": len(cases),
        "split_counts": {
            split: int(np.sum(identity_arrays["split"] == split))
            for split in ("train", "validation", "test")
        },
        "field_shape": list(low_tensor.shape),
        "low_fidelity": low.__dict__,
        "high_fidelity": high.__dict__,
        "independence_contract": (
            "LF and HF are separate fresh solver passes on identical native grids; no "
            "decimation, interpolation, filtering, or access to HF occurs in the LF pass"
        ),
        "rows": rows,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir, args.candidate), indent=2))


if __name__ == "__main__":
    main()
