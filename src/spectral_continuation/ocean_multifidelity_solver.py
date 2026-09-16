"""Independent Pekeris LF/HF solves for the ocean multi-fidelity pilot.

The low-fidelity field is always produced by its own propagation-model call on
the same physical receiver grid as the high-fidelity field.  This module never
constructs LF data by decimating or filtering a high-fidelity array.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    range_start_m: float
    range_end_m: float
    range_steps: int
    depth_start_m: float
    depth_steps: int
    depth_end_fraction: float | None = None
    depth_end_m: float | None = None


@dataclass(frozen=True)
class EnvironmentCase:
    case_id: str
    water_speed_mps: float
    water_depth_m: float
    seabed_speed_mps: float
    seabed_density_kgm3: float
    source_depth_m: float
    source_id: str | None = None
    split: str | None = None


@dataclass(frozen=True)
class SolverSpec:
    solver_id: str
    family: str
    nmodes: int = 0
    ngrid: int = 0
    max_bounces: int = 3


@dataclass(frozen=True)
class SolveResult:
    field: np.ndarray
    elapsed_seconds: float
    allocated_bytes: int
    gc_seconds: float
    component_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def field_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Raw complex RelL2 and target-amplitude-weighted phase metrics."""
    candidate = np.asarray(candidate, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("candidate and reference must be same-shape 2D complex fields")
    denominator = max(float(np.sum(np.abs(reference) ** 2)), np.finfo(float).eps)
    rel_l2 = float(np.sqrt(np.sum(np.abs(candidate - reference) ** 2) / denominator))
    phase_difference = np.angle(candidate * np.conj(reference))
    weights = np.abs(reference)
    weight_sum = max(float(weights.sum()), np.finfo(float).eps)
    return {
        "complex_rel_l2": rel_l2,
        "phase_coherence": float(np.sum(weights * np.cos(phase_difference)) / weight_sum),
        "weighted_phase_mae_rad": float(np.sum(weights * np.abs(phase_difference)) / weight_sum),
        "amplitude_rel_l2": float(
            np.sqrt(np.sum((np.abs(candidate) - np.abs(reference)) ** 2) / denominator)
        ),
    }


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("expected schema_version=1")
    if config["high_fidelity"]["family"] != "pekeris_mode":
        raise ValueError("the reference must be an independently solved Pekeris mode field")
    candidate_ids = [item["solver_id"] for item in config["low_fidelity_candidates"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate low-fidelity solver IDs")
    return config


def parse_grid(config: dict[str, Any]) -> GridSpec:
    grid = GridSpec(**config["receiver_grid"])
    if grid.range_steps < 2 or grid.depth_steps < 2:
        raise ValueError("the pilot requires a 2D receiver grid")
    if (grid.depth_end_fraction is None) == (grid.depth_end_m is None):
        raise ValueError("set exactly one of depth_end_fraction and depth_end_m")
    if grid.depth_end_fraction is not None and not 0 < grid.depth_end_fraction < 1:
        raise ValueError("depth_end_fraction must be in (0, 1)")
    if grid.depth_end_m is not None and grid.depth_end_m <= grid.depth_start_m:
        raise ValueError("depth_end_m must exceed depth_start_m")
    return grid


def parse_environments(config: dict[str, Any], section: str) -> list[EnvironmentCase]:
    cases = [EnvironmentCase(**item) for item in config[section]]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError(f"{section} must contain unique nonempty case IDs")
    for case in cases:
        if not 0 < case.source_depth_m < case.water_depth_m:
            raise ValueError(f"invalid source depth in {case.case_id}")
        if case.seabed_speed_mps <= case.water_speed_mps:
            raise ValueError(f"{case.case_id} has no guided Pekeris modes")
    return cases


def parse_solver(value: dict[str, Any]) -> SolverSpec:
    spec = SolverSpec(**value)
    if spec.family not in {"pekeris_mode", "pekeris_ray"}:
        raise ValueError(f"unsupported solver family: {spec.family}")
    if spec.nmodes < 0 or spec.ngrid < 0 or spec.max_bounces < 0:
        raise ValueError("solver fidelity controls must be nonnegative")
    if spec.ngrid in {1, 2}:
        raise ValueError("Pekeris mode ngrid must be 0 or greater than 2")
    return spec


class JuliaPekerisRunner:
    """Thin lazy Julia bridge; importing this module does not require JuliaCall."""

    def __init__(self) -> None:
        from juliacall import Main as jl  # type: ignore

        self.jl = jl
        jl.seval("using UnderwaterAcoustics")
        jl.seval(
            """
            function __sco_pekeris_field(
                family, nmodes, ngrid, max_bounces, frequency_hz,
                water_speed, water_depth, seabed_speed, seabed_density,
                source_depth, range_start, range_end, range_steps,
                depth_start, depth_end, depth_steps)
              env = UnderwaterEnvironment(
                bathymetry=water_depth,
                seabed=FluidBoundary(seabed_density, seabed_speed, 0.0),
                soundspeed=water_speed)
              tx = AcousticSource(0.0, -source_depth, frequency_hz)
              ranges = LinRange(range_start, range_end, range_steps)
              depths = LinRange(depth_start, depth_end, depth_steps)
              receivers = [AcousticReceiver(r, -z) for z in depths, r in ranges]
              GC.gc()
              timed = @timed begin
                model = if family == "pekeris_mode"
                  PekerisModeSolver(env; nmodes=nmodes, ngrid=ngrid)
                elseif family == "pekeris_ray"
                  PekerisRayTracer(env; max_bounces=max_bounces)
                else
                  error("unknown Pekeris family")
                end
                values = acoustic_field(model, tx, receivers; mode=:coherent)
                count = family == "pekeris_mode" ? length(arrivals(model, tx)) : 1 + 2max_bounces
                (values, count)
              end
              values, count = timed.value
              return values, Float64[timed.time, timed.bytes, timed.gctime, count]
            end
            """
        )

    def solve(
        self,
        solver: SolverSpec,
        case: EnvironmentCase,
        frequency_hz: float,
        grid: GridSpec,
    ) -> SolveResult:
        depth_end = (
            grid.depth_end_m
            if grid.depth_end_m is not None
            else float(grid.depth_end_fraction) * case.water_depth_m
        )
        solve_field = getattr(self.jl, "__sco_pekeris_field")
        raw_field, raw_meta = solve_field(
            solver.family,
            solver.nmodes,
            solver.ngrid,
            solver.max_bounces,
            frequency_hz,
            case.water_speed_mps,
            case.water_depth_m,
            case.seabed_speed_mps,
            case.seabed_density_kgm3,
            case.source_depth_m,
            grid.range_start_m,
            grid.range_end_m,
            grid.range_steps,
            grid.depth_start_m,
            depth_end,
            grid.depth_steps,
        )
        field = np.asarray(raw_field, dtype=np.complex128)
        metadata = np.asarray(raw_meta, dtype=np.float64)
        expected = (grid.depth_steps, grid.range_steps)
        if field.shape != expected or not np.isfinite(field).all():
            raise RuntimeError(f"invalid solver field {field.shape}; expected {expected}")
        return SolveResult(
            field=field,
            elapsed_seconds=float(metadata[0]),
            allocated_bytes=int(metadata[1]),
            gc_seconds=float(metadata[2]),
            component_count=int(metadata[3]),
        )
