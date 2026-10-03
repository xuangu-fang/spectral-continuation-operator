"""Known-medium, target-free frequency queries with small physics-projected bases.

A deliberately minimal classical control inspired by phase-preconditioned ROMs,
not a reproduction or novelty claim for full PPRKS. Unlike earlier unknown-system
interpolation, this experiment explicitly receives the medium, source and PDE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline

from .coherent_helmholtz_data import sample_environment
from .coherent_representation import complex_metrics


def discrete_operator(spec, grid):
    axis = np.linspace(0, 1, grid)
    speed, loss, source = sample_environment(spec, axis)
    h, n = 1 / (grid - 1), grid - 2
    d2 = (
        sparse.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="csc")
        / h**2
    )
    d4 = d2 - h * h / 12 * (d2 @ d2)
    lap = sparse.kronsum(d4, d4, format="csc")
    mass = ((1 + 1j * loss[1:-1, 1:-1]) / speed[1:-1, 1:-1]).ravel() ** 2
    rhs = -source[1:-1, 1:-1].ravel().astype(np.complex128)
    return lap, mass, rhs


def travel_coordinates(spec, axis, include_eikonal=False):
    xx, yy = np.meshgrid(axis[1:-1], axis[1:-1])
    source = np.asarray(spec["source"])
    displacement = np.stack((xx - source[0], yy - source[1]), -1)
    distance = np.linalg.norm(displacement, axis=-1)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    integral = np.zeros_like(distance)
    for node, weight in zip((nodes + 1) / 2, weights / 2, strict=True):
        coordinates = source + node * displacement
        speed = np.ones_like(distance)
        for center, width, contrast in zip(
            spec["centers"], spec["widths"], spec["contrasts"], strict=True
        ):
            speed -= contrast * np.exp(-np.sum((coordinates - center) ** 2, -1) / (2 * width**2))
        integral += weight / speed
    result = {
        "raw": np.zeros(distance.size),
        "reference_phase": distance.ravel(),
        "straight_ray_medium_phase": (distance * integral).ravel(),
    }
    if include_eikonal:
        import skfmm

        full_x, full_y = np.meshgrid(axis, axis)
        full_distance = np.hypot(full_x - source[0], full_y - source[1])
        h = axis[1] - axis[0]
        phi = full_distance - 2 * h
        speed, _, _ = sample_environment(spec, axis)
        source_speed = 1.0
        for center, width, contrast in zip(
            spec["centers"], spec["widths"], spec["contrasts"], strict=True
        ):
            source_speed -= contrast * np.exp(-np.sum((source - center) ** 2) / (2 * width**2))
        # Remove the homogeneous finite-radius / source-grid bias. This is a
        # first-arrival phase estimate, not a full multipath eikonal solution.
        reference_time = skfmm.travel_time(phi, np.ones_like(speed), dx=(h, h), order=2)
        medium_time = skfmm.travel_time(phi, speed, dx=(h, h), order=2)
        tau = medium_time + (full_distance - reference_time) / source_speed
        result["eikonal_phase"] = tau[1:-1, 1:-1].ravel()
    return result


def projected_query(lap, mass, rhs, basis, frequency, rule):
    # SVD is a conditioning step using basis vectors only, not query solutions.
    left, singular, _ = np.linalg.svd(basis, full_matrices=False)
    rank = max(1, int(np.sum(singular > singular[0] * 1e-9)))
    v = left[:, :rank]
    av = lap @ v + (2 * np.pi * frequency) ** 2 * mass[:, None] * v
    if rule == "least_residual":
        coefficient = np.linalg.lstsq(av, rhs, rcond=1e-10)[0]
    elif rule == "galerkin":
        coefficient = np.linalg.solve(v.conj().T @ av, v.conj().T @ rhs)
    else:
        raise ValueError(rule)
    pred = v @ coefficient
    residual = np.linalg.norm(
        lap @ pred + (2 * np.pi * frequency) ** 2 * mass * pred - rhs
    ) / np.linalg.norm(rhs)
    return pred, rank, float(residual)


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as z:
            fields, all_f, axis = z["field"], z["frequencies"], z["axis"]
            spec = json.loads(str(z["spec"]))
        lap, mass, rhs = discrete_operator(spec, len(axis))
        coordinates = travel_coordinates(spec, axis, config.get("include_eikonal", False))
        if config.get("phase_names"):
            coordinates = {k: v for k, v in coordinates.items() if k in config["phase_names"]}
        keep = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
        roi = np.outer(keep, keep).ravel()
        for observed_f in config["observed_frequencies"]:
            observed = np.flatnonzero(np.isin(all_f, observed_f))
            query = np.flatnonzero(
                ~np.isin(all_f, observed_f)
                & (all_f >= min(observed_f))
                & (all_f <= max(observed_f))
            )
            snapshots = (
                fields[observed, 1:-1, 1:-1].reshape(len(observed), -1).T.astype(np.complex128)
            )
            for name, tau in coordinates.items():
                envelope = snapshots * np.exp(-2j * np.pi * tau[:, None] * all_f[observed][None])
                for qi in query:
                    frequency = float(all_f[qi])
                    target = fields[qi, 1:-1, 1:-1].ravel()
                    basis = envelope * np.exp(2j * np.pi * frequency * tau[:, None])
                    # Same rank doubling for all variants; conjugate components are
                    # a generic trial-space enrichment, not a new physical source.
                    if config.get("conjugate_enrichment", True):
                        basis = np.concatenate((basis, basis.conj()), axis=1)
                    if config.get("capacity_oracle", False):
                        coefficient = np.linalg.lstsq(basis[roi], target[roi], rcond=1e-9)[0]
                        rows.append(
                            {
                                "system": system,
                                "observed_frequencies": observed_f,
                                "query_frequency": frequency,
                                "method": f"{name}_query_truth_oracle",
                                "information": "USES QUERY TRUTH; basis capacity only, not deployable",
                                **complex_metrics((basis @ coefficient)[roi], target[roi]),
                            }
                        )
                    for rule in ("galerkin", "least_residual"):
                        start = time.perf_counter()
                        prediction, rank, residual = projected_query(
                            lap, mass, rhs, basis, frequency, rule
                        )
                        row = {
                            "system": system,
                            "observed_frequencies": observed_f,
                            "query_frequency": frequency,
                            "method": f"{name}_{rule}",
                            "rank": rank,
                            "pde_relative_residual": residual,
                            "projection_seconds": time.perf_counter() - start,
                            "information": "known medium/source/PDE plus complete other-frequency snapshots",
                            **complex_metrics(prediction[roi], target[roi]),
                        }
                        rows.append(row)
                    interpolated = CubicSpline(all_f[observed], envelope, axis=1)(
                        frequency
                    ) * np.exp(2j * np.pi * frequency * tau)
                    rows.append(
                        {
                            "system": system,
                            "observed_frequencies": observed_f,
                            "query_frequency": frequency,
                            "method": f"{name}_cubic",
                            "information": "same known medium/source, no PDE projection",
                            **complex_metrics(interpolated[roi], target[roi]),
                        }
                    )
                for rule in ("galerkin", "least_residual", "cubic"):
                    current = [
                        r
                        for r in rows
                        if r["system"] == system
                        and r["observed_frequencies"] == observed_f
                        and r["method"] == f"{name}_{rule}"
                    ]
                    print(
                        json.dumps(
                            {
                                "system": system,
                                "context_count": len(observed),
                                "method": f"{name}_{rule}",
                                "macro_error": float(
                                    np.mean([r["complex_rel_l2"] for r in current])
                                ),
                            }
                        ),
                        flush=True,
                    )
                (output / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
