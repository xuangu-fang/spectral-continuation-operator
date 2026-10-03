"""Strong classical same-physics control before claiming learned ROM utility."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .audit_contrast_source_rom import background_spec
from .audit_phase_reduced_operator import discrete_operator, projected_query
from .coherent_representation import complex_metrics
from .evaluate_continuous_phase_rom import reference_carriers


def right_preconditioned_iterations(matrix, inverse, rhs, initial, iterations):
    """Unrestarted right GMRES with double MGS and explicit application counts."""
    residual = rhs - matrix @ initial
    beta = np.linalg.norm(residual)
    yield 0, initial.copy(), float(beta / np.linalg.norm(rhs))
    if beta == 0:
        return
    vectors = [residual / beta]
    z = []
    hessenberg = np.zeros((iterations + 1, iterations), dtype=complex)
    for j in range(iterations):
        current = inverse(vectors[j])
        z.append(current)
        w = matrix @ current
        for _ in range(2):
            for i in range(j + 1):
                value = np.vdot(vectors[i], w)
                hessenberg[i, j] += value
                w -= value * vectors[i]
        magnitude = np.linalg.norm(w)
        hessenberg[j + 1, j] = magnitude
        right = np.zeros(j + 2, dtype=complex)
        right[0] = beta
        coefficient = np.linalg.lstsq(hessenberg[: j + 2, : j + 1], right, rcond=1e-12)[0]
        prediction = initial + np.column_stack(z) @ coefficient
        relative = np.linalg.norm(rhs - matrix @ prediction) / np.linalg.norm(rhs)
        yield j + 1, prediction, float(relative)
        if magnitude < 1e-13 or relative < 1e-12:
            break
        vectors.append(w / magnitude)


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as z:
            fields, frequencies, axis = z["field"], z["frequencies"], z["axis"]
            spec = json.loads(str(z["spec"]))
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        if np.isin(config["query_frequencies"], frequencies[observed]).any():
            raise ValueError("observed target")
        reference, _ = reference_carriers(spec, axis, frequencies, True, True)
        envelope = (
            fields[observed, 1:-1, 1:-1].reshape(len(observed), -1).T / reference[:, observed]
        )
        lap, mass, rhs = discrete_operator(spec, len(axis))
        _, bg_mass, _ = discrete_operator(background_spec(spec), len(axis))
        xx, yy = np.meshgrid(axis[1:-1], axis[1:-1])
        roi = (
            (xx >= config["roi"][0])
            & (xx <= config["roi"][1])
            & (yy >= config["roi"][0])
            & (yy <= config["roi"][1])
        ).ravel()
        near = (
            np.hypot(xx - spec["source"][0], yy - spec["source"][1]) <= 3 * spec["source_width"]
        ).ravel()
        for f in config["query_frequencies"]:
            qi = int(np.flatnonzero(frequencies == f)[0])
            target = fields[qi, 1:-1, 1:-1].ravel()
            matrix = lap + sparse.diags((2 * np.pi * f) ** 2 * mass, format="csc")
            start = time.perf_counter()
            lu = splu(lap + sparse.diags((2 * np.pi * f) ** 2 * bg_mass, format="csc"))
            setup_seconds = time.perf_counter() - start
            basis = envelope * reference[:, qi, None]
            basis = np.column_stack((basis, basis.conj()))
            warm, _, _ = projected_query(lap, mass, rhs, basis, f, "least_residual")
            for label, initial in [("zero", np.zeros_like(rhs)), ("source_aware_field_rom", warm)]:
                start = time.perf_counter()
                for applications, prediction, residual in right_preconditioned_iterations(
                    matrix, lu.solve, rhs, initial, max(config["application_budgets"])
                ):
                    if applications not in config["application_budgets"]:
                        continue
                    row = {
                        "system": system,
                        "frequency": f,
                        "initialization": label,
                        "background_applications": applications,
                        "setup_seconds": setup_seconds,
                        "cumulative_iteration_seconds": time.perf_counter() - start,
                        "pde_relative_residual": residual,
                        "near_source_target_energy_fraction": float(
                            np.sum(abs(target[roi & near]) ** 2) / np.sum(abs(target[roi]) ** 2)
                        ),
                        "far_region_error": complex_metrics(
                            prediction[roi & ~near], target[roi & ~near]
                        ),
                        "scope": "classical known-PDE benchmark, no query truth used in iteration; full ROI primary preserved",
                        **complex_metrics(prediction[roi], target[roi]),
                    }
                    rows.append(row)
                group = [
                    r
                    for r in rows
                    if r["system"] == system
                    and r["frequency"] == f
                    and r["initialization"] == label
                ]
                print(
                    json.dumps(
                        {
                            "system": system,
                            "frequency": f,
                            "initialization": label,
                            "errors_by_applications": {
                                r["background_applications"]: r["complex_rel_l2"] for r in group
                            },
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
