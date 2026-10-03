"""Classical contrast-source trial space and background-boundary audit.

Known medium/source/PDE and complete context fields. The background inverse is
an explicit classical solve, not a learned decoder or a cost-free operation.
No target field is used except in separately labelled capacity oracles.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .audit_fundamental_solution_rom import gaussian_reference
from .audit_phase_reduced_operator import discrete_operator, projected_query, travel_coordinates
from .coherent_representation import complex_metrics


def background_spec(spec):
    result = json.loads(json.dumps(spec))
    result["contrasts"] = [0.0] * len(result["contrasts"])
    return result


def affine_projection(lap, mass, rhs, background, basis, frequency):
    residual_rhs = rhs - lap @ background - (2 * np.pi * frequency) ** 2 * mass * background
    correction, rank, _ = projected_query(
        lap, mass, residual_rhs, basis, frequency, "least_residual"
    )
    prediction = background + correction
    residual = np.linalg.norm(
        lap @ prediction + (2 * np.pi * frequency) ** 2 * mass * prediction - rhs
    ) / np.linalg.norm(rhs)
    return prediction, rank, float(residual)


def source_trial(lu, contrast, field_trial, frequency):
    """Map candidate total fields to secondary responses through A0^-1."""
    return lu.solve(-((2 * np.pi * frequency) ** 2) * contrast[:, None] * field_trial)


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as z:
            fields, frequencies, axis = z["field"], z["frequencies"], z["axis"]
            spec = json.loads(str(z["spec"]))
        grid = len(axis)
        lap, mass, rhs = discrete_operator(spec, grid)
        _, bg_mass, _ = discrete_operator(background_spec(spec), grid)
        contrast = mass - bg_mass
        coords = travel_coordinates(spec, axis)
        radius = coords["reference_phase"]
        smooth_radius = np.sqrt(radius**2 + spec["source_width"] ** 2)
        tau = coords["straight_ray_medium_phase"] * radius / smooth_radius
        excess = (coords["straight_ray_medium_phase"] - radius) * radius / smooth_radius
        keep = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
        roi = np.outer(keep, keep).ravel()
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        query = np.flatnonzero(np.isin(frequencies, config["query_frequencies"]))
        if set(observed).intersection(query):
            raise ValueError("target observations forbidden")
        snapshots = fields[observed, 1:-1, 1:-1].reshape(len(observed), -1).T
        references = {
            "phase": np.exp(2j * np.pi * tau[:, None] * frequencies[None]),
            "source_aware": np.stack(
                [
                    gaussian_reference(
                        radius, f, spec["source_width"], loss=spec["background_loss"]
                    )
                    for f in frequencies
                ],
                axis=1,
            )
            * np.exp(2j * np.pi * excess[:, None] * frequencies[None]),
        }
        for qi in query:
            frequency = float(frequencies[qi])
            target = fields[qi, 1:-1, 1:-1].ravel()
            start = time.perf_counter()
            lu = splu(lap + sparse.diags((2 * np.pi * frequency) ** 2 * bg_mass, format="csc"))
            factor_seconds = time.perf_counter() - start
            u0 = lu.solve(rhs)
            analytic = gaussian_reference(
                radius, frequency, spec["source_width"], loss=spec["background_loss"]
            )
            rows.append(
                {
                    "system": system,
                    "frequency": frequency,
                    "method": "infinite_reference_vs_discrete_background",
                    "scope": "boundary/discretization diagnostic; NOT heterogeneous prediction",
                    **complex_metrics(analytic[roi], u0[roi]),
                }
            )
            baseline = {
                "background_only": u0,
                "born_one": u0 + source_trial(lu, contrast, u0[:, None], frequency)[:, 0],
            }
            for name, pred in baseline.items():
                rows.append(
                    {
                        "system": system,
                        "frequency": frequency,
                        "method": name,
                        "background_factor_seconds": factor_seconds,
                        **complex_metrics(pred[roi], target[roi]),
                    }
                )
            for name, reference in references.items():
                basis = snapshots / reference[:, observed] * reference[:, qi, None]
                basis = np.column_stack((basis, basis.conj()))
                start = time.perf_counter()
                secondary = source_trial(lu, contrast, basis, frequency)
                inverse_seconds = time.perf_counter() - start
                direct, rank, residual = projected_query(
                    lap, mass, rhs, basis, frequency, "least_residual"
                )
                prediction, rank_s, residual_s = affine_projection(
                    lap, mass, rhs, u0, secondary, frequency
                )
                oracle = (
                    u0
                    + secondary @ np.linalg.lstsq(secondary[roi], (target - u0)[roi], rcond=1e-9)[0]
                )
                for readout, pred, n_rank, resid in [
                    ("field_pde", direct, rank, residual),
                    ("contrast_source_pde", prediction, rank_s, residual_s),
                    ("contrast_source_query_truth_oracle", oracle, rank_s, None),
                ]:
                    rows.append(
                        {
                            "system": system,
                            "frequency": frequency,
                            "method": name + "_" + readout,
                            "rank": n_rank,
                            "pde_relative_residual": resid,
                            "background_factor_seconds": factor_seconds,
                            "background_inverse_basis_seconds": inverse_seconds,
                            "information": "known medium/source/PDE; query truth only in labelled oracle",
                            **complex_metrics(pred[roi], target[roi]),
                        }
                    )
            (output / "results.json").write_text(json.dumps(rows, indent=2))
        group = [r for r in rows if r["system"] == system]
        print(
            json.dumps(
                {
                    "system": system,
                    "errors": {
                        method: float(
                            np.mean([r["complex_rel_l2"] for r in group if r["method"] == method])
                        )
                        for method in sorted({r["method"] for r in group})
                    },
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
