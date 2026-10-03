"""Accuracy/cost audit of a classical tensor-product source decoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .audit_contrast_source_rom import affine_projection
from .audit_phase_reduced_operator import discrete_operator
from .coherent_representation import complex_metrics
from .evaluate_continuous_phase_rom import reference_carriers
from .separable_background import SeparableInverse, one_dimensional_background


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as z:
            spec = json.loads(str(z["spec"]))
            fields, frequencies, axis = z["field"], z["frequencies"], z["axis"]
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        if np.isin(config["query_frequencies"], frequencies[observed]).any():
            raise ValueError("observed target")
        reference, _ = reference_carriers(spec, axis, frequencies, True, True)
        envelopes = (
            fields[observed, 1:-1, 1:-1].reshape(len(observed), -1).T / reference[:, observed]
        )
        lap, mass, rhs = discrete_operator(spec, len(axis))
        keep = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
        roi = np.outer(keep, keep).ravel()
        for f in config["query_frequencies"]:
            qi = int(np.flatnonzero(frequencies == f)[0])
            target = fields[qi, 1:-1, 1:-1].ravel()
            one_d, bg_mass = one_dimensional_background(spec, len(axis), f)
            inverse = SeparableInverse(one_d, config.get("device", "cuda"))
            u0 = inverse.solve(rhs)
            check = np.linalg.norm(
                lap @ u0 + (2 * np.pi * f) ** 2 * bg_mass * u0 - rhs
            ) / np.linalg.norm(rhs)
            basis = envelopes * reference[:, qi, None]
            basis = np.column_stack((basis, basis.conj()))
            # All background differences, including corner absorption, stay here.
            sources = -((2 * np.pi * f) ** 2) * (mass - bg_mass)[:, None] * basis
            start = time.perf_counter()
            propagated = inverse.solve(sources)
            inverse_seconds = time.perf_counter() - start
            prediction, rank, residual = affine_projection(lap, mass, rhs, u0, propagated, f)
            row = {
                "system": system,
                "frequency": f,
                "rank": rank,
                "background_setup_seconds": inverse.setup_seconds,
                "six_rhs_inverse_seconds_including_transfers": inverse_seconds,
                "eigenvector_condition": inverse.eigenvector_condition,
                "background_relative_residual": float(check),
                "pde_relative_residual": residual,
                "scope": "classical separable decoder; GPU cost cannot directly be compared to CPU sparse timing; target PDE unchanged",
                **complex_metrics(prediction[roi], target[roi]),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            (output / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
