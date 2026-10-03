"""Separate smooth envelope capacity, source enrichment, and physics readout.

All deployable dictionaries use context fields and known medium/source only.
Best-in-span query-truth fits are explicitly labelled as nondeployable oracles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_phase_reduced_operator import discrete_operator, projected_query, travel_coordinates
from .coherent_representation import complex_metrics


def candidate_bases(base, rhs, mass, lap, xx, yy, source, width):
    radius = np.hypot(xx - source[0], yy - source[1]).ravel()
    source_local = rhs / mass
    source_second = (lap @ source_local) / mass
    source_columns = np.stack((source_local, source_second), axis=1)
    yield "base", base
    yield "source_vector", np.column_stack((base, rhs))
    yield "source_mass", np.column_stack((base, source_local))
    yield "source_jet2", np.column_stack((base, source_columns))
    coordinates = np.stack((2 * (xx.ravel() - 0.5), 2 * (yy.ravel() - 0.5)), axis=1)
    polynomial = np.column_stack((np.ones(len(radius)), coordinates))
    yield "linear_envelope", (base[:, :, None] * polynomial[:, None]).reshape(len(radius), -1)
    radial = np.column_stack((np.ones(len(radius)), radius, 1 / np.sqrt(radius**2 + width**2)))
    yield "radial_envelope", (base[:, :, None] * radial[:, None]).reshape(len(radius), -1)
    radial_base = (base[:, :, None] * radial[:, None]).reshape(len(radius), -1)
    yield "radial_plus_source", np.column_stack((radial_base, source_columns))


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as z:
            fields, frequencies, axis = z["field"], z["frequencies"], z["axis"]
            spec = json.loads(str(z["spec"]))
        lap, mass, rhs = discrete_operator(spec, len(axis))
        xx, yy = np.meshgrid(axis[1:-1], axis[1:-1])
        roi1 = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
        roi = np.outer(roi1, roi1).ravel()
        radius = np.hypot(xx - spec["source"][0], yy - spec["source"][1]).ravel()
        masks = {
            "full_roi": roi,
            "source_region": roi & (radius <= 3 * spec["source_width"]),
            "outside_source": roi & (radius > 3 * spec["source_width"]),
        }
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        query = np.flatnonzero(np.isin(frequencies, config["query_frequencies"]))
        if set(observed).intersection(query):
            raise ValueError("target observations forbidden")
        tau = travel_coordinates(spec, axis)["straight_ray_medium_phase"]
        snapshots = fields[observed, 1:-1, 1:-1].reshape(len(observed), -1).T
        for qi in query:
            frequency = float(frequencies[qi])
            target = fields[qi, 1:-1, 1:-1].ravel()
            base = snapshots * np.exp(
                2j * np.pi * tau[:, None] * (frequency - frequencies[observed][None])
            )
            base = np.column_stack((base, base.conj()))
            for name, basis in candidate_bases(
                base, rhs, mass, lap, xx, yy, spec["source"], spec["source_width"]
            ):
                # Conditioning must not depend on arbitrary dictionary column units.
                basis /= np.linalg.norm(basis, axis=0, keepdims=True).clip(1e-30)
                prediction, rank, residual = projected_query(
                    lap, mass, rhs, basis, frequency, "least_residual"
                )
                coefficient = np.linalg.lstsq(basis[roi], target[roi], rcond=1e-9)[0]
                oracle = basis @ coefficient
                for readout, pred in [("pde", prediction), ("query_truth_oracle", oracle)]:
                    for region, keep in masks.items():
                        rows.append(
                            {
                                "system": system,
                                "frequency": frequency,
                                "basis": name,
                                "readout": readout,
                                "region": region,
                                "rank": rank,
                                "pde_relative_residual": residual if readout == "pde" else None,
                                "target_energy_fraction": float(
                                    np.sum(abs(target[keep]) ** 2) / np.sum(abs(target[roi]) ** 2)
                                ),
                                "information": "known medium/source/PDE; query truth only for explicitly named oracle",
                                **complex_metrics(pred[keep], target[keep]),
                            }
                        )
        for name in sorted({r["basis"] for r in rows}):
            subset = [
                r
                for r in rows
                if r["system"] == system and r["basis"] == name and r["region"] == "full_roi"
            ]
            print(
                json.dumps(
                    {
                        "system": system,
                        "basis": name,
                        "macro_error": {
                            readout: float(
                                np.mean(
                                    [r["complex_rel_l2"] for r in subset if r["readout"] == readout]
                                )
                            )
                            for readout in ["pde", "query_truth_oracle"]
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
