"""Value capacity versus differential stability under fine-grid transfer.

Coarse PDE coefficients are transferred without query truth, whereas a separate
target fit provides only a nondeployable capacity oracle. Near-source smoothing
tests one analytic hypothesis, with the same six-column dictionary budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_phase_reduced_operator import discrete_operator, projected_query
from .coherent_representation import complex_metrics
from .evaluate_continuous_phase_rom import interpolate_envelopes, reference_carriers


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["context_data"]) / f"system_{system}_grid257.npz") as z:
            old_axis = z["axis"]
            spec = json.loads(str(z["spec"]))
            idx = np.flatnonzero(np.isin(z["frequencies"], config["observed_frequencies"]))
            observed_f = z["frequencies"][idx]
            snapshots = z["field"][idx, 1:-1, 1:-1].reshape(len(idx), -1).T
        with np.load(Path(config["query_data"]) / f"system_{system}_grid513.npz") as z:
            axis = z["axis"]
            ids = np.flatnonzero(np.isin(z["frequencies"], config["query_frequencies"]))
            frequencies, targets = z["frequencies"][ids], z["field"][ids, 1:-1, 1:-1]
        if np.isin(frequencies, observed_f).any():
            raise ValueError("query observations forbidden")
        old_lap, old_mass, old_rhs = discrete_operator(spec, len(old_axis))
        lap, mass, rhs = discrete_operator(spec, len(axis))
        x, y = np.meshgrid(axis[1:-1], axis[1:-1])
        radius = np.hypot(x - spec["source"][0], y - spec["source"][1]).ravel()
        near_source = radius <= 3 * spec["source_width"]
        roi1 = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
        roi = np.outer(roi1, roi1).ravel()
        saved = {}
        for source_aware in [False, True]:
            for regularize in [False, True]:
                method = ("source_aware" if source_aware else "phase") + (
                    "_regularized" if regularize else ""
                )
                carrier, _ = reference_carriers(
                    spec, old_axis, observed_f, source_aware, regularize
                )
                envelopes = snapshots / carrier
                fine_envelopes = interpolate_envelopes(envelopes, old_axis, axis)
                coarse_carrier, _ = reference_carriers(
                    spec, old_axis, frequencies, source_aware, regularize
                )
                fine_carrier, _ = reference_carriers(
                    spec, axis, frequencies, source_aware, regularize
                )
                for qi, frequency in enumerate(frequencies):
                    coarse_basis = envelopes * coarse_carrier[:, qi, None]
                    coarse_basis = np.column_stack((coarse_basis, coarse_basis.conj()))
                    basis = fine_envelopes * fine_carrier[:, qi, None]
                    basis = np.column_stack((basis, basis.conj()))
                    coarse_pred, _, _ = projected_query(
                        old_lap, old_mass, old_rhs, coarse_basis, frequency, "least_residual"
                    )
                    coefficients = np.linalg.lstsq(coarse_basis, coarse_pred, rcond=1e-10)[0]
                    transferred = basis @ coefficients
                    projected, rank, _ = projected_query(
                        lap, mass, rhs, basis, frequency, "least_residual"
                    )
                    target = targets[qi].ravel()
                    oracle = basis @ np.linalg.lstsq(basis[roi], target[roi], rcond=1e-9)[0]
                    for readout, pred in [
                        ("coarse_coefficients", transferred),
                        ("fine_pde", projected),
                        ("query_truth_oracle", oracle),
                    ]:
                        residual = lap @ pred + (2 * np.pi * frequency) ** 2 * mass * pred - rhs
                        rows.append(
                            {
                                "system": system,
                                "frequency": float(frequency),
                                "method": method,
                                "readout": readout,
                                "rank": rank,
                                "pde_relative_residual": float(
                                    np.linalg.norm(residual) / np.linalg.norm(rhs)
                                ),
                                "source_fraction_of_squared_residual": float(
                                    np.sum(abs(residual[near_source]) ** 2)
                                    / max(np.sum(abs(residual) ** 2), 1e-30)
                                ),
                                "information": "target truth ONLY in named oracle; other readouts use known PDE/context",
                                **complex_metrics(pred[roi], target[roi]),
                            }
                        )
                        if system == config["systems"][0] and qi == len(frequencies) - 1:
                            saved[f"{method}_{readout}"] = pred.reshape(targets.shape[-2:])
                group = [r for r in rows if r["system"] == system and r["method"] == method]
                print(
                    json.dumps(
                        {
                            "system": system,
                            "method": method,
                            "errors": {
                                readout: float(
                                    np.mean(
                                        [
                                            r["complex_rel_l2"]
                                            for r in group
                                            if r["readout"] == readout
                                        ]
                                    )
                                )
                                for readout in [
                                    "coarse_coefficients",
                                    "fine_pde",
                                    "query_truth_oracle",
                                ]
                            },
                        }
                    ),
                    flush=True,
                )
                (output / "results.json").write_text(json.dumps(rows, indent=2))
        if saved:
            np.savez_compressed(
                output / f"system{system}_diagnosis.npz",
                target=targets[-1],
                axis=axis[1:-1],
                frequency=frequencies[-1],
                roi=roi.reshape(targets.shape[-2:]),
                **saved,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
