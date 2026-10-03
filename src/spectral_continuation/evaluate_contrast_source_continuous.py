"""Independent-coordinate audit of classical propagated contrast-source bases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .audit_contrast_source_rom import affine_projection, background_spec, source_trial
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
            selected = np.isin(z["frequencies"], config["observed_frequencies"])
            observed_f = z["frequencies"][selected]
            snapshots = z["field"][selected, 1:-1, 1:-1].reshape(len(observed_f), -1).T
        for grid in config["query_grids"]:
            with np.load(Path(config["query_data"]) / f"system_{system}_grid{grid}.npz") as z:
                if json.loads(str(z["spec"])) != spec:
                    raise ValueError("context and query physics differ")
                selected = np.isin(z["frequencies"], config["query_frequencies"])
                frequencies, fields, axis = (
                    z["frequencies"][selected],
                    z["field"][selected],
                    z["axis"],
                )
            if (
                len(frequencies) != len(config["query_frequencies"])
                or np.isin(frequencies, observed_f).any()
            ):
                raise ValueError("missing query or observed target")
            lap, mass, rhs = discrete_operator(spec, grid)
            _, bg_mass, _ = discrete_operator(background_spec(spec), grid)
            contrast = mass - bg_mass
            keep = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
            roi = np.outer(keep, keep).ravel()
            prepared = {}
            for family in ["phase", "source_aware"]:
                old_reference, _ = reference_carriers(
                    spec, old_axis, observed_f, family == "source_aware", True
                )
                envelopes = interpolate_envelopes(snapshots / old_reference, old_axis, axis)
                carrier, _ = reference_carriers(
                    spec, axis, frequencies, family == "source_aware", True
                )
                prepared[family] = envelopes, carrier
            saved = {}
            for qi, frequency in enumerate(frequencies):
                target = fields[qi, 1:-1, 1:-1].ravel()
                start = time.perf_counter()
                lu = splu(lap + sparse.diags((2 * np.pi * frequency) ** 2 * bg_mass, format="csc"))
                factor_seconds = time.perf_counter() - start
                u0 = lu.solve(rhs)
                for family, (envelopes, carrier) in prepared.items():
                    basis = envelopes * carrier[:, qi, None]
                    basis = np.column_stack((basis, basis.conj()))
                    direct, rank, residual = projected_query(
                        lap, mass, rhs, basis, frequency, "least_residual"
                    )
                    start = time.perf_counter()
                    propagated = source_trial(lu, contrast, basis, frequency)
                    inverse_seconds = time.perf_counter() - start
                    pred, rank_s, residual_s = affine_projection(
                        lap, mass, rhs, u0, propagated, frequency
                    )
                    # A one-background-solve defect correction is a cheaper classical
                    # control; it must not be omitted when comparing six solves.
                    defect = rhs - lap @ direct - (2 * np.pi * frequency) ** 2 * mass * direct
                    one_correction = direct + lu.solve(defect)
                    for label, value, n_rank, resid in [
                        ("field_pde", direct, rank, residual),
                        ("field_pde_one_background_correction", one_correction, rank, None),
                        ("contrast_source_pde", pred, rank_s, residual_s),
                    ]:
                        method = family + "_" + label
                        rows.append(
                            {
                                "system": system,
                                "frequency": float(frequency),
                                "grid": grid,
                                "method": method,
                                "rank": n_rank,
                                "pde_relative_residual": resid,
                                "background_factor_seconds": factor_seconds,
                                "background_inverse_basis_seconds": inverse_seconds,
                                "target_frequency_observations": 0,
                                "truth": "independent PDE solve; only context envelopes interpolated",
                                **complex_metrics(value[roi], target[roi]),
                            }
                        )
                        if (
                            system == config["systems"][0]
                            and grid == max(config["query_grids"])
                            and qi == len(frequencies) - 1
                        ):
                            saved[method] = value.reshape(grid - 2, grid - 2)
                print(
                    json.dumps(
                        {
                            "system": system,
                            "grid": grid,
                            "frequency": float(frequency),
                            "errors": {r["method"]: r["complex_rel_l2"] for r in rows[-6:]},
                        }
                    ),
                    flush=True,
                )
                (output / "results.json").write_text(json.dumps(rows, indent=2))
            if saved:
                np.savez_compressed(
                    output / f"system{system}_f{frequencies[-1]}_grid{grid}.npz",
                    axis=axis[1:-1],
                    target=fields[-1, 1:-1, 1:-1],
                    roi=roi.reshape(grid - 2, grid - 2),
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
