"""Direct fractional-frequency and fine-coordinate truth for continuous ROMs.

Only CONTEXT envelopes are spatially interpolated. Query truth is independently
solved at the requested grid/frequency. All methods know the medium/source/PDE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from .audit_fundamental_solution_rom import gaussian_reference
from .audit_phase_reduced_operator import discrete_operator, projected_query, travel_coordinates
from .coherent_representation import complex_metrics
from .learned_phase_rom import EnvelopeCoordinate, phase_features


def reference_carriers(spec, axis, frequencies, source_aware, regularize=False):
    coordinates = travel_coordinates(spec, axis)
    tau = coordinates["straight_ray_medium_phase"]
    radius = coordinates["reference_phase"]
    smooth_factor = radius / np.sqrt(radius**2 + spec["source_width"] ** 2) if regularize else 1.0
    if not source_aware:
        return np.exp(
            2j * np.pi * (tau * smooth_factor)[:, None] * np.asarray(frequencies)[None]
        ), coordinates
    excess = (tau - radius) * smooth_factor
    carriers = np.stack(
        [
            gaussian_reference(radius, f, spec["source_width"], loss=spec["background_loss"])
            * np.exp(2j * np.pi * f * excess)
            for f in frequencies
        ],
        axis=1,
    )
    return carriers, coordinates


def interpolate_envelopes(envelopes, old_axis, new_axis):
    if np.array_equal(old_axis, new_axis):
        return envelopes.copy()
    output = []
    for j in range(envelopes.shape[1]):
        full = np.zeros((len(old_axis), len(old_axis)), dtype=complex)
        full[1:-1, 1:-1] = envelopes[:, j].reshape(len(old_axis) - 2, -1)
        value = RectBivariateSpline(old_axis, old_axis, full.real)(new_axis[1:-1], new_axis[1:-1])
        value = value + 1j * RectBivariateSpline(old_axis, old_axis, full.imag)(
            new_axis[1:-1], new_axis[1:-1]
        )
        output.append(value.ravel())
    return np.stack(output, axis=1)


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    model = EnvelopeCoordinate(
        checkpoint["config"]["kind"], checkpoint["config"].get("width", 32)
    ).double()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["context_data"]) / f"system_{system}_grid257.npz") as z:
            coarse_axis = z["axis"]
            spec = json.loads(str(z["spec"]))
            idx = np.flatnonzero(np.isin(z["frequencies"], config["observed_frequencies"]))
            observed_f = z["frequencies"][idx]
            snapshots = z["field"][idx, 1:-1, 1:-1].reshape(len(idx), -1).T
        for grid in config["query_grids"]:
            with np.load(Path(config["query_data"]) / f"system_{system}_grid{grid}.npz") as z:
                axis, frequencies, fields = z["axis"], z["frequencies"], z["field"]
                if json.loads(str(z["spec"])) != spec:
                    raise ValueError("context/query environment mismatch")
            if config.get("query_frequencies"):
                selected = np.isin(frequencies, config["query_frequencies"])
                frequencies, fields = frequencies[selected], fields[selected]
                if not len(frequencies):
                    raise ValueError("no requested query frequency available")
            if np.isin(frequencies, observed_f).any():
                raise ValueError("query frequency observed")
            lap, mass, rhs = discrete_operator(spec, grid)
            prepared = {}
            methods = config.get(
                "methods", ["phase", "source_aware", "phase_radial", "phase_learned_envelope"]
            )
            settings = sorted(
                {(m.startswith("source_aware"), m.endswith("_regularized")) for m in methods}
            )
            for source_aware, regularize in settings:
                observed_carrier, _ = reference_carriers(
                    spec, coarse_axis, observed_f, source_aware, regularize
                )
                envelopes = interpolate_envelopes(snapshots / observed_carrier, coarse_axis, axis)
                query_carrier, coordinates = reference_carriers(
                    spec, axis, frequencies, source_aware, regularize
                )
                prepared[(source_aware, regularize)] = (envelopes, query_carrier)
            item = {
                "features": torch.tensor(
                    phase_features(
                        spec,
                        axis,
                        coordinates,
                        checkpoint["config"].get("source_regularized", False),
                    ),
                    dtype=torch.float64,
                ),
                "coordinates": {k: torch.tensor(v) for k, v in coordinates.items()},
                "source_width": spec["source_width"],
                "source_regularized": checkpoint["config"].get("source_regularized", False),
            }
            with torch.no_grad():
                factors = {
                    "phase_radial": model.fixed_factors(item).numpy(),
                    "phase_learned_envelope": model.factors(item).numpy(),
                }
            roi1 = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
            roi = np.outer(roi1, roi1).ravel()
            saved = {}
            for qi, frequency in enumerate(frequencies):
                target = fields[qi, 1:-1, 1:-1].ravel()
                for method in methods:
                    regularize = method.endswith("_regularized")
                    base_method = method.removesuffix("_regularized")
                    envelopes, carrier = prepared[(method.startswith("source_aware"), regularize)]
                    basis = envelopes * carrier[:, qi, None]
                    basis = np.column_stack((basis, basis.conj()))
                    if base_method in factors:
                        basis = (basis[:, :, None] * factors[base_method][:, None]).reshape(
                            len(target), -1
                        )
                    basis /= np.linalg.norm(basis, axis=0, keepdims=True).clip(1e-30)
                    start = time.perf_counter()
                    prediction, rank, residual = projected_query(
                        lap, mass, rhs, basis, float(frequency), "least_residual"
                    )
                    rows.append(
                        {
                            "system": system,
                            "frequency": float(frequency),
                            "grid": grid,
                            "method": method,
                            "rank": rank,
                            "pde_relative_residual": residual,
                            "projection_seconds": time.perf_counter() - start,
                            "target_frequency_observations": 0,
                            "truth": "direct independently solved frequency/grid; only context envelopes interpolated",
                            **complex_metrics(prediction[roi], target[roi]),
                        }
                    )
                    if (
                        system == config["systems"][0]
                        and grid == max(config["query_grids"])
                        and qi == len(frequencies) - 1
                    ):
                        saved[method] = prediction.reshape(grid - 2, grid - 2)
                print(
                    json.dumps(
                        {
                            "system": system,
                            "grid": grid,
                            "frequency": float(frequency),
                            "errors": {
                                r["method"]: r["complex_rel_l2"] for r in rows[-len(methods) :]
                            },
                        }
                    ),
                    flush=True,
                )
                (output / "results.json").write_text(json.dumps(rows, indent=2))
            if saved:
                np.savez_compressed(
                    output / f"system{system}_frequency{frequencies[-1]}_grid{grid}.npz",
                    target=fields[-1, 1:-1, 1:-1],
                    axis=axis[1:-1],
                    frequency=frequencies[-1],
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
