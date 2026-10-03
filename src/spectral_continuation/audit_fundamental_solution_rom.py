"""Source-aware complex reference factors, instead of a phase-only carrier.

The outgoing 2D homogeneous Green function is convolved with the actual Gaussian
source, by the radial Bessel addition formula. This is a classical reference,
not learned and not a claim to a new Green-function technique.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.special import hankel1, jv

from .audit_phase_reduced_operator import discrete_operator, projected_query, travel_coordinates
from .coherent_representation import complex_metrics


def gaussian_reference(radius, frequency, width, speed=1.0, loss=0.002, nodes=16385):
    """Solve (Delta+k^2)u=-exp(-r^2/(2 width^2)) in the infinite 2D plane."""
    radial = np.linspace(0, max(float(np.max(radius)), 10 * width), nodes)
    k = 2 * np.pi * frequency / speed * (1 + 1j * loss)
    bessel = jv(0, k * radial)
    hankel = hankel1(0, k * np.maximum(radial, 1e-30))
    weighted_source = radial * np.exp(-0.5 * (radial / width) ** 2)
    first = cumulative_simpson(
        (bessel * weighted_source).real, x=radial, initial=0
    ) + 1j * cumulative_simpson((bessel * weighted_source).imag, x=radial, initial=0)
    second_integrand = hankel * weighted_source
    second_integrand[0] = 0
    integral = cumulative_simpson(
        second_integrand.real, x=radial, initial=0
    ) + 1j * cumulative_simpson(second_integrand.imag, x=radial, initial=0)
    second = integral[-1] - integral
    solution = 0.5j * np.pi * (hankel * first + bessel * second)
    return CubicSpline(radial, solution)(radius)


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
        radius = np.hypot(xx - spec["source"][0], yy - spec["source"][1]).ravel()
        roi1 = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
        roi = np.outer(roi1, roi1).ravel()
        coords = travel_coordinates(spec, axis)
        excess = coords["straight_ray_medium_phase"] - coords["reference_phase"]
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        query = np.flatnonzero(np.isin(frequencies, config["query_frequencies"]))
        if set(observed).intersection(query):
            raise ValueError("query frequency observations forbidden")
        reference = np.stack(
            [
                gaussian_reference(radius, f, spec["source_width"], loss=spec["background_loss"])
                for f in frequencies
            ],
            axis=1,
        )
        snapshots = fields[observed, 1:-1, 1:-1].reshape(len(observed), -1).T
        for name in ["gaussian_green", "gaussian_green_medium_phase"]:
            factor = reference.copy()
            if name.endswith("medium_phase"):
                factor *= np.exp(2j * np.pi * excess[:, None] * frequencies[None])
            envelope = snapshots / factor[:, observed]
            for qi in query:
                frequency = float(frequencies[qi])
                target = fields[qi, 1:-1, 1:-1].ravel()
                basis = envelope * factor[:, qi, None]
                basis = np.column_stack((basis, basis.conj()))
                prediction, rank, residual = projected_query(
                    lap, mass, rhs, basis, frequency, "least_residual"
                )
                coefficient = np.linalg.lstsq(basis[roi], target[roi], rcond=1e-9)[0]
                cubic = (
                    CubicSpline(frequencies[observed], envelope, axis=1)(frequency) * factor[:, qi]
                )
                variants = {
                    "pde": prediction,
                    "query_truth_oracle": basis @ coefficient,
                    "cubic": cubic,
                }
                for readout, pred in variants.items():
                    rows.append(
                        {
                            "system": system,
                            "frequency": frequency,
                            "reference": name,
                            "readout": readout,
                            "rank": rank,
                            "pde_residual": residual if readout == "pde" else None,
                            "minimum_carrier_magnitude": float(abs(factor[:, observed]).min()),
                            "information": "known medium/source/PDE; no target samples except labelled oracle",
                            **complex_metrics(pred[roi], target[roi]),
                        }
                    )
        for name in sorted({r["reference"] for r in rows}):
            group = [r for r in rows if r["system"] == system and r["reference"] == name]
            print(
                json.dumps(
                    {
                        "system": system,
                        "reference": name,
                        "macro_error": {
                            readout: float(
                                np.mean(
                                    [r["complex_rel_l2"] for r in group if r["readout"] == readout]
                                )
                            )
                            for readout in ["pde", "query_truth_oracle", "cubic"]
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
