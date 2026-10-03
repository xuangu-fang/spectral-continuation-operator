"""Scalar versus operator-valued phase carriers on a one-way waveguide.

Context modal coefficients are recovered from complete observed fields, not
read from privileged teacher coefficients. All carriers use known background
medium only. Query solutions are used exclusively for reporting metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, interp1d

from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, local_generator, quadrature, sound_speed


def encode_field(field, depths, depth, modes):
    weights = np.full(len(depths), depths[1] - depths[0])
    weights[[0, -1]] *= 0.5
    basis = depth_basis(depths, depth, modes)
    return field.T @ (weights[:, None] * basis)


def background_modal_carrier(spec, frequency, modes, gauge="largest_component"):
    background = json.loads(json.dumps(spec))
    background["contrasts_m_s"] = [0.0] * len(background["contrasts_m_s"])
    z, weights, basis = quadrature(background, modes, 160)
    kr, vectors = local_generator(background, frequency, 0.0, z, weights, basis)
    # Align eigenvector signs using the largest fixed-basis component, without
    # context or query truth. This is not a physical global-phase alignment.
    if gauge == "largest_component":
        signs = np.sign(vectors[np.argmax(abs(vectors), axis=0), np.arange(modes)])
    elif gauge == "boundary_derivative":
        kz = (np.arange(modes) + 0.5) * np.pi / spec["depth_m"]
        signs = np.sign(kz @ vectors)
    else:
        raise ValueError(gauge)
    vectors *= np.where(signs == 0, 1, signs)[None]
    return kr.real, vectors


def diagonal_medium_phase(spec, frequency, ranges, vectors):
    """Known-medium diagonal phase; neglects off-diagonal modal interactions."""
    modes = vectors.shape[0]
    z, weights, fixed_basis = quadrature(spec, modes, 160)
    vertical_modes = fixed_basis @ vectors
    speed = sound_speed(spec, ranges[:, None], z[None])
    kz = (np.arange(modes) + 0.5) * np.pi / spec["depth_m"]
    diagonal = (2 * np.pi * frequency) ** 2 * (
        (1 / speed**2) @ (weights[:, None] * vertical_modes**2)
    )
    diagonal -= (kz[:, None] ** 2 * vectors**2).sum(axis=0)[None]
    kr = np.sqrt(diagonal.astype(complex)).real
    return cumulative_trapezoid(kr, x=ranges, axis=0, initial=0)


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system{system}_m32_r1025.npz") as z:
            fields, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
            spec = json.loads(str(z["spec"]))
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        queries = np.flatnonzero(~np.isin(frequencies, config["observed_frequencies"]))
        modes = 32
        context = np.stack(
            [encode_field(fields[i], depths, spec["depth_m"], modes) for i in observed]
        )
        roi = ranges >= config["roi_min_range_m"]
        depth_readout = depth_basis(depths, spec["depth_m"], modes)
        for family in config.get("carriers", ["raw", "scalar_phase", "background_modal_phase"]):
            envelopes = []
            for ci, fi in enumerate(frequencies[observed]):
                if family == "raw":
                    envelope = context[ci]
                elif family == "scalar_phase":
                    envelope = context[ci] * np.exp(
                        -2j * np.pi * fi / spec["reference_speed_m_s"] * ranges[:, None]
                    )
                else:
                    kr, vectors = background_modal_carrier(
                        spec, fi, modes, config.get("gauge", "largest_component")
                    )
                    phase = (
                        diagonal_medium_phase(spec, fi, ranges, vectors)
                        if family == "diagonal_medium_phase"
                        else ranges[:, None] * kr[None]
                    )
                    envelope = (context[ci] @ vectors) * np.exp(-1j * phase)
                envelopes.append(envelope)
            envelopes = np.stack(envelopes)
            for interpolation in ["linear", "cubic"]:
                interpolator = (
                    interp1d(frequencies[observed], envelopes, axis=0)
                    if interpolation == "linear"
                    else CubicSpline(frequencies[observed], envelopes, axis=0)
                )
                for qi in queries:
                    frequency = float(frequencies[qi])
                    envelope = interpolator(frequency)
                    if family == "raw":
                        prediction_modes = envelope
                    elif family == "scalar_phase":
                        prediction_modes = envelope * np.exp(
                            2j * np.pi * frequency / spec["reference_speed_m_s"] * ranges[:, None]
                        )
                    else:
                        kr, vectors = background_modal_carrier(
                            spec, frequency, modes, config.get("gauge", "largest_component")
                        )
                        phase = (
                            diagonal_medium_phase(spec, frequency, ranges, vectors)
                            if family == "diagonal_medium_phase"
                            else ranges[:, None] * kr[None]
                        )
                        prediction_modes = (envelope * np.exp(1j * phase)) @ vectors.T
                    prediction = depth_readout @ prediction_modes.T
                    rows.append(
                        {
                            "system": system,
                            "frequency": frequency,
                            "method": family + "_" + interpolation,
                            "target_frequency_observations": 0,
                            "information": "complete context fields; known full SSP for diagonal_medium_phase, stationary background SSP for other carriers; no target samples",
                            **complex_metrics(prediction[:, roi], fields[qi][:, roi]),
                        }
                    )
            print(
                json.dumps(
                    {
                        "system": system,
                        "carrier": family,
                        "errors": {
                            kind: float(
                                np.mean(
                                    [
                                        r["complex_rel_l2"]
                                        for r in rows
                                        if r["system"] == system
                                        and r["method"] == family + "_" + kind
                                    ]
                                )
                            )
                            for kind in ["linear", "cubic"]
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
