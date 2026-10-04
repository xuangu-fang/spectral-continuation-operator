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
    if "spectral_coefficients_m_s" in background:
        count = len(background["spectral_coefficients_m_s"])
        background["spectral_coefficients_m_s"] = [[0.0] * count for _ in range(count)]
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


def component_replacement(prediction, target, component):
    """QUERY-TRUTH ORACLE: isolate coefficient magnitude versus phase error.

    No phase alignment is optimized. Zero coefficients use unit phase only to
    define the intervention; this is not a deployable correction.
    """
    if component == "amplitude":
        return abs(target) * np.exp(1j * np.angle(prediction))
    if component == "phase":
        return abs(prediction) * np.exp(1j * np.angle(target))
    raise ValueError(component)


def local_modal_frame(spec, frequency, ranges, modes, return_decay=False):
    """Known-full-medium local eigenbasis. Classical, not a cheap learned encoder.

    Surface-derivative sign fixes the real eigenvector gauge. Integrating the
    real wavenumber does not invert evanescent decay. Range derivatives of the
    basis (mode coupling) are deliberately left in the encoded envelope.
    """
    z, weights, basis = quadrature(spec, modes, 160)
    speed = sound_speed(spec, ranges[:, None], z[None])
    vertical = np.einsum(
        "zm,rz,zn->rmn", basis, weights[None] * (2 * np.pi * frequency / speed) ** 2, basis
    )
    kz = (np.arange(modes) + 0.5) * np.pi / spec["depth_m"]
    vertical -= np.diag(kz**2)[None]
    values, vectors = np.linalg.eigh(vertical)
    signs = np.sign(np.einsum("m,rmn->rn", kz, vectors))
    vectors *= np.where(signs == 0, 1, signs)[:, None]
    kr = np.sqrt(values.astype(complex))
    phase = cumulative_trapezoid(kr.real, x=ranges, axis=0, initial=0)
    if return_decay:
        decay = cumulative_trapezoid(kr.imag, x=ranges, axis=0, initial=0)
        decay += spec["attenuation_per_m"] * ranges[:, None]
        return phase, vectors, np.exp(-decay)
    return phase, vectors


def modal_projection(coefficients, vectors, inverse=False):
    if vectors.ndim == 2:
        return coefficients @ (vectors.T if inverse else vectors)
    if inverse:
        return np.einsum("rn,rmn->rm", coefficients, vectors)
    return np.einsum("rm,rmn->rn", coefficients, vectors)


def carrier_frame(spec, frequency, ranges, modes, family, gauge):
    if family == "local_modal_phase":
        return local_modal_frame(spec, frequency, ranges, modes)
    kr, vectors = background_modal_carrier(spec, frequency, modes, gauge)
    phase = (
        diagonal_medium_phase(spec, frequency, ranges, vectors)
        if family == "diagonal_medium_phase"
        else ranges[:, None] * kr[None]
    )
    return phase, vectors


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        modes = config.get("modes", 32)
        range_count = config.get("range_count", 1025)
        with np.load(Path(config["data"]) / f"system{system}_m{modes}_r{range_count}.npz") as z:
            fields, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
            spec = json.loads(str(z["spec"]))
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        queries = np.flatnonzero(~np.isin(frequencies, config["observed_frequencies"]))
        if "query_frequencies" in config:
            if np.isin(config["query_frequencies"], frequencies[observed]).any():
                raise ValueError("query frequencies overlap observed context")
            queries = np.flatnonzero(np.isin(frequencies, config["query_frequencies"]))
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
                    phase, vectors = carrier_frame(
                        spec, fi, ranges, modes, family, config.get("gauge", "largest_component")
                    )
                    envelope = modal_projection(context[ci], vectors) * np.exp(-1j * phase)
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
                        phase, vectors = carrier_frame(
                            spec,
                            frequency,
                            ranges,
                            modes,
                            family,
                            config.get("gauge", "largest_component"),
                        )
                        prediction_modes = modal_projection(
                            envelope * np.exp(1j * phase), vectors, inverse=True
                        )
                    prediction = depth_readout @ prediction_modes.T
                    rows.append(
                        {
                            "system": system,
                            "frequency": frequency,
                            "method": family + "_" + interpolation,
                            "target_frequency_observations": 0,
                            "information": "complete context fields; known full SSP for diagonal/local modal phase, stationary background SSP for other carriers; no target samples",
                            **complex_metrics(prediction[:, roi], fields[qi][:, roi]),
                        }
                    )
                    if config.get("component_oracles", False) and family in [
                        "background_modal_phase",
                        "diagonal_medium_phase",
                        "local_modal_phase",
                    ]:
                        # Query truth is accessed only below, AFTER prediction.
                        target_modes = encode_field(fields[qi], depths, spec["depth_m"], modes)
                        target_envelope = modal_projection(target_modes, vectors) * np.exp(
                            -1j * phase
                        )
                        for component in ["amplitude", "phase"]:
                            replaced = component_replacement(envelope, target_envelope, component)
                            oracle_modes = modal_projection(
                                replaced * np.exp(1j * phase), vectors, inverse=True
                            )
                            oracle_field = depth_readout @ oracle_modes.T
                            rows.append(
                                {
                                    "system": system,
                                    "frequency": frequency,
                                    "method": family + "_" + interpolation + "_truth_" + component,
                                    "scope": "query-truth component replacement oracle; NOT deployable",
                                    **complex_metrics(oracle_field[:, roi], fields[qi][:, roi]),
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
