"""Known-source adiabatic baseline and context-residual interpolation.

The classical adiabatic approximation removes only diagonal propagation, not
range-dependent mode coupling. Its full-medium eigenproblem cost is explicit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from .audit_waveguide_spectral_transport import encode_field, local_modal_frame, modal_projection
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, quadrature


def adiabatic_reference(spec, frequency, ranges, modes):
    phase, vectors, attenuation = local_modal_frame(spec, frequency, ranges, modes, True)
    z, weights, basis = quadrature(spec, modes, 160)
    source = np.exp(-0.5 * ((z - spec["source_depth_m"]) / spec["source_width_m"]) ** 2)
    initial = basis.T @ (weights * source)
    envelope = (initial @ vectors[0])[None] * attenuation
    return envelope, phase, vectors


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(
            Path(config["data"]) / f"system{system}_m32_r{config['range_count']}.npz"
        ) as z:
            fields, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
            spec = json.loads(str(z["spec"]))
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        queries = np.flatnonzero(~np.isin(frequencies, config["observed_frequencies"]))
        roi = ranges >= config["roi_min_range_m"]
        basis = depth_basis(depths, spec["depth_m"], 32)
        corrections = []
        start = time.perf_counter()
        for fi in observed:
            reference, phase, vectors = adiabatic_reference(spec, frequencies[fi], ranges, 32)
            encoded = modal_projection(
                encode_field(fields[fi], depths, spec["depth_m"], 32), vectors
            )
            corrections.append(encoded * np.exp(-1j * phase) - reference)
        corrections = np.stack(corrections)
        context_seconds = time.perf_counter() - start
        interpolators = {
            "linear": interp1d(frequencies[observed], corrections, axis=0),
            "cubic": CubicSpline(frequencies[observed], corrections, axis=0),
        }
        for fi in queries:
            frequency = float(frequencies[fi])
            start = time.perf_counter()
            reference, phase, vectors = adiabatic_reference(spec, frequency, ranges, 32)
            setup_seconds = time.perf_counter() - start
            for method in ["adiabatic_only", "residual_linear", "residual_cubic"]:
                envelope = (
                    reference
                    if method == "adiabatic_only"
                    else reference + interpolators[method.split("_")[1]](frequency)
                )
                coefficients = modal_projection(
                    envelope * np.exp(1j * phase), vectors, inverse=True
                )
                prediction = basis @ coefficients.T
                rows.append(
                    {
                        "system": system,
                        "frequency": frequency,
                        "method": method,
                        "frame_setup_seconds": setup_seconds,
                        "context_encoding_seconds": context_seconds,
                        "target_frequency_observations": 0,
                        "scope": "classical known-full-medium/source; local eigenproblems at every range node, not free decoder",
                        **complex_metrics(prediction[:, roi], fields[fi][:, roi]),
                    }
                )
        print(
            json.dumps(
                {
                    "system": system,
                    "errors": {
                        method: float(
                            np.mean(
                                [
                                    r["complex_rel_l2"]
                                    for r in rows
                                    if r["system"] == system and r["method"] == method
                                ]
                            )
                        )
                        for method in ["adiabatic_only", "residual_linear", "residual_cubic"]
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
