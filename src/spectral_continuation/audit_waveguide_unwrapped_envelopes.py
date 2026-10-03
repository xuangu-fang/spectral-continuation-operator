"""Classical magnitude / spatially unwrapped modal-phase spectral control.

Modal phase is referenced to the observed initial source profile. No target
phase or environment contrasts are used. All errors remain raw complex errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from .audit_waveguide_spectral_transport import background_modal_carrier, encode_field
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def relative_unwrapped_phase(envelope, initial, axis):
    unit = np.exp(1j * np.angle(initial))
    phase = np.unwrap(np.angle(envelope * unit[:, None].conj()), axis=axis)
    if axis == 1:
        phase = phase - phase[:, :1]
    return phase


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system{system}_m32_r2049.npz") as z:
            fields, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
            spec = json.loads(str(z["spec"]))
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        queries = np.flatnonzero(~np.isin(frequencies, config["observed_frequencies"]))
        coefficients = np.stack(
            [encode_field(fields[i], depths, spec["depth_m"], 32) for i in observed]
        )
        initial = coefficients[0, 0]
        mismatch = np.linalg.norm(coefficients[:, 0] - initial) / np.linalg.norm(coefficients[:, 0])
        if mismatch > 1e-6:
            raise ValueError(
                "initial source is not frequency independent; protocol requires explicit source calibration"
            )
        envelopes, source = [], []
        for ci, oi in enumerate(observed):
            kr, vectors = background_modal_carrier(spec, frequencies[oi], 32, "boundary_derivative")
            envelopes.append(
                (coefficients[ci] @ vectors) * np.exp(-1j * ranges[:, None] * kr[None])
            )
            source.append(initial @ vectors)
        envelopes, source = np.stack(envelopes), np.stack(source)
        rms = np.sqrt(np.mean(abs(envelopes) ** 2))
        source_strength = np.min(abs(source), axis=0) / np.linalg.norm(initial)
        readout = depth_basis(depths, spec["depth_m"], 32)
        roi = ranges >= config["roi_min_range_m"]
        for unwrap in ["frequency", "space"]:
            phase = relative_unwrapped_phase(
                envelopes, source, axis=0 if unwrap == "frequency" else 1
            )
            for interpolation in ["linear", "cubic"]:

                def make(values, axis_frequencies=frequencies[observed], kind=interpolation):
                    return (
                        interp1d(axis_frequencies, values, axis=0)
                        if kind == "linear"
                        else CubicSpline(axis_frequencies, values, axis=0)
                    )

                phase_interp = make(phase)
                complex_interp = make(envelopes)
                for amplitude in ["absolute", "log"]:
                    amp_interp = make(
                        abs(envelopes)
                        if amplitude == "absolute"
                        else np.log(np.maximum(abs(envelopes), 1e-8 * rms))
                    )
                    for qi in queries:
                        frequency = float(frequencies[qi])
                        kr, vectors = background_modal_carrier(
                            spec, frequency, 32, "boundary_derivative"
                        )
                        predicted_amp = amp_interp(frequency)
                        predicted_amp = (
                            np.maximum(predicted_amp, 0)
                            if amplitude == "absolute"
                            else np.exp(np.clip(predicted_amp, np.log(rms) - 30, np.log(rms) + 10))
                        )
                        predicted = (
                            predicted_amp
                            * np.exp(1j * phase_interp(frequency))
                            * np.exp(1j * np.angle(initial @ vectors))[None]
                        )
                        for threshold in config["source_phase_fallback_thresholds"]:
                            envelope = np.where(
                                (source_strength < threshold)[None],
                                complex_interp(frequency),
                                predicted,
                            )
                            modes = envelope * np.exp(1j * ranges[:, None] * kr[None])
                            field = (readout @ vectors) @ modes.T
                            rows.append(
                                {
                                    "system": system,
                                    "frequency": frequency,
                                    "unwrap": unwrap,
                                    "interpolation": interpolation,
                                    "amplitude": amplitude,
                                    "source_threshold": threshold,
                                    "fallback_mode_count": int(np.sum(source_strength < threshold)),
                                    "scope": "known stationary background and 3 complete context fields, common initial profile inferred from context, no full SSP or target phase",
                                    **complex_metrics(field[:, roi], fields[qi][:, roi]),
                                }
                            )
        summary = {}
        for unwrap in ["frequency", "space"]:
            summary[unwrap] = float(
                np.mean(
                    [
                        r["complex_rel_l2"]
                        for r in rows
                        if r["system"] == system
                        and r["unwrap"] == unwrap
                        and r["interpolation"] == "linear"
                        and r["amplitude"] == "absolute"
                        and r["source_threshold"] == 0
                    ]
                )
            )
        print(json.dumps({"system": system, "linear_abs_errors": summary}), flush=True)
        (output / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
