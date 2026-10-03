"""R0 held-coordinate audit before training any modal-envelope decoder.

All methods observe identical complete depth profiles at selected ranges of
the SAME frequency. This is representation/compression, not spectral prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .audit_waveguide_spectral_transport import (
    background_modal_carrier,
    diagonal_medium_phase,
    encode_field,
)
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def curve_codec(coefficients, ranges, take, phase, vectors):
    encoded = (coefficients[take] @ vectors) * np.exp(-1j * phase[take])
    envelope = CubicSpline(ranges[take], encoded, axis=0)(ranges)
    return (envelope * np.exp(1j * phase)) @ vectors.T


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
        readout = depth_basis(depths, spec["depth_m"], 32)
        for frequency in config["frequencies"]:
            fi = int(np.flatnonzero(frequencies == frequency)[0])
            coefficients = encode_field(fields[fi], depths, spec["depth_m"], 32)
            kr, vectors = background_modal_carrier(spec, frequency, 32, "boundary_derivative")
            frames = {
                "raw_modes": (np.zeros_like(coefficients.real), np.eye(32)),
                "background_phase": (ranges[:, None] * kr[None], vectors),
                "diagonal_phase": (
                    diagonal_medium_phase(spec, frequency, ranges, vectors),
                    vectors,
                ),
            }
            for count in config["range_samples"]:
                take = np.linspace(0, len(ranges) - 1, count, dtype=int)
                query = ranges >= config["roi_min_range_m"]
                query[take] = False
                for label, (phase, frame) in frames.items():
                    predicted = curve_codec(coefficients, ranges, take, phase, frame)
                    field = readout @ predicted.T
                    rows.append(
                        {
                            "system": system,
                            "frequency": frequency,
                            "method": label,
                            "range_samples": count,
                            "modal_complex_values": count * 32,
                            "scope": "R0 same-frequency profile observations; held range coordinates only; not shared learning or unobserved-frequency generation",
                            **complex_metrics(field[:, query], fields[fi][:, query]),
                        }
                    )
        print(
            json.dumps(
                {
                    "system": system,
                    "best_budget_errors": {
                        m: float(
                            np.mean(
                                [
                                    r["complex_rel_l2"]
                                    for r in rows
                                    if r["system"] == system
                                    and r["method"] == m
                                    and r["range_samples"] == config["range_samples"][-1]
                                ]
                            )
                        )
                        for m in frames
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
