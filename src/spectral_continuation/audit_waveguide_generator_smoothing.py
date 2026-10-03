"""Stronger same-input classical controls before learned generator inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_generator_identification import (
    generator_geometry,
    identify,
    latent_from_coefficients,
    normal_statistics,
)
from .audit_waveguide_spectral_transport import encode_field
from .coarse_waveguide_operator import query_field
from .coherent_representation import complex_metrics


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system{system}_m32_r2049.npz") as z:
            spec = json.loads(str(z["spec"]))
            fields, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
        obs = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        take = np.linspace(0, len(ranges) - 1, config["context_range_samples"], dtype=int)
        coefficients = np.stack(
            [encode_field(fields[i][:, take], depths, spec["depth_m"], 32) for i in obs]
        )
        geometry = generator_geometry(spec, config["cells"], config["depth_terms"])
        roi = ranges >= config["roi_min_range_m"]
        for order in config["orders"]:
            gram, rhs = normal_statistics(
                spec, coefficients, frequencies[obs], ranges[take], geometry, order
            )
            for penalty in config["range_penalties"]:
                beta = identify(gram, rhs, config["ridge"], penalty)
                latent, clipping = latent_from_coefficients(beta, coefficients[0, 0], geometry)
                for frequency in config["query_frequencies"]:
                    fi = int(np.flatnonzero(frequencies == frequency)[0])
                    prediction = query_field(latent, frequency, ranges, depths)
                    rows.append(
                        {
                            "system": system,
                            "order": order,
                            "range_penalty": penalty,
                            "frequency": frequency,
                            "clipping_fraction": clipping,
                            **complex_metrics(prediction[:, roi], fields[fi][:, roi]),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "system": system,
                            "order": order,
                            "range_penalty": penalty,
                            "macro_error": float(
                                np.mean(
                                    [
                                        r["complex_rel_l2"]
                                        for r in rows[-len(config["query_frequencies"]) :]
                                    ]
                                )
                            ),
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
