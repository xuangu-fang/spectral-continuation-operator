"""Classical frequency-affine operator latent with continuous phase readout.

This tests representation scaling, NOT a new numerical or neural method.
Store depth mass matrices at sparse range cells. At each frequency, matrix
square roots generate rapid oscillations analytically within each cell.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, quadrature, sound_speed


def encode_operator(spec, cells, modes=32, quadrature_nodes=160):
    edges = np.linspace(0, spec["range_m"], cells + 1)
    middle = (edges[:-1] + edges[1:]) / 2
    z, weights, basis = quadrature(spec, modes, quadrature_nodes)
    speed = sound_speed(spec, middle[:, None], z[None])
    mass = np.einsum("zm,rz,zn->rmn", basis, weights[None] / speed**2, basis)
    source = np.exp(-0.5 * ((z - spec["source_depth_m"]) / spec["source_width_m"]) ** 2)
    initial = basis.T @ (weights * source)
    kz2 = ((np.arange(modes) + 0.5) * np.pi / spec["depth_m"]) ** 2
    return {
        "edges": edges,
        "mass": mass,
        "kz2": kz2,
        "initial": initial,
        "attenuation": spec["attenuation_per_m"],
        "depth": spec["depth_m"],
    }


def query_coefficients(latent, frequency, ranges):
    ranges = np.asarray(ranges)
    edges = latent["edges"]
    if np.any(ranges < edges[0]) or np.any(ranges > edges[-1]):
        raise ValueError("query outside encoded range")
    vertical = (2 * np.pi * frequency) ** 2 * latent["mass"] - np.diag(latent["kz2"])[None]
    values, vectors = np.linalg.eigh(vertical)
    rate = 1j * np.sqrt(values.astype(complex)) - latent["attenuation"]
    current = latent["initial"].astype(complex)
    result = np.zeros((len(ranges), len(current)), dtype=complex)
    assignment = np.clip(np.searchsorted(edges, ranges, side="right") - 1, 0, len(edges) - 2)
    for cell in range(len(edges) - 1):
        take = assignment == cell
        modal = vectors[cell].T @ current
        if take.any():
            partial = np.exp((ranges[take] - edges[cell])[:, None] * rate[cell][None]) * modal[None]
            result[take] = partial @ vectors[cell].T
        current = vectors[cell] @ (np.exp((edges[cell + 1] - edges[cell]) * rate[cell]) * modal)
    return result


def query_field(latent, frequency, ranges, depths):
    coefficients = query_coefficients(latent, frequency, ranges)
    return depth_basis(depths, latent["depth"], len(latent["initial"])) @ coefficients.T


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        path = (
            Path(config["data"]) / f"system{system}_m{config['modes']}_r{config['range_count']}.npz"
        )
        with np.load(path) as z:
            fields, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
            spec = json.loads(str(z["spec"]))
        roi = ranges >= config["roi_min_range_m"]
        for cells in config["cells"]:
            start = time.perf_counter()
            latent = encode_operator(spec, cells, config["modes"])
            encoding_seconds = time.perf_counter() - start
            for frequency in config["query_frequencies"]:
                qi = int(np.flatnonzero(np.isclose(frequencies, frequency))[0])
                start = time.perf_counter()
                prediction = query_field(latent, frequency, ranges, depths)
                seconds = time.perf_counter() - start
                rows.append(
                    {
                        "system": system,
                        "frequency": frequency,
                        "cells": cells,
                        "modes": config["modes"],
                        "encoding_seconds": encoding_seconds,
                        "query_seconds": seconds,
                        "mass_real_entries": int(latent["mass"].size),
                        "output_complex_values": int(prediction.size),
                        "scope": "classical known-full-medium/source operator compression; no context fields required; not a learned method",
                        **complex_metrics(prediction[:, roi], fields[qi][:, roi]),
                    }
                )
            print(
                json.dumps(
                    {
                        "system": system,
                        "cells": cells,
                        "macro_error": float(
                            np.mean(
                                [
                                    r["complex_rel_l2"]
                                    for r in rows
                                    if r["system"] == system and r["cells"] == cells
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
