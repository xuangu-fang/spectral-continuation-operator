"""Reusable reduced-trajectory supervision with explicit numerical audits.

Not full Helmholtz data. Same one-way Galerkin model as existing development
fields; range propagation is independently controlled by the cell count.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np

from .coarse_waveguide_operator import encode_operator, query_coefficients
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, environment


def generate_one(task):
    system, config, output = task
    path = output / f"system{system}_modal.npz"
    if path.exists():
        raise FileExistsError(path)
    start = time.perf_counter()
    spec = environment(system)
    latent = encode_operator(spec, config["cells"], config["modes"])
    ranges = np.linspace(0, spec["range_m"], config["range_samples"])
    coefficients = np.stack([query_coefficients(latent, f, ranges) for f in config["frequencies"]])
    np.savez_compressed(
        path,
        modal_coefficients=coefficients.astype(np.complex64),
        ranges_m=ranges,
        frequencies_hz=config["frequencies"],
        spec=json.dumps(spec),
        modes=config["modes"],
        propagation_cells=config["cells"],
        evidence="training-only compressed modal trajectories; one-way Galerkin teacher; no full field or formal test",
        phase_convention="exp(-i omega t), positive/decaying square root; real initial Gaussian source",
    )
    return {
        "system": system,
        "path": str(path),
        "seconds": time.perf_counter() - start,
        "shape": list(coefficients.shape),
        "bytes": path.stat().st_size,
    }


def audit(config):
    rows = []
    for system in config["audit_systems"]:
        spec = environment(system)
        coarse = encode_operator(spec, config["cells"], config["modes"])
        fine = encode_operator(spec, config["reference_cells"], config["modes"])
        ranges = np.linspace(0, spec["range_m"], 2049)
        depths = np.linspace(0, spec["depth_m"], 129)
        readout = depth_basis(depths, spec["depth_m"], config["modes"])
        roi = ranges >= 50
        for frequency in config["audit_frequencies"]:
            prediction = readout @ query_coefficients(coarse, frequency, ranges).T
            reference = readout @ query_coefficients(fine, frequency, ranges).T
            rows.append(
                {
                    "system": system,
                    "frequency": frequency,
                    "cells": config["cells"],
                    "reference_cells": config["reference_cells"],
                    "scope": "within-model discretization convergence; no statement about full Helmholtz accuracy",
                    **complex_metrics(prediction[:, roi], reference[:, roi]),
                }
            )
    return rows


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    audits = audit(config)
    (output / "convergence.json").write_text(json.dumps(audits, indent=2))
    maximum = max(r["complex_rel_l2"] for r in audits)
    print(
        json.dumps({"max_convergence_error": maximum, "gate": config["convergence_gate"]}),
        flush=True,
    )
    if maximum > config["convergence_gate"]:
        raise ValueError("teacher numerical gate failed; do not generate training supervision")
    systems = config["existing_training_systems"] + list(
        range(config["new_seed_start"], config["new_seed_start"] + config["new_system_count"])
    )
    if len(systems) != len(set(systems)) or set(systems) & set(
        config["excluded_validation_systems"]
    ):
        raise ValueError("duplicate or leaked development system")
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=config["workers"]) as executor:
        futures = [executor.submit(generate_one, (system, config, output)) for system in systems]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            if len(records) % 16 == 0 or len(records) == len(systems):
                print(json.dumps({"completed": len(records), "total": len(systems)}), flush=True)
                (output / "generation.json").write_text(
                    json.dumps(sorted(records, key=lambda r: r["system"]), indent=2)
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
