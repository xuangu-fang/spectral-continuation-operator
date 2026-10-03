"""Exact local one-way generator inversion versus approximate second-order ID.

The earlier linear inverse drops i K'(r) a. This audit uses a'=(iK-alpha)a
directly, with an analytic square-root Frechet derivative. Context interpolation
can still be inaccurate; this is not assumed to solve identifiability.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .audit_waveguide_generator_identification import envelope_derivatives, identify
from .audit_waveguide_spectral_transport import background_modal_carrier, encode_field
from .learned_waveguide_generator import error, evaluate, prepare


def generator_action(beta, values, frequency, geometry, with_jacobian=True):
    mass = geometry["mass0"] + np.einsum("rd,dmn->rmn", beta, geometry["matrices"])
    scale = (2 * np.pi * frequency / geometry["reference_speed"]) ** 2
    lam, vectors = np.linalg.eigh(scale * mass - np.diag(geometry["kz2"]))
    root = np.sqrt(lam.astype(complex))
    projected = np.einsum("rmi,rm->ri", vectors, values)
    action = 1j * np.einsum("rmi,ri->rm", vectors, root * projected)
    action -= geometry["attenuation"] * values
    if not with_jacobian:
        return action, None
    denominator = root[:, :, None] + root[:, None, :]
    if np.min(np.abs(denominator)) < 1e-10:
        raise FloatingPointError("near-cutoff square-root derivative; do not silently regularize")
    transformed = scale * np.einsum(
        "rmi,dmn,rnj->rdij", vectors, geometry["matrices"], vectors, optimize=True
    )
    divided = transformed / denominator[:, None]
    jacobian = 1j * np.einsum("rmi,rdij,rj->rmd", vectors, divided, projected, optimize=True)
    return action, jacobian


def run(options, output):
    base = json.loads(Path(options["base_config"]).read_text())
    base["validation_systems"] = options["systems"]
    base["interpolation_order"], base["range_penalty"] = 5, 1
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({**options, "base": base}, indent=2))
    val, metadata = prepare(base, options["systems"], False)
    rows, diagnostics = [], []
    for order in options["derivative_orders"]:
        interpolants = []
        for item in metadata:
            # Original environment spec is used ONLY to construct known-background carriers.
            with np.load(Path(base["data"]) / f"system{item['system']}_m32_r2049.npz") as z:
                spec = json.loads(str(z["spec"]))
            take = np.linspace(0, len(item["ranges"]) - 1, base["context_range_samples"], dtype=int)
            contexts = []
            for frequency in base["observed_frequencies"]:
                fi = int(np.flatnonzero(item["frequencies"] == frequency)[0])
                modal = encode_field(
                    item["fields"][fi][:, take], item["depths"], spec["depth_m"], 32
                )
                kr, vectors = background_modal_carrier(spec, frequency, 32, "boundary_derivative")
                a, da, _ = envelope_derivatives(
                    modal, item["ranges"][take], item["geometry"]["middle"], kr, vectors, order
                )
                contexts.append((a, da))
            interpolants.append(contexts)
        for penalty in options["range_penalties"]:
            betas = val["baseline"].astype(float).copy()
            elapsed = 0.0
            for step in range(max(options["checkpoints"]) + 1):
                if step in options["checkpoints"]:
                    current = evaluate(betas, metadata, base)
                    record = {
                        "order": order,
                        "range_penalty": penalty,
                        "step": step,
                        "macro_error": error(current),
                        "optimization_seconds": elapsed,
                    }
                    diagnostics.append(record)
                    print(json.dumps(record), flush=True)
                    rows.extend(
                        {**r, "order": order, "range_penalty": penalty, "step": step}
                        for r in current
                    )
                    (output / "results.json").write_text(json.dumps(rows, indent=2))
                    (output / "history.json").write_text(json.dumps(diagnostics, indent=2))
                    np.savez_compressed(
                        output / f"order{order}_penalty{penalty}_step{step}.npz",
                        coefficients=betas,
                        systems=options["systems"],
                    )
                if step == max(options["checkpoints"]):
                    break
                start = time.perf_counter()
                for si, item in enumerate(metadata):
                    design, residual = [], []
                    for frequency, (a, da) in zip(
                        base["observed_frequencies"], interpolants[si], strict=True
                    ):
                        prediction, jacobian = generator_action(
                            betas[si], a, frequency, item["geometry"]
                        )
                        design.append(jacobian)
                        residual.append(da - prediction)
                    design = np.concatenate(design, axis=1)
                    residual = np.concatenate(residual, axis=1)
                    gram = np.einsum("rmd,rme->rde", design.conj(), design).real
                    rhs = np.einsum("rmd,rm->rd", design.conj(), residual).real
                    scale = np.maximum(
                        np.trace(gram, axis1=-2, axis2=-1) / base["depth_terms"], 1e-20
                    )
                    gram, rhs = gram / scale[:, None, None], rhs / scale[:, None]
                    proposal = identify(
                        gram,
                        rhs + np.einsum("rde,re->rd", gram, betas[si]),
                        options["ridge"],
                        penalty,
                    )
                    delta = proposal - betas[si]
                    safety = min(
                        1.0,
                        options["max_coefficient_step"] / max(float(np.max(np.abs(delta))), 1e-20),
                    )
                    betas[si] += options["damping"] * safety * delta
                elapsed += time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
