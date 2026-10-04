"""Classical local identification of a shared frequency-affine generator.

Only observed fields and stationary background enter the identifier. This is
an approximate second-order Helmholtz inverse: the exact one-way dynamics also
contain i K'(r) a. It must be measured, not assumed negligible. True medium
projection is separately labelled an oracle and is never an encoder input.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.sparse.linalg import spsolve

from .audit_waveguide_spectral_transport import background_modal_carrier, encode_field
from .coarse_waveguide_operator import query_field
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import quadrature, sound_speed


def background_spec(spec):
    return {**spec, "contrasts_m_s": [0.0] * len(spec["contrasts_m_s"])}


def generator_geometry(spec, cells=64, depth_terms=16, modes=32, quadrature_nodes=160):
    """Known-background geometry only; no hidden contrast in returned arrays."""
    background = background_spec(spec)
    z, weights, basis = quadrature(background, modes, quadrature_nodes)
    psi = np.cos(np.pi * z[:, None] * np.arange(depth_terms)[None] / spec["depth_m"])
    psi[:, 1:] *= np.sqrt(2)
    matrices = np.einsum("zm,zd,z,zn->dmn", basis, psi, weights, basis)
    background_q = (spec["reference_speed_m_s"] / sound_speed(background, 0, z)) ** 2
    mass0 = basis.T @ ((weights * background_q)[:, None] * basis)
    edges = np.linspace(0, spec["range_m"], cells + 1)
    return {
        "z": z,
        "weights": weights,
        "basis": basis,
        "psi": psi,
        "matrices": matrices,
        "background_q": background_q,
        "mass0": mass0,
        "middle": (edges[:-1] + edges[1:]) / 2,
        "edges": edges,
        "kz2": ((np.arange(modes) + 0.5) * np.pi / spec["depth_m"]) ** 2,
        "reference_speed": spec["reference_speed_m_s"],
        "depth": spec["depth_m"],
        "attenuation": spec["attenuation_per_m"],
    }


def envelope_derivatives(coefficients, ranges, positions, kr, vectors, order=3):
    """Analytic derivatives of the same background-demodulated cubic codec."""
    curve = (coefficients @ vectors) * np.exp(-1j * ranges[:, None] * kr[None])
    spline = (
        CubicSpline(ranges, curve, axis=0)
        if order == 3
        else make_interp_spline(ranges, curve, k=order, axis=0)
    )
    b, db, ddb = (spline(positions, order) for order in range(3))
    phase = np.exp(1j * positions[:, None] * kr[None])
    values = (phase * b) @ vectors.T
    first = (phase * (db + 1j * kr * b)) @ vectors.T
    second = (phase * (ddb + 2j * kr * db - kr**2 * b)) @ vectors.T
    return values, first, second


def normal_statistics(spec, context_coefficients, frequencies, ranges, geometry, order=3):
    """Real local LS statistics. No query fields, true speed or target labels."""
    designs, right = [], []
    alpha, c0 = geometry["attenuation"], geometry["reference_speed"]
    for frequency, coefficients in zip(frequencies, context_coefficients, strict=True):
        kr, vectors = background_modal_carrier(
            spec, frequency, coefficients.shape[-1], "boundary_derivative"
        )
        a, da, dda = envelope_derivatives(
            coefficients, ranges, geometry["middle"], kr, vectors, order
        )
        omega2 = (2 * np.pi * frequency) ** 2
        c_background = omega2 / c0**2 * geometry["mass0"] - np.diag(geometry["kz2"])
        residual = dda + 2 * alpha * da + alpha**2 * a + a @ c_background.T
        design = np.einsum("dmn,rn->rmd", geometry["matrices"], a)
        designs.append(design)
        right.append(-residual * c0**2 / omega2)
    design = np.concatenate(designs, axis=1)
    right = np.concatenate(right, axis=1)
    gram = np.einsum("rmd,rme->rde", design.conj(), design).real
    rhs = np.einsum("rmd,rm->rd", design.conj(), right).real
    scale = np.maximum(np.trace(gram, axis1=-2, axis2=-1) / gram.shape[-1], 1e-20)
    return gram / scale[:, None, None], rhs / scale[:, None]


def identify(gram, rhs, ridge, range_penalty=0.0):
    # Higher vertical frequencies regularized, with a small penalty for DC too.
    penalty = (1 + np.arange(rhs.shape[-1]) ** 2) ** 2
    matrix = gram + ridge * np.diag(penalty)[None]
    if range_penalty:
        cells, terms = rhs.shape
        second = sparse.diags(
            [np.ones(cells - 2), -2 * np.ones(cells - 2), np.ones(cells - 2)],
            [0, 1, 2],
            shape=(cells - 2, cells),
        )
        smooth = sparse.kron(second.T @ second, sparse.eye(terms))
        full = sparse.block_diag(matrix) + range_penalty * smooth
        return spsolve(full.tocsc(), rhs.reshape(-1)).reshape(rhs.shape)
    return np.linalg.solve(matrix, rhs[..., None])[..., 0]


def true_projection(spec, geometry):
    """ORACLE only. Do not use these values in deployment identification."""
    q = (
        geometry["reference_speed"]
        / sound_speed(spec, geometry["middle"][:, None], geometry["z"][None])
    ) ** 2 - geometry["background_q"]
    return (q * geometry["weights"]) @ geometry["psi"] / geometry["depth"]


def latent_from_coefficients(coefficient, initial, geometry):
    total = geometry["background_q"] + coefficient @ geometry["psi"].T
    clipped = np.clip(total, 0.65, 1.5)
    mass = np.einsum(
        "zm,rz,zn->rmn",
        geometry["basis"],
        clipped * geometry["weights"][None] / geometry["reference_speed"] ** 2,
        geometry["basis"],
    )
    latent = {
        "mass": mass,
        "initial": initial,
        "edges": geometry["edges"],
        "kz2": geometry["kz2"],
        "depth": geometry["depth"],
        "attenuation": geometry["attenuation"],
    }
    return latent, float(np.mean(total != clipped))


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows, diagnostics = [], []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system{system}_m32_r2049.npz") as z:
            spec = json.loads(str(z["spec"]))
            fields, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        coefficients = np.stack(
            [encode_field(fields[i], depths, spec["depth_m"], 32) for i in observed]
        )
        initial = coefficients[0, 0]
        geometry = generator_geometry(spec, config["cells"], config["depth_terms"])
        oracle = true_projection(spec, geometry)
        methods = [("true_medium_projection_oracle", 0, 0.0, oracle, 0.0)]
        for samples in config["context_range_samples"]:
            take = np.linspace(0, len(ranges) - 1, samples, dtype=int)
            start = time.perf_counter()
            gram, rhs = normal_statistics(
                spec, coefficients[:, take], frequencies[observed], ranges[take], geometry
            )
            stats_seconds = time.perf_counter() - start
            for ridge in config["ridges"]:
                start = time.perf_counter()
                estimate = identify(gram, rhs, ridge)
                seconds = time.perf_counter() - start + stats_seconds
                methods.append(("context_second_order_inverse", samples, ridge, estimate, seconds))
                diagnostics.append(
                    {
                        "system": system,
                        "context_range_samples": samples,
                        "ridge": ridge,
                        "normalized_slowness_coefficient_rmse": float(
                            np.sqrt(np.mean((estimate - oracle) ** 2))
                        ),
                        "oracle_second_order_residual": float(
                            np.linalg.norm(np.einsum("rde,re->rd", gram, oracle) - rhs)
                            / max(np.linalg.norm(rhs), 1e-20)
                        ),
                    }
                )
        roi = ranges >= config["roi_min_range_m"]
        for method, samples, ridge, coefficient, seconds in methods:
            latent, clip_rate = latent_from_coefficients(coefficient, initial, geometry)
            for frequency in config["query_frequencies"]:
                fi = int(np.flatnonzero(frequencies == frequency)[0])
                start = time.perf_counter()
                prediction = query_field(latent, frequency, ranges, depths)
                rows.append(
                    {
                        "system": system,
                        "method": method,
                        "context_range_samples": samples,
                        "ridge": ridge,
                        "frequency": frequency,
                        "identification_seconds": seconds,
                        "query_seconds": time.perf_counter() - start,
                        "positivity_clipping_fraction": clip_rate,
                        **complex_metrics(prediction[:, roi], fields[fi][:, roi]),
                    }
                )
            print(
                json.dumps(
                    {
                        "system": system,
                        "method": method,
                        "samples": samples,
                        "ridge": ridge,
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
        (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
