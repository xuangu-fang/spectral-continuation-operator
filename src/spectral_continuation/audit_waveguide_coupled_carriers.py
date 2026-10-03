"""Does each output mode require several propagation carriers under coupling?

Classical same-background/context R1 test. All candidate coefficient budgets
are matched per output mode. Target coefficient fits are separate oracles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, CubicSpline, interp1d

from .audit_waveguide_spectral_transport import background_modal_carrier, encode_field
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, local_generator, quadrature


def splines(ranges, count):
    degree = 3
    interior = np.linspace(0, 1, count - degree + 1)[1:-1]
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    coordinate = np.asarray(ranges) / ranges[-1]
    return BSpline.design_matrix(coordinate, knots, degree).toarray()


def complex_background(spec, frequency):
    background = json.loads(json.dumps(spec))
    background["contrasts_m_s"] = [0.0] * len(background["contrasts_m_s"])
    z, weights, basis = quadrature(background, 32, 160)
    kr, _ = local_generator(background, frequency, 0, z, weights, basis)
    _, vectors = background_modal_carrier(spec, frequency, 32, "boundary_derivative")
    return kr, vectors


def fit_basis(basis, values, ridge):
    scale = np.maximum(np.sqrt(np.mean(abs(basis) ** 2, axis=0)), 1e-30)
    normalized = basis / scale[None]
    gram = normalized.conj().T @ normalized / len(basis)
    right = normalized.conj().T @ values / len(basis)
    return np.linalg.solve(gram + ridge * np.eye(len(scale)), right) / scale[:, None]


def carrier_basis(ranges, kr, branches, envelope_count):
    b = splines(ranges, envelope_count)
    waves = np.exp(1j * ranges[:, None] * kr[-branches:][None])
    return (waves[:, :, None] * b[:, None]).reshape(len(ranges), -1)


def local_carrier_basis(ranges, kr, output_mode, branches, envelope_count):
    """Bounded local relative phases; always retain the output's own carrier."""
    b = splines(ranges, envelope_count)
    candidates = [output_mode] + [j for j in range(len(kr) - 1, -1, -1) if j != output_mode][
        : branches - 1
    ]
    interior = np.linspace(0, 1, envelope_count - 2)[1:-1]
    knots = np.r_[np.zeros(4), interior, np.ones(4)]
    centers = np.array([np.mean(knots[j + 1 : j + 4]) for j in range(envelope_count)]) * ranges[-1]
    relative = kr.real[candidates] - kr.real[output_mode]
    phase = (ranges[:, None, None] - centers[None, None]) * relative[None, :, None]
    return (b[:, None] * np.exp(1j * phase)).reshape(len(ranges), -1)


def fit_coefficients(modes, ranges, kr, branches, envelope_count, ridge, basis_kind="global_decay"):
    if branches == 0:
        # One own-mode carrier, with the same total coefficients per output.
        demodulated = modes * np.exp(-1j * ranges[:, None] * kr.real[None])
        return fit_basis(splines(ranges, envelope_count), demodulated, ridge)
    if basis_kind == "local_unitary":
        demodulated = modes * np.exp(-1j * ranges[:, None] * kr.real[None])
        return np.concatenate(
            [
                fit_basis(
                    local_carrier_basis(ranges, kr, m, branches, envelope_count),
                    demodulated[:, m : m + 1],
                    ridge,
                )
                for m in range(len(kr))
            ],
            axis=1,
        )
    return fit_basis(carrier_basis(ranges, kr, branches, envelope_count), modes, ridge)


def decode_coefficients(
    coefficients, ranges, kr, branches, envelope_count, basis_kind="global_decay"
):
    if branches == 0:
        envelope = splines(ranges, envelope_count) @ coefficients
        return envelope * np.exp(1j * ranges[:, None] * kr.real[None])
    if basis_kind == "local_unitary":
        envelope = np.column_stack(
            [
                local_carrier_basis(ranges, kr, m, branches, envelope_count) @ coefficients[:, m]
                for m in range(len(kr))
            ]
        )
        return envelope * np.exp(1j * ranges[:, None] * kr.real[None])
    return carrier_basis(ranges, kr, branches, envelope_count) @ coefficients


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    basis_kind = config.get("basis_kind", "global_decay")
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
        roi = ranges >= config["roi_min_range_m"]
        depth_readout = depth_basis(depths, spec["depth_m"], 32)
        frames = [complex_background(spec, f) for f in frequencies]
        context_modes = {
            int(fi): encode_field(fields[fi], depths, spec["depth_m"], 32) @ frames[fi][1]
            for fi in observed
        }
        for variant in config["variants"]:
            branches, count = variant["branches"], variant["envelope_count"]
            for ridge in config["ridges"]:
                context = np.stack(
                    [
                        fit_coefficients(
                            context_modes[int(fi)],
                            ranges,
                            frames[fi][0],
                            branches,
                            count,
                            ridge,
                            basis_kind,
                        )
                        for fi in observed
                    ]
                )
                interpolators = {
                    "linear": interp1d(frequencies[observed], context, axis=0),
                    "cubic": CubicSpline(frequencies[observed], context, axis=0),
                }
                for qi in queries:
                    kr, vectors = frames[qi]
                    for label, interpolator in interpolators.items():
                        predicted_modes = decode_coefficients(
                            interpolator(frequencies[qi]), ranges, kr, branches, count, basis_kind
                        )
                        field = (depth_readout @ vectors) @ predicted_modes.T
                        rows.append(
                            {
                                "system": system,
                                "frequency": float(frequencies[qi]),
                                "branches": branches,
                                "envelope_count": count,
                                "coefficients_per_output_mode": count * max(branches, 1),
                                "ridge": ridge,
                                "method": label,
                                "scope": "known stationary background + 3 complete context fields; no full SSP or query values in inference",
                                **complex_metrics(field[:, roi], fields[qi][:, roi]),
                            }
                        )
                    # Separate query-label capacity fit. Never used above.
                    target_modes = encode_field(fields[qi], depths, spec["depth_m"], 32) @ vectors
                    oracle = fit_coefficients(
                        target_modes, ranges, kr, branches, count, ridge, basis_kind
                    )
                    oracle_modes = decode_coefficients(
                        oracle, ranges, kr, branches, count, basis_kind
                    )
                    field = (depth_readout @ vectors) @ oracle_modes.T
                    rows.append(
                        {
                            "system": system,
                            "frequency": float(frequencies[qi]),
                            "branches": branches,
                            "envelope_count": count,
                            "coefficients_per_output_mode": count * max(branches, 1),
                            "ridge": ridge,
                            "method": "query_truth_coefficients",
                            "scope": "query-truth value projection oracle, NOT deployable or held-coordinate prediction",
                            **complex_metrics(field[:, roi], fields[qi][:, roi]),
                        }
                    )
            print(
                json.dumps(
                    {
                        "system": system,
                        "branches": branches,
                        "envelope_count": count,
                        "errors": {
                            method: float(
                                np.mean(
                                    [
                                        r["complex_rel_l2"]
                                        for r in rows
                                        if r["system"] == system
                                        and r["branches"] == branches
                                        and r["method"] == method
                                        and r["ridge"] == config["ridges"][0]
                                    ]
                                )
                            )
                            for method in ["linear", "cubic", "query_truth_coefficients"]
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
