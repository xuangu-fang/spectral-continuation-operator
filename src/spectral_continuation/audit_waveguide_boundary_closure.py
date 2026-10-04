"""Test boundary-slope asymptotics of high depth harmonics, oracle only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import (
    background_spec,
    generator_geometry,
    true_projection,
)
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, sound_speed


def boundary_derivatives(spec, centers, h):
    """Second-order one-sided differences within physical [0,H] depth."""
    depth = spec["depth_m"]
    if not (0 < h < depth / 4):
        raise ValueError("derivative step outside physical interval")
    background = background_spec(spec)

    def contrast(z):
        location = np.full_like(centers, z, dtype=float)
        return (spec["reference_speed_m_s"] / sound_speed(spec, centers, location)) ** 2 - (
            spec["reference_speed_m_s"] / sound_speed(background, 0, z)
        ) ** 2

    at_zero, at_h, at_2h = (contrast(z) for z in [0.0, h, 2 * h])
    at_depth, at_last, at_before = (contrast(z) for z in [depth, depth - h, depth - 2 * h])
    top = (-3 * at_zero + 4 * at_h - at_2h) / (2 * h)
    bottom = (3 * at_depth - 4 * at_last + at_before) / (2 * h)
    return np.stack([top, bottom], axis=-1)


def harmonic_tail(slopes, depth, start=32, stop=64):
    """Projection onto normalized sqrt(2)cos(m*pi*z/H), leading m^-2 term."""
    if slopes.shape[-1] != 2 or start < 1 or stop <= start:
        raise ValueError("two boundary derivatives and positive high harmonics required")
    modes = np.arange(start, stop)
    return (np.sqrt(2) * depth / (np.pi * modes) ** 2) * (
        (-1.0) ** modes * slopes[..., 1, None] - slopes[..., 0, None]
    )


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    rows, mechanisms = [], []
    for system in config["systems"]:
        with np.load(Path(config["prior32"]) / f"system{system}_cells256_step5.npz") as data:
            beta32 = data["beta"]
        with np.load(Path(config["query_data"]) / f"system{system}_f400.npz") as data:
            spec = json.loads(str(data["spec"]))
        geometry = generator_geometry(
            spec, config["decoder_cells"], 64, config["decoder_modes"], config["quadrature_nodes"]
        )
        centers = geometry["middle"]
        slopes = boundary_derivatives(spec, centers, config["derivative_step_m"])
        half = boundary_derivatives(spec, centers, config["derivative_step_m"] / 2)
        derivative_rel = float(np.linalg.norm(slopes - half) / max(np.linalg.norm(half), 1e-20))
        if derivative_rel > 1e-5:
            raise ValueError("boundary derivative finite difference unstable")
        truth_beta = true_projection(spec, geometry)
        tail = harmonic_tail(slopes, spec["depth_m"])
        expanded = np.pad(beta32, ((0, 0), (0, 32)))
        oracle = expanded.copy()
        oracle[:, 32:] = tail
        np.testing.assert_allclose(
            beta32 @ geometry["psi"][:, :32].T,
            expanded @ geometry["psi"].T,
            atol=1e-13,
        )
        mechanism = {
            "system": system,
            "derivative_step_rel": derivative_rel,
            "true_tail_norm": float(np.linalg.norm(truth_beta[:, 32:])),
            "boundary_tail_norm": float(np.linalg.norm(tail)),
            "tail_relative_coefficient_error": float(
                np.linalg.norm(tail - truth_beta[:, 32:])
                / max(np.linalg.norm(truth_beta[:, 32:]), 1e-20)
            ),
            "tail_rms_absolute": float(np.sqrt(np.mean((tail - truth_beta[:, 32:]) ** 2))),
        }
        mechanisms.append(mechanism)
        (output / "mechanism.json").write_text(json.dumps(mechanisms, indent=2))
        np.savez_compressed(
            output / f"system{system}_oracle.npz",
            beta32=beta32,
            truth_beta64=truth_beta,
            boundary_slopes_oracle=slopes,
            analytic_tail_oracle=tail,
        )
        for frequency in config["frequencies"]:
            with np.load(Path(config["query_data"]) / f"system{system}_f{frequency}.npz") as data:
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
            roi = ranges >= config["roi_min_range_m"]
            truth_field = np.einsum(
                "dm,rms->sdr",
                depth_basis(depths, spec["depth_m"], truth.shape[1]),
                truth,
                optimize=True,
            )
            for method, beta, extra in [
                ("zero_high_terms", expanded, False),
                ("true_boundary_derivative_oracle", oracle, True),
                ("true_medium64_oracle", truth_beta, True),
            ]:
                start = time.perf_counter()
                prediction = propagate_code(
                    spec,
                    frequency,
                    ranges,
                    config["decoder_cells"],
                    config["decoder_modes"],
                    config["quadrature_nodes"],
                    config["eigh_device"],
                    config["eigh_batch"],
                    beta,
                )
                elapsed = time.perf_counter() - start
                field = np.einsum(
                    "dm,rms->sdr",
                    depth_basis(depths, spec["depth_m"], prediction.shape[1]),
                    prediction,
                    optimize=True,
                )
                modal = modal_raw(prediction[roi], truth[roi])
                for si, source in enumerate(config["source_variants"]):
                    row = {
                        "system": system,
                        "frequency": frequency,
                        "source": source,
                        "method": method,
                        "oracle": extra,
                        "modal_parseval_raw": float(modal[si]),
                        "query_only_seconds": elapsed,
                        **complex_metrics(field[si][:, roi], truth_field[si][:, roi]),
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "system": system,
                                "frequency": frequency,
                                "source": source,
                                "method": method,
                                "raw": row["complex_rel_l2"],
                            }
                        ),
                        flush=True,
                    )
                (output / "results.json").write_text(json.dumps(rows, indent=2))
    summary = []
    for method in ["zero_high_terms", "true_boundary_derivative_oracle", "true_medium64_oracle"]:
        for frequency in config["frequencies"]:
            for source in config["source_variants"]:
                selected = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["frequency"] == frequency
                    and r["source"] == source
                ]
                summary.append(
                    {
                        "method": method,
                        "frequency": frequency,
                        "source": source,
                        "systems": len(selected),
                        **{
                            key: float(np.mean([r[key] for r in selected]))
                            for key in [
                                "complex_rel_l2",
                                "energy_weighted_phase_rmse_rad",
                                "amplitude_rel_l2",
                                "modal_parseval_raw",
                            ]
                        },
                    }
                )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    decisions = []
    for source in config["source_variants"]:
        old = np.array(
            [
                next(
                    r["complex_rel_l2"]
                    for r in rows
                    if r["method"] == "zero_high_terms"
                    and r["frequency"] == 800
                    and r["source"] == source
                    and r["system"] == s
                )
                for s in config["systems"]
            ]
        )
        new = np.array(
            [
                next(
                    r["complex_rel_l2"]
                    for r in rows
                    if r["method"] == "true_boundary_derivative_oracle"
                    and r["frequency"] == 800
                    and r["source"] == source
                    and r["system"] == s
                )
                for s in config["systems"]
            ]
        )
        decisions.append(
            {
                "source": source,
                "relative_improvement": float(1 - new.mean() / old.mean()),
                "systems_improved": int(np.count_nonzero(new < old)),
                "go": bool(
                    1 - new.mean() / old.mean() >= config["m1_entry_min_relative_improvement"]
                    and np.count_nonzero(new < old) >= config["m1_entry_min_system_wins"]
                ),
            }
        )
    (output / "decision.json").write_text(json.dumps(decisions, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
