"""One predeclared 64-depth-term, 128-code development diagnostic."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_continuous_generator import fit_extended_prior, resample_generator
from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_trajectory_newton import newton_step
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    base = json.loads((Path(config["training_run"]) / "config.json").read_text())
    if config["systems"] != base["validation_systems"]:
        raise ValueError("only original development systems allowed")
    mean, factor = fit_extended_prior(
        base,
        {
            "training_range_centers": config["training_range_centers"],
            "depth_terms": config["depth_terms"],
            "rank": config["rank"],
        },
        output,
    )
    factor64 = factor[:, :32].reshape(-1, config["rank"])
    fine_mean, fine_factor = (
        resample_generator(mean, config["decoder_cells"]),
        resample_generator(factor, config["decoder_cells"]),
    )
    inverse = np.linalg.pinv(fine_factor.reshape(-1, config["rank"]))
    rows, history, projection = [], [], []
    for system in config["systems"]:
        prior_path = Path(config["initial32"]) / f"system{system}_cells256_step5.npz"
        with np.load(prior_path) as prior:
            beta32 = prior["beta"]
        path = Path(base["data"]) / f"system{system}_m32_r2049.npz"
        with np.load(path) as original:
            spec = json.loads(str(original["spec"]))
            take = np.linspace(
                0, len(original["ranges_m"]) - 1, config["context_profiles"], dtype=int
            )
            frequencies = original["frequencies_hz"]
            indices = [
                int(np.flatnonzero(frequencies == f)[0]) for f in config["context_frequencies"]
            ]
            observed = original["modal_coefficients"][indices][:, take]
            context_ranges = original["ranges_m"][take]
        if beta32.shape != (config["decoder_cells"], 32):
            raise ValueError("expected frozen32 code shape")
        old_at_centers = resample_generator(beta32, config["training_range_centers"])
        coordinates = np.linalg.lstsq(
            factor64, (old_at_centers - mean[:, :32]).ravel(), rcond=None
        )[0]
        beta = fine_mean + np.einsum("rdk,k->rd", fine_factor, coordinates)
        projection.append(
            {"system": system, "low32_rmse": float(np.sqrt(np.mean((beta[:, :32] - beta32) ** 2)))}
        )
        (output / "initial_projection.json").write_text(json.dumps(projection, indent=2))
        geometry = generator_geometry(spec, config["decoder_cells"], config["depth_terms"])
        obsindices = np.linspace(0, config["decoder_cells"], config["context_profiles"])
        if not np.array_equal(obsindices, obsindices.astype(int)):
            raise ValueError("profile positions must coincide with decoder edges")
        options = {
            "ridge_relative": config["ridge_relative"],
            "max_coefficient_step": config["max_coefficient_step"],
            "line_search": config["line_search"],
            "observation_indices": obsindices.astype(int),
        }
        roi = context_ranges >= config["roi_min_range_m"]
        started = time.perf_counter()
        for step in range(config["steps"]):
            beta, diagnostic = newton_step(
                beta,
                fine_factor,
                observed[0, 0],
                geometry,
                config["context_frequencies"],
                observed,
                roi,
                options,
            )
            history.append({"system": system, "step": step + 1, **diagnostic})
        correction_seconds = time.perf_counter() - started
        np.savez_compressed(output / f"system{system}_beta.npz", beta=beta)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        true64 = true_projection(spec, geometry)
        true_coordinates = inverse @ (true64 - fine_mean).ravel()
        capacity = fine_mean + np.einsum("rdk,k->rd", fine_factor, true_coordinates)
        for frequency in config["query_frequencies"]:
            target_path = Path(config["frequency_data"]) / f"system{system}_f{frequency}.npz"
            with np.load(target_path) as data:
                truth = data["modal_coefficients"]
                ranges, depths = data["ranges_m"], data["depths_m"]
            truth_field = np.einsum(
                "dm,rms->sdr",
                depth_basis(depths, spec["depth_m"], truth.shape[1]),
                truth,
                optimize=True,
            )
            roi = ranges >= config["roi_min_range_m"]
            for method, current, oracle in [
                ("prior64_inferred", beta, False),
                ("prior64_capacity_oracle", capacity, True),
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
                    current,
                )
                query_seconds = time.perf_counter() - start
                field = np.einsum(
                    "dm,rms->sdr",
                    depth_basis(depths, spec["depth_m"], prediction.shape[1]),
                    prediction,
                    optimize=True,
                )
                modal = modal_raw(prediction[roi], truth[roi])
                for source_index, source in enumerate(config["source_variants"]):
                    row = {
                        "system": system,
                        "frequency": frequency,
                        "source": source,
                        "method": method,
                        "oracle": oracle,
                        "modal_parseval_raw": float(modal[source_index]),
                        "correction_seconds": correction_seconds if not oracle else None,
                        "query_only_seconds": query_seconds,
                        **complex_metrics(
                            field[source_index][:, roi], truth_field[source_index][:, roi]
                        ),
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
    grouped = {}
    for row in rows:
        key = (row["method"], row["frequency"], row["source"])
        grouped.setdefault(key, []).append(row)
    summary = [
        {
            "method": key[0],
            "frequency": key[1],
            "source": key[2],
            "systems": len(subset),
            **{
                name: float(np.mean([r[name] for r in subset]))
                for name in [
                    "complex_rel_l2",
                    "energy_weighted_phase_rmse_rad",
                    "amplitude_rel_l2",
                    "modal_parseval_raw",
                    "query_only_seconds",
                ]
            },
        }
        for key, subset in grouped.items()
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
