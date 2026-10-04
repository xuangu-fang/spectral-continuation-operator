"""Frozen physical code queried at higher frequencies, with numerical audits.

No new code inference or model selection. Tensor eigendecomposition is only
a computational backend; the physical teacher remains explicitly one-way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_generator_identification import (
    background_spec,
    generator_geometry,
    true_projection,
)
from .audit_waveguide_operator_action import propagate_actions
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import (
    depth_basis,
    environment,
    local_generator,
    quadrature,
    sound_speed,
)


def spectral_batches(q, basis, weights, kz2, frequency, device="cpu", batch=64):
    """q is dimensional slowness squared; symmetric float64 matrix function."""
    tensor_basis = torch.as_tensor(basis, dtype=torch.float64, device=device)
    diagonal = torch.diag(torch.as_tensor(kz2, dtype=torch.float64, device=device))
    roots, vectors = [], []
    for start in range(0, len(q), batch):
        wq = torch.as_tensor(q[start : start + batch] * weights, device=device)
        matrix = (2 * np.pi * frequency) ** 2 * (
            tensor_basis.T @ (wq[:, :, None] * tensor_basis)
        ) - diagonal
        values, rotation = torch.linalg.eigh(matrix)
        roots.append(np.sqrt(values.cpu().numpy().astype(complex)))
        vectors.append(rotation.cpu().numpy())
    return np.concatenate(roots), np.concatenate(vectors)


def source_columns(spec, z, weights, basis):
    widths = [spec["source_width_m"], 2.0]
    return basis.T @ (
        weights[:, None]
        * np.exp(-0.5 * ((z[:, None] - spec["source_depth_m"]) / np.array(widths)) ** 2)
    )


def propagate_code(spec, frequency, ranges, cells, modes, nodes, device, batch, beta=None):
    z, weights, basis = quadrature(spec, modes, nodes)
    edges = np.linspace(0, spec["range_m"], cells + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    if beta is None:
        q = 1 / sound_speed(spec, centers[:, None], z[None]) ** 2
    else:
        if len(beta) != cells:
            raise ValueError("generator and decoder cell counts differ")
        psi = np.cos(np.pi * z[:, None] * np.arange(beta.shape[1]) / spec["depth_m"])
        psi[:, 1:] *= np.sqrt(2)
        total = (spec["reference_speed_m_s"] / sound_speed(background_spec(spec), 0, z)) ** 2
        total = total[None] + beta @ psi.T
        if np.any((total <= 0.65) | (total >= 1.5)):
            raise ValueError("generator clipping forbidden in this audit")
        q = total / spec["reference_speed_m_s"] ** 2
    kz2 = ((np.arange(modes) + 0.5) * np.pi / spec["depth_m"]) ** 2
    roots, vectors = spectral_batches(q, basis, weights, kz2, frequency, device, batch)
    return propagate_actions(
        edges,
        source_columns(spec, z, weights, basis),
        ranges,
        lambda cell: (roots[cell], vectors[cell]),
        spec["attenuation_per_m"],
    )


def modal_raw(prediction, truth):
    if prediction.shape[0] != truth.shape[0] or prediction.shape[-1] != truth.shape[-1]:
        raise ValueError("matched range/source dimensions required")
    if prediction.shape[1] > truth.shape[1]:
        raise ValueError("truth mode count must dominate prediction")
    padded = np.zeros_like(truth)
    padded[:, : prediction.shape[1]] = prediction
    return np.linalg.norm(padded - truth, axis=(0, 1)) / np.linalg.norm(truth, axis=(0, 1))


def backend_audit(config):
    spec = environment(config["audit_systems"][0])
    modes, cells, nodes = 48, 16, 160
    ranges = np.linspace(0, spec["range_m"], 33)
    z, weights, basis = quadrature(spec, modes, nodes)
    edges = np.linspace(0, spec["range_m"], cells + 1)
    reference = propagate_actions(
        edges,
        source_columns(spec, z, weights, basis),
        ranges,
        lambda cell: local_generator(
            spec, 400, (edges[cell] + edges[cell + 1]) / 2, z, weights, basis
        ),
        spec["attenuation_per_m"],
    )
    candidate = propagate_code(
        spec, 400, ranges, cells, modes, nodes, config["eigh_device"], config["eigh_batch"]
    )
    error = modal_raw(candidate, reference)
    if max(error) > 1e-8:
        raise ValueError(f"independent SciPy backend mismatch {error}")
    return {"scipy_comparison_raw": error.tolist(), "frequency": 400, "modes": modes}


def decode_coordinates(directory, system, mean, factor, factor_inverse):
    path = directory / f"system{system}_cells256_step5.npz"
    with np.load(path) as data:
        beta = data["beta"]
    fine_mean, fine_factor = resample_generator(mean, 256), resample_generator(factor, 256)
    coordinates = factor_inverse @ (beta - fine_mean).ravel()
    np.testing.assert_allclose(
        beta, fine_mean + np.einsum("rdk,k->rd", fine_factor, coordinates), atol=1e-10, rtol=1e-8
    )
    return coordinates, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(config["data_output"])
    data_root.mkdir(parents=True, exist_ok=False)
    for directory in [output, data_root]:
        (directory / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    if config["eigh_device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("declared CUDA backend unavailable; no silent config change")
    backend = backend_audit(config)
    backend.update(
        {
            "device": config["eigh_device"],
            "torch_version": torch.__version__,
            "hardware": torch.cuda.get_device_name() if config["eigh_device"] == "cuda" else "cpu",
        }
    )
    (output / "backend.json").write_text(json.dumps(backend, indent=2))
    convergence = []
    for system in config["audit_systems"]:
        spec = environment(system)
        ranges = (np.arange(513) + 0.37) / 513 * spec["range_m"]
        arrays = {}
        for frequency in config["audit_frequencies"]:
            start = time.perf_counter()
            lower = propagate_code(
                spec,
                frequency,
                ranges,
                config["teacher_cells"],
                config["teacher_modes"],
                config["quadrature_nodes"],
                config["eigh_device"],
                config["eigh_batch"],
            )
            higher = propagate_code(
                spec,
                frequency,
                ranges,
                config["audit_cells"],
                config["audit_modes"],
                config["audit_quadrature_nodes"],
                config["eigh_device"],
                config["eigh_batch"],
            )
            roi = ranges >= config["roi_min_range_m"]
            errors = modal_raw(lower[roi], higher[roi])
            row = {
                "system": system,
                "frequency": frequency,
                "raw_by_source": errors.tolist(),
                "seconds": time.perf_counter() - start,
            }
            convergence.append(row)
            print(json.dumps({"convergence": row}), flush=True)
            arrays[f"coarse{frequency}"] = lower.astype(np.complex64)
            arrays[f"fine{frequency}"] = higher.astype(np.complex64)
            np.savez_compressed(
                data_root / f"audit{system}.npz", **arrays, ranges_m=ranges, spec=json.dumps(spec)
            )
            (data_root / "convergence.json").write_text(json.dumps(convergence, indent=2))
            if max(errors) >= config["convergence_gate"]:
                raise ValueError("frequency-scaling numerical gate failed; outputs retained")
    prior_dir = Path(config["continuous32"])
    with np.load(prior_dir / "training_prior.npz") as data:
        mean, factor = data["mean"], data["factor"]
    factor_inverse = np.linalg.pinv(resample_generator(factor, 256).reshape(-1, factor.shape[-1]))
    records, rows, sources = [], [], []
    for system in config["systems"]:
        spec = environment(system)
        ranges = (np.arange(config["range_count"]) + 0.37) / config["range_count"] * spec["range_m"]
        depths = np.linspace(0.1, spec["depth_m"] - 0.1, config["depth_count"])
        coordinates, source = decode_coordinates(prior_dir, system, mean, factor, factor_inverse)
        sources.append({"system": system, **source})
        candidates = []
        beta256 = resample_generator(mean, 256) + np.einsum(
            "rdk,k->rd", resample_generator(factor, 256), coordinates
        )
        for modes in config["prediction_modes"]:
            candidates.append((f"frozen_code_cells256_modes{modes}", False, 256, modes, beta256))
        beta512 = resample_generator(mean, 512) + np.einsum(
            "rdk,k->rd", resample_generator(factor, 512), coordinates
        )
        candidates.append(("frozen_code_cells512_modes128", False, 512, 128, beta512))
        for terms in [32, 64]:
            geometry = generator_geometry(spec, 512, terms, 128, config["quadrature_nodes"])
            candidates.append(
                (f"true_medium{terms}_oracle", True, 512, 128, true_projection(spec, geometry))
            )
        candidates.append(("full_medium_oracle", True, 512, 128, None))
        for frequency in config["frequencies"]:
            start = time.perf_counter()
            truth = propagate_code(
                spec,
                frequency,
                ranges,
                config["teacher_cells"],
                config["teacher_modes"],
                config["quadrature_nodes"],
                config["eigh_device"],
                config["eigh_batch"],
            )
            truth_path = data_root / f"system{system}_f{frequency}.npz"
            np.savez_compressed(
                truth_path,
                modal_coefficients=truth.astype(np.complex64),
                ranges_m=ranges,
                depths_m=depths,
                frequency_hz=frequency,
                source_variants=config["source_variants"],
                spec=json.dumps(spec),
                phase_convention="exp(-i omega t), positive forward phase; real Gaussian source",
                evidence=config["scope"],
            )
            records.append(
                {
                    "system": system,
                    "frequency": frequency,
                    "path": str(truth_path),
                    "sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
                    "seconds": time.perf_counter() - start,
                    "shape": list(truth.shape),
                }
            )
            (data_root / "generation.json").write_text(json.dumps(records, indent=2))
            basis_truth = depth_basis(depths, spec["depth_m"], truth.shape[1])
            truth_fields = np.einsum("dm,rms->sdr", basis_truth, truth, optimize=True)
            predicted_arrays = {}
            for method, oracle, cells, modes, beta in candidates:
                start = time.perf_counter()
                prediction = propagate_code(
                    spec,
                    frequency,
                    ranges,
                    cells,
                    modes,
                    config["quadrature_nodes"],
                    config["eigh_device"],
                    config["eigh_batch"],
                    beta,
                )
                seconds = time.perf_counter() - start
                predicted_arrays[method] = prediction.astype(np.complex64)
                predicted_fields = np.einsum(
                    "dm,rms->sdr",
                    depth_basis(depths, spec["depth_m"], modes),
                    prediction,
                    optimize=True,
                )
                for limit in config["range_limits"]:
                    roi = (ranges >= config["roi_min_range_m"]) & (ranges <= limit)
                    raw = modal_raw(prediction[roi], truth[roi])
                    for source_index, variant in enumerate(config["source_variants"]):
                        rows.append(
                            {
                                "system": system,
                                "frequency": frequency,
                                "method": method,
                                "oracle": oracle,
                                "source": variant,
                                "range_limit_m": limit,
                                "modal_parseval_raw": float(raw[source_index]),
                                "query_only_seconds": seconds,
                                **complex_metrics(
                                    predicted_fields[source_index][:, roi],
                                    truth_fields[source_index][:, roi],
                                ),
                            }
                        )
                print(
                    json.dumps(
                        {
                            "system": system,
                            "frequency": frequency,
                            "method": method,
                            "raw": [r["complex_rel_l2"] for r in rows[-2:]],
                            "seconds": seconds,
                        }
                    ),
                    flush=True,
                )
                (output / "results.json").write_text(json.dumps(rows, indent=2))
            np.savez_compressed(
                output / f"system{system}_f{frequency}_predictions.npz",
                **predicted_arrays,
                ranges_m=ranges,
                depths_m=depths,
                source_variants=config["source_variants"],
            )
        (output / "frozen_code_sources.json").write_text(json.dumps(sources, indent=2))
    keys = ["method", "oracle", "frequency", "source", "range_limit_m"]
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for key, selected in groups.items():
        summary.append(
            {
                **dict(zip(keys, key, strict=True)),
                "systems": len(selected),
                **{
                    metric: float(np.mean([r[metric] for r in selected]))
                    for metric in [
                        "complex_rel_l2",
                        "energy_weighted_phase_rmse_rad",
                        "amplitude_rel_l2",
                        "modal_parseval_raw",
                        "query_only_seconds",
                    ]
                },
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
