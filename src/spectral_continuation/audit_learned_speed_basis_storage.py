"""Lossless-if-supported depth-factor storage audit for a FROZEN speed prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_generator_identification import generator_geometry
from .audit_waveguide_spectral_context_robustness import digest
from .learned_speed_manifold import speed_projection


def factorize_depth(mean: np.ndarray, factor: np.ndarray, rank: int):
    if mean.ndim != 2 or factor.shape[:2] != mean.shape:
        raise ValueError("mean/factor must share a range-depth grid")
    if not 0 < rank <= mean.shape[1]:
        raise ValueError("invalid depth rank")
    cells, depths = mean.shape
    codes = factor.shape[-1]
    curves = np.concatenate([
        mean.reshape(cells, depths),
        factor.transpose(0, 2, 1).reshape(cells * codes, depths),
    ])
    _, singular, right = np.linalg.svd(curves, full_matrices=False)
    depth_basis = right[:rank].T
    mean_core = mean @ depth_basis
    factor_core = np.einsum("rzk,zj->rjk", factor, depth_basis, optimize=True)
    return mean_core, factor_core, depth_basis, singular


def reconstruct(mean_core: np.ndarray, factor_core: np.ndarray,
                depth_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = mean_core @ depth_basis.T
    factor = np.einsum("rjk,zj->rzk", factor_core, depth_basis, optimize=True)
    return mean, factor


def run(config: dict, output: Path) -> None:
    if digest(config["frozen_priors"]) != config["frozen_priors_sha256"]:
        raise ValueError("frozen shared prior changed")
    with np.load(config["frozen_priors"]) as data:
        mean = data["mean_speed"]
        factor = data["factor_speed"]
        q_mean = data["mean_q"]
        q_factor = data["factor_q"]
    mean_core, factor_core, depth_basis, singular = factorize_depth(
        mean, factor, config["depth_rank"]
    )
    reconstructed_mean, reconstructed_factor = reconstruct(
        mean_core, factor_core, depth_basis
    )
    flattened_original = np.concatenate([mean.ravel(), factor.ravel()])
    flattened_reconstructed = np.concatenate([
        reconstructed_mean.ravel(), reconstructed_factor.ravel()
    ])
    prior_relative = float(np.linalg.norm(
        flattened_reconstructed - flattened_original
    ) / np.linalg.norm(flattened_original))
    prior_max_abs = float(np.max(np.abs(
        flattened_reconstructed - flattened_original
    )))
    with np.load(config["dev_code"]) as data:
        spec = json.loads(str(data["spec"])) if "spec" in data else None
        coordinates = data["coordinates"]
    # The dev speed-code NPZ stores coordinates but its spec is in the original
    # taper-shift context artifact; never read target-frequency fields here.
    if spec is None:
        system = Path(config["dev_code"]).name.split("_")[0]
        shift_code = (
            Path("artifacts/coherent_waveguide_spectral_taper_shift_20261004")
            / f"{system}_codes.npz"
        )
        with np.load(shift_code) as data:
            spec = json.loads(str(data["spec"]))
    cells = 256
    geometry = generator_geometry(spec, cells, 32, modes=32, quadrature_nodes=160)
    original_beta, _ = speed_projection(
        spec, geometry, resample_generator(mean, cells),
        resample_generator(factor, cells), coordinates, derivative=False,
    )
    reconstructed_beta, _ = speed_projection(
        spec, geometry, resample_generator(reconstructed_mean, cells),
        resample_generator(reconstructed_factor, cells), coordinates,
        derivative=False,
    )
    beta_relative = float(np.linalg.norm(reconstructed_beta - original_beta)
                          / np.linalg.norm(original_beta))
    beta_max_abs = float(np.max(np.abs(reconstructed_beta - original_beta)))
    original_values = int(mean.size + factor.size)
    compressed_values = int(mean_core.size + factor_core.size + depth_basis.size)
    q_values = int(q_mean.size + q_factor.size)
    result = {
        "prior_relative_error": prior_relative,
        "prior_max_absolute_error": prior_max_abs,
        "dev_fixed_code_beta_relative_error": beta_relative,
        "dev_fixed_code_beta_max_absolute_error": beta_max_abs,
        "training_prior_depth_singular_values_first20": singular[:20].tolist(),
        "stored_float64_values": {
            "uncompressed_speed": original_values,
            "compressed_speed": compressed_values,
            "uncompressed_q_pca": q_values,
        },
        "compressed_speed_over_q_pca_storage": compressed_values / q_values,
        "compression_factor_vs_uncompressed_speed": original_values / compressed_values,
        "gate_checks": {
            "prior_reconstruction": prior_relative <= config["max_prior_relative_error"],
            "fixed_code_beta": beta_relative <= config["max_beta_relative_error"],
        },
        "scope": config["scope"],
    }
    result["all_gates_pass"] = all(result["gate_checks"].values())
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "frozen_priors_sha256": digest(config["frozen_priors"]),
        "dev_code_sha256": digest(config["dev_code"]),
        "implementation_sha256": digest(__file__),
    }, indent=2))
    np.savez_compressed(
        output / "compressed_speed_prior.npz",
        mean_core=mean_core, factor_core=factor_core, depth_basis=depth_basis,
    )
    result["compressed_prior_sha256"] = digest(output / "compressed_speed_prior.npz")
    (output / "summary.json").write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
