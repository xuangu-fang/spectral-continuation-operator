"""Reusable complex 2D Helmholtz teacher with continuous environment definitions.

Fourth-order Dirichlet Laplacian from the sine-compatible second-order matrix.
Gaussian sources/inclusions avoid source-cell rounding between grids. A lossy
boundary layer is a sponge, NOT a perfectly matched layer. Continuum claims
require the included nested-grid convergence audit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .coherent_representation import complex_metrics


def environment(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "seed": seed,
        "centers": rng.uniform(0.25, 0.75, (4, 2)).tolist(),
        "widths": rng.uniform(0.05, 0.12, 4).tolist(),
        "contrasts": rng.uniform(0.05, 0.20, 4).tolist(),
        "source": rng.uniform([0.18, 0.25], [0.25, 0.75]).tolist(),
        "source_width": 0.025,
        "sponge_width": 0.16,
        "sponge_strength": 2.0,
        "background_loss": 0.002,
    }


def sample_environment(spec: dict, axis: np.ndarray):
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    speed = np.ones_like(xx)
    for c, w, contrast in zip(spec["centers"], spec["widths"], spec["contrasts"], strict=True):
        speed -= contrast * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * w * w))
    if speed.min() < 0.4:
        raise ValueError("unsafe wavespeed in generated environment")
    edge = np.minimum.reduce([xx, yy, 1 - xx, 1 - yy])
    loss = (
        spec["background_loss"]
        + spec["sponge_strength"] * np.maximum(1 - edge / spec["sponge_width"], 0) ** 3
    )
    sx, sy = spec["source"]
    src = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * spec["source_width"] ** 2))
    return speed, loss, src


def solve(spec: dict, grid: int, frequencies: list[float]):
    axis = np.linspace(0, 1, grid)
    speed, loss, source = sample_environment(spec, axis)
    h, n = 1 / (grid - 1), grid - 2
    d2 = (
        sparse.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="csc")
        / h**2
    )
    d4 = d2 - (h * h / 12) * (d2 @ d2)
    lap = sparse.kronsum(d4, d4, format="csc")
    fields, residuals, timings = [], [], []
    rhs = -source[1:-1, 1:-1].ravel()
    for freq in frequencies:
        start = time.perf_counter()
        k = 2 * np.pi * freq / speed[1:-1, 1:-1] * (1 + 1j * loss[1:-1, 1:-1])
        matrix = lap + sparse.diags(k.ravel() ** 2, format="csc")
        result = splu(matrix).solve(rhs.astype(np.complex128))
        residuals.append(float(np.linalg.norm(matrix @ result - rhs) / np.linalg.norm(rhs)))
        timings.append(time.perf_counter() - start)
        field = np.zeros((grid, grid), dtype=np.complex128)
        field[1:-1, 1:-1] = result.reshape(n, n)
        fields.append(field)
    return np.stack(fields), speed, residuals, timings


def run(config: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    seeds = config.get("seeds")
    if seeds is None:
        seeds = list(range(config["seed_start"], config["seed_start"] + config["system_count"]))
    for seed in seeds:
        spec = environment(seed)
        nested = {}
        for grid in config["grids"]:
            fields, speed, residuals, seconds = solve(spec, grid, config["frequencies"])
            nested[grid] = fields
            np.savez_compressed(
                output / f"system_{seed}_grid{grid}.npz",
                field=fields.astype(np.complex64),
                speed=speed,
                axis=np.linspace(0, 1, grid),
                frequencies=config["frequencies"],
                spec=json.dumps(spec),
                relative_residual=residuals,
                phase_convention="exp(-i omega t); positive imaginary k gives decay",
                evidence="development; direct solver coordinates, no interpolation truth",
            )
            row = {
                "seed": seed,
                "grid": grid,
                "residual_max": max(residuals),
                "solve_seconds": sum(seconds),
                "minimum_speed": float(speed.min()),
                "min_points_per_wavelength": float(
                    speed.min() / max(config["frequencies"]) * (grid - 1)
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        if len(config["grids"]) > 1:
            coarse, fine = config["grids"][-2:]
            ratio = (fine - 1) // (coarse - 1)
            if ratio * (coarse - 1) != fine - 1:
                raise ValueError("convergence grids must nest exactly")
            axis = np.linspace(0, 1, coarse)
            keep = (axis >= 0.18) & (axis <= 0.82)
            for i, freq in enumerate(config["frequencies"]):
                a = nested[coarse][i][np.ix_(keep, keep)]
                b = nested[fine][i, ::ratio, ::ratio][np.ix_(keep, keep)]
                row = {
                    "seed": seed,
                    "coarse": coarse,
                    "fine": fine,
                    "frequency": freq,
                    "scope": "interior convergence; not a learning result",
                    **complex_metrics(a, b),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        (output / "audit.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
