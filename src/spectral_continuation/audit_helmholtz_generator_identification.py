"""Classical real-generator identification from interior complex observations.

Known source/loss/background, NOT known heterogeneous speed. True speed is
used solely in the explicitly labelled projection oracle and diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.fft import dstn, idstn
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import splu

from .coherent_helmholtz_data import sample_environment
from .coherent_representation import complex_metrics


def known_terms(spec, axis):
    # Zero the hidden contrasts before calling the public geometry/source routine.
    _, loss, source = sample_environment(
        {**spec, "contrasts": [0.0] * len(spec["contrasts"])}, axis
    )
    return loss, source


def laplacian(grid):
    h = 1 / (grid - 1)
    n = grid - 2
    d2 = (
        sparse.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="csc")
        / h**2
    )
    d4 = d2 - (h * h / 12) * (d2 @ d2)
    return sparse.kronsum(d4, d4, format="csc")


def estimate_slowness(fields, frequencies, spec, axis, derivative, smoothing):
    loss, source = known_terms(spec, axis)
    u = fields[:, 1:-1, 1:-1]
    if derivative == "fourth_order":
        lap = laplacian(len(axis))
        derivative_value = np.stack([(lap @ field.ravel()).reshape(field.shape) for field in u])
    elif derivative == "sine_spectral":
        eigen = (np.arange(1, len(axis) - 1) * np.pi) ** 2
        coefficients = dstn(u, type=1, axes=(-2, -1), norm="ortho")
        derivative_value = idstn(
            -coefficients * (eigen[:, None] + eigen[None]), type=1, axes=(-2, -1), norm="ortho"
        )
    else:
        raise ValueError(derivative)
    a = (1 + 1j * loss[1:-1, 1:-1]) ** 2 * u
    y = (
        -(derivative_value + source[None, 1:-1, 1:-1])
        / (2 * np.pi * np.asarray(frequencies))[:, None, None] ** 2
    )
    gram = np.sum(abs(a) ** 2, axis=0)
    rhs = np.sum((a.conj() * y).real, axis=0)
    if smoothing:
        gram = gaussian_filter(gram, smoothing, mode="nearest")
        rhs = gaussian_filter(rhs, smoothing, mode="nearest")
    q = rhs / np.maximum(gram, gram.max() * 1e-12)
    clipped = np.clip(q, 1, 6.25)
    result = np.ones((len(axis), len(axis)))
    result[1:-1, 1:-1] = clipped
    return result, float(np.mean(clipped != q))


def query(q, source, loss, frequency, lap):
    coefficient = (2 * np.pi * frequency * (1 + 1j * loss[1:-1, 1:-1])) ** 2 * q[1:-1, 1:-1]
    matrix = lap + sparse.diags(coefficient.ravel(), format="csc")
    solution = splu(matrix).solve(-source[1:-1, 1:-1].ravel().astype(complex))
    field = np.zeros_like(source, dtype=complex)
    field[1:-1, 1:-1] = solution.reshape(coefficient.shape)
    return field


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as z:
            spec = json.loads(str(z["spec"]))
            target, frequencies, fine = z["field"], z["frequencies"], z["axis"]
            true_speed = z["speed"]
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        loss, source = known_terms(spec, fine)
        lap = laplacian(len(fine))
        keep = (fine >= 0.18) & (fine <= 0.82)
        for grid in config["context_grids"]:
            stride = (len(fine) - 1) // (grid - 1)
            axis = fine[::stride]
            if len(axis) != grid:
                raise ValueError("non-nested observations")
            fields = target[observed, ::stride, ::stride]
            candidates = [
                (
                    "true_medium_projection_oracle",
                    0,
                    1 / true_speed[::stride, ::stride] ** 2,
                    0.0,
                    0.0,
                )
            ]
            for derivative in config["derivatives"]:
                for smoothing in config["smoothing"]:
                    start = time.perf_counter()
                    q, clipping = estimate_slowness(
                        fields, frequencies[observed], spec, axis, derivative, smoothing
                    )
                    candidates.append(
                        (derivative, smoothing, q, time.perf_counter() - start, clipping)
                    )
            for method, smoothing, q, seconds, clipping in candidates:
                reconstructed = np.clip(RectBivariateSpline(axis, axis, q)(fine, fine), 1, 6.25)
                for frequency in config["query_frequencies"]:
                    fi = int(np.flatnonzero(frequencies == frequency)[0])
                    start = time.perf_counter()
                    prediction = query(reconstructed, source, loss, frequency, lap)
                    rows.append(
                        {
                            "system": system,
                            "context_grid": grid,
                            "method": method,
                            "smoothing_grid_units": smoothing,
                            "frequency": frequency,
                            "encoding_seconds": seconds,
                            "query_seconds": time.perf_counter() - start,
                            "clipping_fraction": clipping,
                            "speed_rmse_diagnostic_only": float(
                                np.sqrt(np.mean((1 / np.sqrt(reconstructed) - true_speed) ** 2))
                            ),
                            **complex_metrics(
                                prediction[np.ix_(keep, keep)], target[fi][np.ix_(keep, keep)]
                            ),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "system": system,
                            "grid": grid,
                            "method": method,
                            "smoothing": smoothing,
                            "raw": float(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
