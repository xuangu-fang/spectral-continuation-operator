"""Matched irregular-coordinate interpolation controls for learned charts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from scipy.special import j0

from .coherent_representation import complex_metrics


def kernel_predict(x, y, query, config):
    if "carrier" in config or "metric" in config:
        carrier = np.asarray(config.get("carrier", [0, 0]))
        metric = np.asarray(config.get("metric", [1, 1]))
        base = {k: v for k, v in config.items() if k not in ("carrier", "metric")}
        p = kernel_predict(x * metric, y * np.exp(-1j * (x @ carrier)), query * metric, base)
        return p * np.exp(1j * (query @ carrier))
    if config["kind"] == "plane_wave":
        angle = (
            np.linspace(-config["cone"], config["cone"], config["branches"], endpoint=False)
            * np.pi
            / 180
        )
        k = 2 * np.pi * config["extent"] * np.stack((np.cos(angle), np.sin(angle)), -1)

        def basis(positions):
            waves = np.exp(1j * positions @ k.T)
            if config["degree"]:
                poly = np.c_[np.ones(len(positions)), positions]
                waves = (waves[..., None] * poly[:, None]).reshape(len(positions), -1)
            return waves

        phi = basis(x)
        weights = np.prod(np.maximum(1 - abs(x), 0.05), -1)
        gram = phi.conj().T @ (phi * weights[:, None])
        penalty = config["ridge"] * np.trace(gram).real / len(gram)
        coeff = np.linalg.solve(gram + penalty * np.eye(len(gram)), phi.conj().T @ (weights * y))
        return basis(query) @ coeff
    if config["kind"] == "rbf":
        values = np.stack((y.real, y.imag), -1)
        model = RBFInterpolator(
            x,
            values,
            kernel=config["kernel"],
            epsilon=config.get("epsilon", 1),
            smoothing=config.get("ridge", 0),
            degree=config.get("degree", -1),
        )
        p = model(query)
        return p[:, 0] + 1j * p[:, 1]
    if config["kind"] == "wave_kernel":
        # Isotropic positive spectral measure: physically matched angular kernel.
        k = np.asarray(config["radii"]) * 2 * np.pi * config["extent"]

        def covariance(a, b):
            distance = cdist(a, b)
            return j0(distance[..., None] * k).mean(-1)

        gram = covariance(x, x)
        coeff = np.linalg.solve(gram + config["ridge"] * np.eye(len(x)), y)
        return covariance(query, x) @ coeff
    raise ValueError(config["kind"])


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    if config.get("pool_config"):
        from .learned_wave_chart import make_ocean_pool

        source_config = json.loads(Path(config["pool_config"]).read_text())["data"]
        arrays = make_ocean_pool(
            {**source_config, "frequencies": source_config["validation_frequencies"]},
            source_config["validation_ids"],
            source_config["validation_patches_per_system"],
            970031,
        )
        np.savez_compressed(output / "development_query_pool.npz", **arrays)
    else:
        with np.load(config["pool"]) as z:
            arrays = dict(z)
    rows, summary = [], []
    for method in config["methods"]:
        start = time.perf_counter()
        predictions = []
        for i, (x, y, q) in enumerate(zip(arrays["x"], arrays["y"], arrays["q"], strict=True)):
            gain_x, gain_q = np.ones(len(x)), np.ones(len(q))
            if method.get("spreading"):
                cx, radius = arrays["center"][i, 0], arrays["physical_radius"][i]
                gain_x = np.sqrt((cx + radius * x[:, 0]) / cx)
                gain_q = np.sqrt((cx + radius * q[:, 0]) / cx)
            predictions.append(kernel_predict(x, y * gain_x, q, method) / gain_q)
        p = np.stack(predictions)
        current = []
        for system in np.unique(arrays["system"]):
            for freq in np.unique(arrays["frequency"]):
                mask = (arrays["system"] == system) & (arrays["frequency"] == freq)
                if not mask.any():
                    continue
                scale = arrays["scale"][mask, None]
                row = {
                    "method": method["name"],
                    "system": int(system),
                    "frequency": float(freq),
                    **complex_metrics(p[mask] * scale, arrays["target"][mask] * scale),
                }
                current.append(row)
                rows.append(row)
        entry = {
            "method": method["name"],
            "macro_rel_l2": float(np.mean([r["complex_rel_l2"] for r in current])),
            "seconds": time.perf_counter() - start,
        }
        print(json.dumps(entry), flush=True)
        summary.append(entry)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        (output / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
