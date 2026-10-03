"""Paired system statistics; training seeds/frequencies are not extra systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_summary(model, baseline, repetitions=10000, seed=202623):
    model, baseline = np.asarray(model, float), np.asarray(baseline, float)
    if model.shape != baseline.shape or model.ndim != 1 or len(model) < 2:
        raise ValueError("paired one-dimensional system errors required")
    if not (np.isfinite(model).all() and np.isfinite(baseline).all()) or np.any(baseline <= 0):
        raise ValueError("finite errors and strictly positive baseline required")
    indices = np.random.default_rng(seed).integers(0, len(model), (repetitions, len(model)))
    improvements = 1 - model[indices].mean(1) / baseline[indices].mean(1)
    return {
        "systems": len(model),
        "model_raw": float(model.mean()),
        "baseline_raw": float(baseline.mean()),
        "relative_improvement": float(1 - model.mean() / baseline.mean()),
        "paired_bootstrap_95_percent": np.quantile(improvements, [0.025, 0.975]).tolist(),
        "system_win_rate": float(np.mean(model < baseline)),
        "model_p90": float(np.quantile(model, 0.9)),
        "model_p95": float(np.quantile(model, 0.95)),
    }


def run(config, output):
    frozen = json.loads(Path(config["freeze_manifest"]).read_text())
    systems = frozen["systems"]
    pools = {}
    for seed, directory in config["inference_runs"].items():
        directory = Path(directory)
        if not (directory / "summary.json").exists():
            raise ValueError("all seed runs must finish before summary")
        settings = json.loads((directory / "config.json").read_text())
        if settings["systems"] != systems:
            raise ValueError("system ordering changed")
        pools[seed] = json.loads((directory / "results.json").read_text())

    def values(seed, method, key="raw"):
        result = []
        for system in systems:
            rows = [r for r in pools[seed] if r["system"] == system and r["method"] == method]
            if not rows:
                raise ValueError(f"missing {seed}/{method}/{system}")
            result.append(float(np.median([r[key] for r in rows])))
        return np.array(result)

    result = []
    for comparison in config["primary_comparisons"]:
        method, baseline = comparison["model"], comparison["baseline"]
        errors = np.array([values(seed, method) for seed in pools])
        reference = values("42", baseline)
        aggregate = paired_summary(
            errors.mean(0), reference, config["bootstrap_repetitions"], config["bootstrap_seed"]
        )
        aggregate.update(
            {
                **comparison,
                "seed_raw_means": {s: float(values(s, method).mean()) for s in pools},
                "seed_average_is_not_prediction_ensemble": True,
                "evidence": "independent frozen IID systems; one-way synthetic domain only",
            }
        )
        result.append(aggregate)
    tables = []
    for seed, seed_rows in pools.items():
        for method in sorted({r["method"] for r in seed_rows}):
            tables.append(
                {
                    "seed": seed,
                    "method": method,
                    **{
                        key: float(values(seed, method, key).mean())
                        for key in [
                            "raw",
                            "phase_rmse_rad",
                            "encoding_seconds",
                            "six_queries_seconds",
                            "total_seconds",
                        ]
                    },
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "primary_comparisons.json").write_text(json.dumps(result, indent=2))
    (output / "all_methods.json").write_text(json.dumps(tables, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
