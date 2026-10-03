"""New-source/higher-frequency queries from already frozen inferred codes.

No context refitting or model selection. Completeness of all 64 systems and the
registered inference runs is checked before loading any new query truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audit_waveguide_generator_identification import (
    generator_geometry,
    latent_from_coefficients,
    true_projection,
)
from .coarse_waveguide_operator import query_field
from .coherent_representation import complex_metrics


def completed_systems(rows, systems, expected):
    completed = set()
    for system in {row["system"] for row in rows}:
        if system not in systems:
            raise ValueError("unregistered query system")
        selected = [row for row in rows if row["system"] == system]
        keys = [(row["training_seed"], row["method"], row["frequency"]) for row in selected]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("partial/duplicate frozen query system")
        completed.add(system)
    return completed


def run(config, output, resume=False, max_new_systems=None):
    frozen = json.loads(Path(config["freeze_manifest"]).read_text())
    systems = frozen["systems"]
    for name, digest in frozen["frozen_files_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"changed frozen candidate {name}")
    candidates = []
    for seed, directory in config["inference_runs"].items():
        directory = Path(directory)
        settings = json.loads((directory / "config.json").read_text())
        if settings["systems"] != systems or not (directory / "summary.json").exists():
            raise ValueError("complete original-query inference required first")
        methods = (
            config["shared_methods"] + config["neural_methods"]
            if seed == "42"
            else config["neural_methods"]
        )
        for method in methods:
            paths = [directory / "coefficients" / f"system{s}_{method}.npz" for s in systems]
            if not all(p.exists() for p in paths):
                raise ValueError(f"missing frozen coefficients: {seed}/{method}")
            candidates.append((seed, method, paths))
    output.mkdir(parents=True, exist_ok=True)
    configuration = output / "config.json"
    if configuration.exists():
        if not resume:
            raise ValueError("existing output requires explicit --resume")
        if json.loads(configuration.read_text()) != config:
            raise ValueError("resume configuration changed")
    configuration.write_text(json.dumps(config, indent=2))
    expected = {
        (seed, method, float(frequency))
        for seed, method in [(s, m) for s, m, _ in candidates]
        + [("oracle", "true_medium_projection")]
        for frequency in config["query_frequencies"]
    }
    result_path = output / "results.json"
    rows = json.loads(result_path.read_text()) if resume and result_path.exists() else []
    completed = completed_systems(rows, systems, expected)
    if rows:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.copy2(result_path, output / f"pre_resume_{stamp}.json")
    new_systems = 0
    for si, system in enumerate(systems):
        if system in completed:
            continue
        with np.load(Path(config["data"]) / f"system{system}_narrow.npz") as z:
            spec = json.loads(str(z["spec"]))
            fields, freqs, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
        if freqs.tolist() != config["query_frequencies"]:
            raise ValueError("new-query frequency cohort changed")
        geometry = generator_geometry(spec, 64, 16, modes=64)
        initial = geometry["basis"].T @ (
            geometry["weights"]
            * np.exp(
                -0.5 * ((geometry["z"] - spec["source_depth_m"]) / spec["source_width_m"]) ** 2
            )
        )
        coefficients = []
        for seed, method, paths in candidates:
            with np.load(paths[si]) as z:
                if int(z["system"]) != system:
                    raise ValueError("coefficient identity mismatch")
                coefficients.append((seed, method, z["beta"]))
        coefficients.append(("oracle", "true_medium_projection", true_projection(spec, geometry)))
        for seed, method, beta in coefficients:
            latent, clipping = latent_from_coefficients(beta, initial, geometry)
            for fi, frequency in enumerate(freqs):
                prediction = query_field(latent, frequency, ranges, depths)
                roi = ranges >= 50
                rows.append(
                    {
                        "system": system,
                        "training_seed": seed,
                        "method": method,
                        "frequency": float(frequency),
                        "decoder_modes": 64,
                        "source_width_m": 2.0,
                        "clipping_fraction": clipping,
                        **complex_metrics(prediction[:, roi], fields[fi][:, roi]),
                    }
                )
        temporary = output / "results.json.tmp"
        temporary.write_text(json.dumps(rows, indent=2))
        temporary.replace(result_path)
        completed.add(system)
        new_systems += 1
        print(json.dumps({"system": system, "complete_systems": si + 1}), flush=True)
        if max_new_systems is not None and new_systems >= max_new_systems:
            break
    (output / "execution_progress.json").write_text(
        json.dumps(
            {
                "completed_systems": sorted(completed),
                "total_systems": len(systems),
                "complete": completed == set(systems),
            },
            indent=2,
        )
    )
    if completed != set(systems):
        return
    summary = []
    for seed, method in sorted({(r["training_seed"], r["method"]) for r in rows}):
        for frequency in sorted({r["frequency"] for r in rows}):
            selected = [
                r
                for r in rows
                if r["training_seed"] == seed
                and r["method"] == method
                and r["frequency"] == frequency
            ]
            summary.append(
                {
                    "training_seed": seed,
                    "method": method,
                    "frequency": frequency,
                    "systems": len(selected),
                    "raw": float(np.mean([r["complex_rel_l2"] for r in selected])),
                    "phase_rmse_rad": float(
                        np.mean([r["energy_weighted_phase_rmse_rad"] for r in selected])
                    ),
                }
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-systems", type=int)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.resume, args.max_new_systems)


if __name__ == "__main__":
    main()
