"""Numerically audited two-way 2D Helmholtz multiscale-medium M0 pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .audit_waveguide_spectral_context_robustness import digest
from .coherent_helmholtz_data import sample_environment, solve
from .coherent_representation import complex_metrics


def environment(seed: int, config: dict) -> dict:
    rng = np.random.default_rng(seed)
    modes = config["spectral_modes_per_axis"]
    index = np.arange(1, modes + 1)
    radius2 = (index[:, None] / 3) ** 2 + (index[None] / 3) ** 2
    coefficients = rng.normal(size=(modes, modes)) / (1 + radius2) ** 1.25
    diagnostic_axis = np.linspace(0, 1, 257)
    cosines = np.cos(np.pi * np.outer(diagnostic_axis, index))
    texture = cosines @ coefficients.T @ cosines.T
    scale = min(
        config["texture_target_rms"] / float(np.sqrt(np.mean(texture**2))),
        config["texture_max_abs"] / float(np.max(np.abs(texture))),
    )
    return {
        "seed": seed,
        "centers": rng.uniform(0.20, 0.80, (3, 2)).tolist(),
        "widths": rng.uniform(0.035, 0.09, 3).tolist(),
        "contrasts": rng.uniform(0.055, 0.095, 3).tolist(),
        "spectral_texture": coefficients.tolist(),
        "texture_scale": float(scale),
        "source": rng.uniform([0.17, 0.20], [0.27, 0.80]).tolist(),
        "source_width": 0.025,
        "sponge_width": 0.16,
        "sponge_strength": 2.0,
        "background_loss": 0.002,
        "family": "multiscale_cosine_plus_three_inclusions",
    }


def run(config: dict, output: Path, data: Path) -> None:
    if (
        config["train_seed_start"], config["train_count"], config["pilot_systems"],
        config["audit_systems"], config["teacher_grid"], config["audit_grid"],
        config["frequencies_hz"], config["audit_frequencies_hz"],
    ) != (
        20269000, 512, list(range(20269600, 20269608)),
        [20269700, 20269701], 257, 513,
        [6.0, 10.0, 14.0, 11.3, 17.0], [14.0, 17.0],
    ):
        raise ValueError("frozen multiscale M0 pilot settings changed")
    output.mkdir(parents=True, exist_ok=False)
    data.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (data / "config.json").write_text(json.dumps(config, indent=2))
    manifest = {
        "scope": config["scope"], "implementation_sha256": digest(__file__),
        "old_solver_sha256": digest(Path(__file__).with_name("coherent_helmholtz_data.py")),
        "train_seeds": list(range(config["train_seed_start"],
                                  config["train_seed_start"] + config["train_count"])),
        "pilot_systems": config["pilot_systems"],
        "audit_systems": config["audit_systems"],
        "generated_files_sha256": {},
    }
    if set(manifest["train_seeds"]) & set(config["pilot_systems"] + config["audit_systems"]):
        raise ValueError("training and pilot/audit systems overlap")
    label_axis = np.linspace(0, 1, config["medium_label_grid"])
    labels = []
    for seed in manifest["train_seeds"]:
        spec = environment(seed, config)
        speed, _, _ = sample_environment(spec, label_axis)
        labels.append(speed.astype(np.float32))
    training_path = data / "training_medium_labels.npz"
    np.savez_compressed(
        training_path, systems=np.array(manifest["train_seeds"]),
        speed=np.stack(labels), axis=label_axis,
        evidence="medium labels only; no training wave responses in this M0 pilot",
    )
    manifest["generated_files_sha256"][str(training_path)] = digest(training_path)
    records = []
    for seed in config["pilot_systems"]:
        spec = environment(seed, config)
        started = time.perf_counter()
        fields, speed, residuals, seconds = solve(
            spec, config["teacher_grid"], config["frequencies_hz"]
        )
        path = data / f"system_{seed}_grid257.npz"
        np.savez_compressed(
            path, field=fields.astype(np.complex64), speed=speed.astype(np.float32),
            axis=np.linspace(0, 1, config["teacher_grid"]),
            frequencies=np.array(config["frequencies_hz"]), spec=json.dumps(spec),
            relative_residual=np.array(residuals),
            evidence="M0 development pilot; not independent paper test",
        )
        manifest["generated_files_sha256"][str(path)] = digest(path)
        row = {"seed": seed, "kind": "pilot", "max_solver_residual": max(residuals),
               "min_speed": float(speed.min()),
               "min_points_per_wavelength": float(speed.min()
                   * (config["teacher_grid"] - 1) / max(config["frequencies_hz"])),
               "solve_seconds": float(sum(seconds)),
               "wall_seconds": time.perf_counter() - started}
        records.append(row)
        (output / "progress.json").write_text(json.dumps(records, indent=2))
        print(json.dumps(row), flush=True)
    for seed in config["audit_systems"]:
        spec = environment(seed, config)
        coarse, speed, coarse_residuals, _ = solve(
            spec, config["teacher_grid"], config["audit_frequencies_hz"]
        )
        fine, _, fine_residuals, _ = solve(
            spec, config["audit_grid"], config["audit_frequencies_hz"]
        )
        axis = np.linspace(0, 1, config["teacher_grid"])
        keep = (axis >= config["numeric_roi"][0]) & (axis <= config["numeric_roi"][1])
        stride = (config["audit_grid"] - 1) // (config["teacher_grid"] - 1)
        if stride * (config["teacher_grid"] - 1) != config["audit_grid"] - 1:
            raise ValueError("numerical audit grids must nest exactly")
        for index, frequency in enumerate(config["audit_frequencies_hz"]):
            metrics = complex_metrics(
                coarse[index][np.ix_(keep, keep)],
                fine[index, ::stride, ::stride][np.ix_(keep, keep)],
            )
            records.append({
                "seed": seed, "kind": "numerical_audit", "frequency": frequency,
                "max_solver_residual": max(coarse_residuals + fine_residuals),
                "min_speed": float(speed.min()), **metrics,
            })
            (output / "progress.json").write_text(json.dumps(records, indent=2))
            print(json.dumps(records[-1]), flush=True)
    numeric = [r for r in records if r["kind"] == "numerical_audit"]
    pilots = [r for r in records if r["kind"] == "pilot"]
    checks = {
        "numerical_raw": all(r["complex_rel_l2"] < config["numerical_raw_max_below"]
                             for r in numeric),
        "resolution": all(r["min_points_per_wavelength"]
                          >= config["min_teacher_points_per_wavelength_at_least"]
                          for r in pilots),
        "solver_residual": all(r["max_solver_residual"] < 1e-8 for r in records),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "results.json").write_text(json.dumps(records, indent=2))
    (output / "summary.json").write_text(json.dumps({
        "checks": checks, "all_m0_pilot_checks_pass": all(checks.values()),
        "max_numerical_raw": max(r["complex_rel_l2"] for r in numeric),
        "min_teacher_points_per_wavelength": min(
            r["min_points_per_wavelength"] for r in pilots
        ),
        "training_medium_labels": len(labels),
        "pilot_systems": len(pilots), "audit_pairs": len(numeric),
        "scope": config["scope"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.data)


if __name__ == "__main__":
    main()
