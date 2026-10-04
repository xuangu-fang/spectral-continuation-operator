"""Posthoc full-field observation-grid audit for 2D multiscale Helmholtz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .audit_helmholtz_generator_identification import estimate_slowness, laplacian, query
from .audit_helmholtz_multiscale_fullfield_classic import public_spec
from .audit_waveguide_spectral_context_robustness import digest
from .coherent_helmholtz_data import sample_environment
from .coherent_representation import complex_metrics


def run(config: dict, output: Path) -> None:
    if (config["systems"], config["observation_grids"],
        config["context_frequencies_hz"], config["query_frequencies_hz"],
        config["teacher_grid"]) != (
            list(range(20269600, 20269608)), [65, 129, 257],
            [6.0, 10.0, 14.0], [11.3, 17.0], 257):
        raise ValueError("frozen multiscale sampling audit settings changed")
    previous = json.loads(Path(config["source_65_results"]).read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    fine_axis = np.linspace(0, 1, config["teacher_grid"])
    keep = (fine_axis >= config["roi"][0]) & (fine_axis <= config["roi"][1])
    lap = laplacian(config["teacher_grid"])
    rows = []
    manifest = {
        "scope": config["scope"], "implementation_sha256": digest(__file__),
        "source_65_results_sha256": digest(config["source_65_results"]),
        "truth_sha256": {},
    }
    for system in config["systems"]:
        path = Path(config["source_data"]) / f"system_{system}_grid257.npz"
        manifest["truth_sha256"][str(path)] = digest(path)
        with np.load(path) as data:
            spec = json.loads(str(data["spec"]))
            fields = data["field"]
            frequencies = data["frequencies"]
        stripped = public_spec(spec)
        stripped["contrasts"] = [0.0] * len(spec["contrasts"])
        if "spectral_texture" in stripped:
            raise ValueError("hidden texture leaked")
        _, loss, source = sample_environment(stripped, fine_axis)
        context_idx = [int(np.flatnonzero(frequencies == f)[0])
                       for f in config["context_frequencies_hz"]]
        for grid in config["observation_grids"]:
            stride = (config["teacher_grid"] - 1) // (grid - 1)
            axis = fine_axis[::stride]
            if len(axis) != grid:
                raise ValueError("non-nested observation grid")
            observed = fields[context_idx, ::stride, ::stride].astype(np.complex128)
            q, clipped = estimate_slowness(
                observed, np.array(config["context_frequencies_hz"]), stripped,
                axis, "sine_spectral", 0,
            )
            q_fine = np.clip(RectBivariateSpline(axis, axis, q)(fine_axis, fine_axis), 1, 6.25)
            for frequency in config["query_frequencies_hz"]:
                index = int(np.flatnonzero(frequencies == frequency)[0])
                prediction = query(q_fine, source, loss, frequency, lap)
                case = {
                    "system": system, "observation_grid": grid,
                    "frequency": frequency,
                    "clipping_fraction": clipped,
                    "information": f"THREE_FULL_{grid}x{grid}_COMPLEX_LOWBAND_FIELDS",
                    **complex_metrics(
                        prediction[np.ix_(keep, keep)],
                        fields[index][np.ix_(keep, keep)],
                    ),
                }
                if grid == 65:
                    old = [r for r in previous if r["system"] == system
                           and r["frequency"] == frequency]
                    if len(old) != 1 or abs(
                        old[0]["complex_rel_l2"] - case["complex_rel_l2"]
                    ) > config["max_grid65_reproduction_raw_difference"]:
                        raise ValueError("65-grid strong baseline did not reproduce")
                rows.append(case)
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for grid in config["observation_grids"]:
        for frequency in config["query_frequencies_hz"]:
            sample = [r for r in rows if r["observation_grid"] == grid
                      and r["frequency"] == frequency]
            if len(sample) != len(config["systems"]):
                raise ValueError("incomplete spatial audit")
            summary.append({
                "observation_grid": grid, "frequency": frequency,
                "systems": len(sample),
                "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                "mean_phase_rad": float(np.mean([
                    r["energy_weighted_phase_rmse_rad"] for r in sample
                ])),
                "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
            })
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "grid65_reproduction_passed": True,
        "scope": config["scope"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
