"""Strong classical full-field identification on the multiscale M0 pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .audit_helmholtz_generator_identification import (
    estimate_slowness,
    laplacian,
    query,
)
from .audit_waveguide_spectral_context_robustness import digest
from .coherent_helmholtz_data import sample_environment
from .coherent_representation import complex_metrics


def public_spec(spec: dict) -> dict:
    """Keep source/loss/BC; remove all hidden medium coefficients."""
    public = {key: value for key, value in spec.items()
              if key not in {"spectral_texture", "texture_scale", "family",
                             "centers", "widths", "contrasts"}}
    return {**public, "centers": [], "widths": [], "contrasts": []}


def run(config: dict, output: Path) -> None:
    if (config["systems"], config["context_frequencies_hz"],
        config["query_frequencies_hz"], config["observation_grid"],
        config["teacher_grid"]) != (
            list(range(20269600, 20269608)), [6.0, 10.0, 14.0],
            [11.3, 17.0], 65, 257):
        raise ValueError("frozen classical pilot audit settings changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    manifest = {
        "scope": config["scope"], "implementation_sha256": digest(__file__),
        "pilot_truth_sha256": {},
    }
    fine_axis = np.linspace(0, 1, config["teacher_grid"])
    stride = (config["teacher_grid"] - 1) // (config["observation_grid"] - 1)
    coarse_axis = fine_axis[::stride]
    if len(coarse_axis) != config["observation_grid"]:
        raise ValueError("non-nested context sampling")
    keep = ((fine_axis >= config["roi"][0]) & (fine_axis <= config["roi"][1]))
    lap = laplacian(config["teacher_grid"])
    for system in config["systems"]:
        path = Path(config["source_data"]) / f"system_{system}_grid257.npz"
        manifest["pilot_truth_sha256"][str(path)] = digest(path)
        with np.load(path) as data:
            spec = json.loads(str(data["spec"]))
            truth = data["field"]
            frequencies = data["frequencies"]
        stripped = public_spec(spec)
        stripped["contrasts"] = [0.0] * len(spec["contrasts"])
        if "spectral_texture" in stripped:
            raise ValueError("hidden multiscale coefficients leaked to classical baseline")
        context_idx = [int(np.flatnonzero(frequencies == f)[0])
                       for f in config["context_frequencies_hz"]]
        context = truth[context_idx, ::stride, ::stride].astype(np.complex128)
        started = time.perf_counter()
        q, clipped = estimate_slowness(
            context, np.array(config["context_frequencies_hz"]), stripped,
            coarse_axis, "sine_spectral", 0,
        )
        q_fine = np.clip(RectBivariateSpline(coarse_axis, coarse_axis, q)(
            fine_axis, fine_axis,
        ), 1, 6.25)
        encoding_seconds = time.perf_counter() - started
        _, loss, source = sample_environment(stripped, fine_axis)
        for frequency in config["query_frequencies_hz"]:
            index = int(np.flatnonzero(frequencies == frequency)[0])
            started = time.perf_counter()
            prediction = query(q_fine, source, loss, frequency, lap)
            rows.append({
                "system": system, "frequency": frequency,
                "information": "THREE_FULL_65x65_COMPLEX_LOWBAND_FIELDS",
                "method": config["method"], "medium_hidden": True,
                "clipping_fraction": clipped,
                "encoding_seconds": encoding_seconds,
                "query_seconds": time.perf_counter() - started,
                **complex_metrics(
                    prediction[np.ix_(keep, keep)],
                    truth[index][np.ix_(keep, keep)],
                ),
            })
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for frequency in config["query_frequencies_hz"]:
        sample = [r for r in rows if r["frequency"] == frequency]
        if len(sample) != len(config["systems"]):
            raise ValueError("incomplete classical pilot rows")
        summary.append({
            "frequency": frequency, "systems": len(sample),
            "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
            "mean_phase_rad": float(np.mean([
                r["energy_weighted_phase_rmse_rad"] for r in sample
            ])),
            "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
        })
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "scope": config["scope"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
