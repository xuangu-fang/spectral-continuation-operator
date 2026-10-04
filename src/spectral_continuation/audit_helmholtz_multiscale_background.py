"""Descriptive homogeneous-background contrast for multiscale 2D wave fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_helmholtz_multiscale_fullfield_classic import public_spec
from .audit_waveguide_spectral_context_robustness import digest
from .coherent_helmholtz_data import solve
from .coherent_representation import complex_metrics


def run(config: dict, output: Path) -> None:
    if (config["systems"], config["query_frequencies_hz"], config["grid"]) != (
            list(range(20269600, 20269608)), [11.3, 17.0], 257):
        raise ValueError("fixed background diagnostic changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    axis = np.linspace(0, 1, config["grid"])
    keep = (axis >= config["roi"][0]) & (axis <= config["roi"][1])
    manifest = {"scope": config["scope"], "implementation_sha256": digest(__file__),
                "public_spec_sha256": digest(
                    Path(__file__).with_name("audit_helmholtz_multiscale_fullfield_classic.py")
                ), "truth_sha256": {}}
    rows = []
    for system in config["systems"]:
        path = Path(config["source_data"]) / f"system_{system}_grid257.npz"
        manifest["truth_sha256"][str(path)] = digest(path)
        with np.load(path) as data:
            spec = json.loads(str(data["spec"]))
            truth = data["field"]
            frequencies = data["frequencies"]
        public = public_spec(spec)
        background, _, residuals, _ = solve(
            public, config["grid"], config["query_frequencies_hz"]
        )
        for index, frequency in enumerate(config["query_frequencies_hz"]):
            target_index = int(np.flatnonzero(frequencies == frequency)[0])
            rows.append({
                "system": system, "frequency": frequency,
                "information": "KNOWN_BACKGROUND_SOURCE_PDE_NO_LOWBAND_ADAPTATION",
                "solver_residual": residuals[index],
                **complex_metrics(
                    background[index][np.ix_(keep, keep)],
                    truth[target_index][np.ix_(keep, keep)],
                ),
            })
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for frequency in config["query_frequencies_hz"]:
        sample = [r for r in rows if r["frequency"] == frequency]
        if len(sample) != len(config["systems"]):
            raise ValueError("incomplete background rows")
        summary.append({
            "frequency": frequency, "systems": len(sample),
            "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
            "mean_phase_rad": float(np.mean([
                r["energy_weighted_phase_rmse_rad"] for r in sample
            ])),
            "min_raw": float(np.min([r["complex_rel_l2"] for r in sample])),
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
