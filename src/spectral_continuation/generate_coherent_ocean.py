"""Independent dense coordinate teacher for previously opened Ocean environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ocean_multifidelity_solver import EnvironmentCase, GridSpec, JuliaPekerisRunner, SolverSpec


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    runner = JuliaPekerisRunner()
    solver = SolverSpec("all_modes", "pekeris_mode", 0, 0)
    rows = []
    with np.load(config["source_archive"]) as z:
        raw = z["complex_field"]
        names = list(z["environment_parameter_names"])
        params = z["environment_parameters"]
        for idx in config["system_indices"]:
            if z["split"][idx] not in ("train", "val"):
                raise ValueError("preserve strict reserved systems")
            p = dict(zip(names, params[idx], strict=True))
            case = EnvironmentCase(
                str(z["environment_ids"][idx]),
                p["water_speed_mps"],
                p["water_depth_m"],
                p["seabed_speed_mps"],
                p["seabed_density_kgm3"],
                p["source_depth_m"],
            )
            coarse = GridSpec(50, 750, 384, 1, 48, depth_end_m=78)
            verify = runner.solve(solver, case, 110, coarse)
            old = raw[idx, int(np.flatnonzero(z["frequencies_hz"] == 110)[0])]
            old = old[..., 0] + 1j * old[..., 1]
            error = float(np.linalg.norm(verify.field - old) / np.linalg.norm(old))
            if error > 1e-5:
                raise RuntimeError(f"solver lineage mismatch: {error}")
            nz, nr = config["spatial_grid"]
            fine = GridSpec(50, 750, nr, 1, nz, depth_end_m=78)
            fields = []
            for f in config["frequencies"]:
                result = runner.solve(solver, case, f, fine)
                fields.append(result.field.astype(np.complex64))
                rows.append(
                    {
                        "system": case.case_id,
                        "frequency": f,
                        "seconds": result.elapsed_seconds,
                        "modes": result.component_count,
                        "legacy_110_grid_identity_error": error,
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
            np.savez_compressed(
                output / f"system{idx}_fine.npz",
                field=np.stack(fields),
                x=np.linspace(50, 750, nr),
                y=np.linspace(1, 78, nz),
                frequencies=config["frequencies"],
                environment=json.dumps(p),
                system_id=case.case_id,
                legacy_index=idx,
                evidence="independent point evaluation; development system, not sealed test",
                phase_convention="exp(-i omega t), outgoing exp(+i kr r)",
            )
            (output / "audit.json").write_text(json.dumps(rows, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
