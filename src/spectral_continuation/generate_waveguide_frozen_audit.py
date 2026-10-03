"""Freeze candidates before generating independent original/new-source queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .coarse_waveguide_operator import encode_operator, query_field
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import environment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config, output):
    if output.exists():
        raise FileExistsError("freeze audit must use a fresh directory")
    systems = list(
        range(config["system_seed_start"], config["system_seed_start"] + config["systems_count"])
    )
    old = json.loads((Path(config["coefficient_run"]) / "config.json").read_text())
    if set(systems) & set(
        old["train_systems"] + old["validation_systems"] + config["audit_systems"]
    ):
        raise ValueError("independent system overlap")
    frozen = {}
    for run in config["frozen_neural_runs"]:
        for name in ["best.pt", "config.json"]:
            path = Path(run) / name
            frozen[str(path)] = digest(path)
    for name in config["frozen_hyperparameter_files"]:
        frozen[name] = digest(Path(name))
    sources = {str(p): digest(p) for p in sorted(Path(__file__).parent.glob("*waveguide*.py"))}
    output.mkdir(parents=True)
    (output / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "config": config,
                "systems": systems,
                "frozen_files_sha256": frozen,
                "sources_sha256": sources,
                "meaning": "captured before generating new audit systems; no retrospective model freeze",
            },
            indent=2,
        )
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    coordinates = {
        "original": (np.linspace(0, 2000, 2049), np.linspace(0, 100, 129)),
        "narrow": ((np.arange(2049) + 0.37) / 2049 * 2000, np.linspace(0.3, 99.7, 385)),
    }

    def spec_for(system, variant):
        spec = environment(system)
        if variant == "narrow":
            spec["source_width_m"] = config["narrow_source_width_m"]
        return spec

    audits = []
    for system in config["audit_systems"]:
        for variant, frequencies in [("original", [70, 140]), ("narrow", [200, 400])]:
            spec = spec_for(system, variant)
            ranges, depths = coordinates[variant]
            coarse = encode_operator(
                spec,
                config["teacher_cells"],
                config[f"{variant}_modes"],
                config["quadrature_nodes"],
            )
            reference = encode_operator(
                spec,
                config["reference_cells"],
                config[f"{variant}_reference_modes"],
                config["quadrature_nodes"],
            )
            for f in frequencies:
                prediction = query_field(coarse, f, ranges, depths)
                target = query_field(reference, f, ranges, depths)
                audits.append(
                    {
                        "system": system,
                        "variant": variant,
                        "frequency": f,
                        **complex_metrics(prediction[:, ranges >= 50], target[:, ranges >= 50]),
                    }
                )
    (output / "convergence.json").write_text(json.dumps(audits, indent=2))
    maximum = max(r["complex_rel_l2"] for r in audits)
    print(json.dumps({"convergence_max": maximum, "gate": config["convergence_gate"]}), flush=True)
    if maximum > config["convergence_gate"]:
        raise ValueError("teacher convergence gate failed; do not generate candidate audit")
    records = []
    for system in systems:
        for variant in ["original", "narrow"]:
            directory = output / variant
            directory.mkdir(exist_ok=True)
            spec = spec_for(system, variant)
            ranges, depths = coordinates[variant]
            latent = encode_operator(
                spec,
                config["teacher_cells"],
                config[f"{variant}_modes"],
                config["quadrature_nodes"],
            )
            frequencies = config[f"{variant}_frequencies"]
            fields = np.stack([query_field(latent, f, ranges, depths) for f in frequencies])
            filename = (
                f"system{system}_m32_r2049.npz"
                if variant == "original"
                else f"system{system}_narrow.npz"
            )
            path = directory / filename
            np.savez_compressed(
                path,
                field=fields.astype(np.complex64),
                frequencies_hz=frequencies,
                ranges_m=ranges,
                depths_m=depths,
                spec=json.dumps(spec),
                evidence="frozen-model independent IID system audit; not training/development reuse",
                phase_convention="exp(-i omega t), real Gaussian source profile",
            )
            records.append(
                {
                    "system": system,
                    "variant": variant,
                    "path": str(path),
                    "shape": list(fields.shape),
                    "sha256": digest(path),
                }
            )
        (output / "generation.json").write_text(json.dumps(records, indent=2))
        print(json.dumps({"system": system, "complete_files": len(records)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
