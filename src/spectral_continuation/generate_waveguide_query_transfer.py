"""Direct new-coordinate/source/frequency teacher for frozen generator audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .coarse_waveguide_operator import encode_operator, query_field
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import environment


def source_variant(spec, variant, config):
    result = dict(spec)
    if variant == "shifted":
        result["source_depth_m"] = float(
            np.clip(
                spec["source_depth_m"] + config["source_depth_shift_m"],
                *config["source_depth_bounds_m"],
            )
        )
    elif variant == "narrow":
        result["source_width_m"] = config["narrow_source_width_m"]
    elif variant != "original":
        raise ValueError(variant)
    return result


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    ranges = (np.arange(config["range_points"]) + 0.37) / config["range_points"] * 2000
    depths = np.linspace(0.3, 99.7, config["depth_points"])
    audits = []
    for system in config["audit_systems"]:
        spec = environment(system)
        for variant in config["source_variants"]:
            changed = source_variant(spec, variant, config)
            coarse = encode_operator(
                changed, config["cells"], config["modes"], config.get("quadrature_nodes", 160)
            )
            fine = encode_operator(
                changed,
                config["reference_cells"],
                config["reference_modes"],
                config.get("quadrature_nodes", 160),
            )
            for frequency in config["audit_frequencies"]:
                prediction = query_field(coarse, frequency, ranges, depths)
                target = query_field(fine, frequency, ranges, depths)
                audits.append(
                    {
                        "system": system,
                        "variant": variant,
                        "frequency": frequency,
                        **complex_metrics(prediction[:, ranges >= 50], target[:, ranges >= 50]),
                    }
                )
    (output / "convergence.json").write_text(json.dumps(audits, indent=2))
    maximum = max(row["complex_rel_l2"] for row in audits)
    print(
        json.dumps({"max_convergence_error": maximum, "gate": config["convergence_gate"]}),
        flush=True,
    )
    if maximum > config["convergence_gate"]:
        raise ValueError("numerical gate failed; do not generate transfer truth")
    records = []
    for system in config["systems"]:
        for variant in config["source_variants"]:
            path = output / f"system{system}_{variant}.npz"
            if path.exists():
                raise FileExistsError(path)
            original = environment(system)
            spec = source_variant(original, variant, config)
            latent = encode_operator(
                spec, config["cells"], config["modes"], config.get("quadrature_nodes", 160)
            )
            fields = np.stack(
                [query_field(latent, f, ranges, depths) for f in config["frequencies"]]
            )
            np.savez_compressed(
                path,
                field=fields.astype(np.complex64),
                frequencies_hz=config["frequencies"],
                ranges_m=ranges,
                depths_m=depths,
                spec=json.dumps(spec),
                original_spec=json.dumps(original),
                source_variant=variant,
                initial_coefficients=latent["initial"],
                evidence="paired development direct teacher; not full Helmholtz or formal unseen-system test",
                phase_convention="exp(-i omega t), real source initial profile",
            )
            records.append(
                {
                    "system": system,
                    "variant": variant,
                    "path": str(path),
                    "shape": list(fields.shape),
                    "bytes": path.stat().st_size,
                }
            )
            (output / "generation.json").write_text(json.dumps(records, indent=2))
            print(json.dumps(records[-1]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
