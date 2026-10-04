"""Fixed-code continuous-frequency stress on confirmed richer-medium systems."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, spectral_environment


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(config, output):
    source = json.loads(Path(config["source_config"]).read_text())
    complete = json.loads(Path(config["source_results"]).read_text())
    if len(complete) != 20 * len(source["systems"]):
        raise ValueError("independent confirmation source cohort is incomplete")
    if not set(config["systems"]) <= set(source["systems"]):
        raise ValueError("query system outside frozen confirmation cohort")
    if set(config["query_frequencies_hz"]) & set(source["query_frequencies"]):
        raise ValueError("additional frequency sweep must not duplicate confirmation targets")
    for system in config["systems"]:
        if not (Path(config["rank256_codes"]) /
                f"system{system}_rank256_code.npz").exists():
            raise ValueError("rank256 confirmation code missing; do not run before gate check")
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(config["output_data"])
    data_root.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (data_root / "config.json").write_text(json.dumps(config, indent=2))
    tracked = [Path(__file__), Path(config["source_config"]),
               Path(config["source_results"])]
    for system in config["systems"]:
        tracked.append(Path(config["rank256_codes"]) /
                       f"system{system}_rank256_code.npz")
    (output / "supplemental_manifest.json").write_text(json.dumps({
        "registration": config["scope"],
        "tracked_sha256": {str(path): digest(path) for path in tracked},
    }, indent=2))
    convergence = []
    for system in config["audit_systems"]:
        spec = spectral_environment(
            system, source["spectral_terms"], source["perturbation_rms_m_s"]
        )
        ranges = (np.arange(513) + 0.37) / 513 * spec["range_m"]
        roi = ranges >= config["roi_min_range_m"]
        for frequency in config["audit_frequencies_hz"]:
            low = propagate_code(spec, frequency, ranges, config["teacher_cells"],
                                 config["teacher_modes"], config["teacher_quadrature_nodes"],
                                 config["eigh_device"], config["eigh_batch"])
            high = propagate_code(spec, frequency, ranges, config["audit_cells"],
                                  config["audit_modes"], config["audit_quadrature_nodes"],
                                  config["eigh_device"], config["eigh_batch"])
            raw = modal_raw(low[roi], high[roi])
            convergence.append({"system": system, "frequency": frequency,
                                "raw_by_source": raw.tolist()})
            (data_root / "convergence.json").write_text(json.dumps(convergence, indent=2))
            print(json.dumps({"audit": convergence[-1]}), flush=True)
            if np.max(raw) >= config["audit_gate"]:
                raise ValueError("frequency-curve numerical audit failed")
    rows, generation = [], []
    for system in config["systems"]:
        with np.load(Path(config["rank128_codes"]) / f"system{system}_codes.npz") as data:
            spec = json.loads(str(data["spec"]))
            beta128 = data["beta32"]
        with np.load(Path(config["rank256_codes"]) /
                     f"system{system}_rank256_code.npz") as data:
            beta256 = data["beta"]
        ranges = ((np.arange(config["query_range_count"]) + 0.37)
                  / config["query_range_count"] * spec["range_m"])
        depths = np.linspace(0.1, spec["depth_m"] - 0.1, config["query_depth_count"])
        roi = ranges >= config["roi_min_range_m"]
        for frequency in config["query_frequencies_hz"]:
            truth = propagate_code(
                spec, frequency, ranges, config["teacher_cells"],
                config["teacher_modes"], config["teacher_quadrature_nodes"],
                config["eigh_device"], config["eigh_batch"],
            )
            path = data_root / f"system{system}_f{frequency}.npz"
            np.savez_compressed(
                path, modal_coefficients=truth.astype(np.complex64),
                ranges_m=ranges, depths_m=depths, frequency_hz=frequency,
                source_variants=source["source_variants"], spec=json.dumps(spec),
                phase_convention="exp(-i omega t), real Gaussian source, positive forward phase",
                evidence=config["scope"],
            )
            generation.append({"system": system, "frequency": frequency,
                               "path": str(path), "sha256": digest(path)})
            target = (
                depth_basis(depths, spec["depth_m"], truth.shape[1])
                @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)
            ).reshape(len(depths), len(ranges), len(source["source_variants"]))
            for name, beta in [("original_rank128", beta128), ("context_rank256", beta256)]:
                prediction = propagate_code(
                    spec, frequency, ranges, config["decoder_cells"],
                    config["decoder_modes"], config["teacher_quadrature_nodes"],
                    config["eigh_device"], config["eigh_batch"], beta,
                )
                modal = modal_raw(prediction[roi], truth[roi])
                field = (
                    depth_basis(depths, spec["depth_m"], prediction.shape[1])
                    @ prediction.transpose(1, 0, 2).reshape(prediction.shape[1], -1)
                ).reshape(len(depths), len(ranges), len(source["source_variants"]))
                for source_index, variant in enumerate(source["source_variants"]):
                    rows.append({
                        "system": system, "method": name, "frequency": frequency,
                        "source": variant, "modal_parseval_raw": float(modal[source_index]),
                        **complex_metrics(field[:, roi, source_index], target[:, roi, source_index]),
                    })
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (data_root / "generation.json").write_text(json.dumps(generation, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for name in ["original_rank128", "context_rank256"]:
        for frequency in config["query_frequencies_hz"]:
            for variant in source["source_variants"]:
                sample = [r for r in rows if r["method"] == name and
                          r["frequency"] == frequency and r["source"] == variant]
                summary.append({
                    "method": name, "frequency": frequency, "source": variant,
                    "systems": len(sample),
                    "complex_rel_l2": float(np.mean([r["complex_rel_l2"] for r in sample])),
                    "phase_rmse_rad": float(np.mean([
                        r["energy_weighted_phase_rmse_rad"] for r in sample])),
                    "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                })
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
