"""Fixed environment code, higher-frequency and adaptive physical readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import generator_geometry
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_taper_shift import metric_rows, shifted_environment
from .learned_speed_manifold import speed_projection


def numerical_audit(config: dict, source: dict) -> list[dict]:
    rows = []
    for system in config["audit_systems"]:
        spec = shifted_environment(
            system, config["terms"], config["perturbation_rms_m_s"],
            config["taper_scale"],
        )
        ranges = (np.arange(513) + 0.37) / 513 * spec["range_m"]
        roi = ranges >= source["roi_min_range_m"]
        for frequency in config["frequencies_hz"]:
            teacher = propagate_code(
                spec, frequency, ranges, config["teacher_cells"],
                config["teacher_modes"], config["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"],
            )
            finer = propagate_code(
                spec, frequency, ranges, config["audit_cells"],
                config["audit_modes"], config["audit_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"],
            )
            raw = modal_raw(teacher[roi], finer[roi])
            rows.append({"system": system, "frequency": frequency,
                         "raw_by_source": raw.tolist()})
            print(json.dumps({"numerical_audit": rows[-1]}), flush=True)
    return rows


def summarize(rows: list[dict], audits: list[dict], config: dict) -> dict:
    summary = []
    for variant in config["decoder_variants"]:
        for frequency in config["frequencies_hz"]:
            for source in ["original", "narrow2m"]:
                sample = [r for r in rows if r["method"] == variant["name"]
                          and r["frequency"] == frequency and r["source"] == source]
                if len(sample) != len(config["systems"]):
                    raise ValueError("missing fixed-code resolution rows")
                summary.append({
                    "method": variant["name"], "frequency": frequency,
                    "source": source, "systems": len(sample),
                    "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                    "mean_phase_rad": float(np.mean([
                        r["energy_weighted_phase_rmse_rad"] for r in sample
                    ])),
                    "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                })
    adaptive = [r for r in summary if r["method"] == "adaptive_512x192"
                and r["frequency"] == 1200]
    comparison = []
    for row in adaptive:
        fixed = next(r["mean_raw"] for r in summary
                     if r["method"] == "fixed_256x128" and r["frequency"] == 1200
                     and r["source"] == row["source"])
        comparison.append({"source": row["source"], "fixed_mean_raw": fixed,
                           "adaptive_raw_improvement": float(1 - row["mean_raw"] / fixed)})
    gate = config["gate_1200hz_both_sources"]
    checks = {
        "mean_raw": all(r["mean_raw"] <= gate["adaptive_mean_raw_at_most"]
                        for r in adaptive),
        "mean_phase": all(r["mean_phase_rad"] <=
                          gate["adaptive_mean_phase_rad_at_most"] for r in adaptive),
        "readout_improvement": all(r["adaptive_raw_improvement"] >=
                                   gate["adaptive_raw_improvement_vs_fixed_at_least"]
                                   for r in comparison),
        "numerical_audit": max(max(r["raw_by_source"]) for r in audits)
        < gate["numerical_audit_max_raw_below"],
    }
    return {"summary": summary, "comparison": comparison, "gate_checks": checks,
            "all_gates_pass": all(checks.values()), "scope": config["scope"]}


def run(config: dict, output: Path) -> None:
    source = json.loads(Path(config["source_config"]).read_text())
    confirm = json.loads(Path(config["confirm_config"]).read_text())
    if config["systems"] != list(range(20268000, 20268008)):
        raise ValueError("only fixed first8 confirmed systems are permitted")
    if (
        config["frequencies_hz"], config["audit_systems"], config["taper_scale"]
    ) != ([1000, 1200], [20268200, 20268201], 7.5):
        raise ValueError("fixed conditional frequency/audit protocol changed")
    expected_variants = [
        ("fixed_256x128", 256, 128), ("range_512x128", 512, 128),
        ("modal_256x192", 256, 192), ("adaptive_512x192", 512, 192),
    ]
    if [(v["name"], v["cells"], v["modes"])
        for v in config["decoder_variants"]] != expected_variants:
        raise ValueError("fixed readout resolutions changed")
    if config["systems"] != confirm["systems"][:8]:
        raise ValueError("systems differ from prospective confirmation order")
    if digest(config["frozen_priors"]) != config["frozen_priors_sha256"]:
        raise ValueError("frozen speed prior changed")
    with np.load(config["frozen_priors"]) as data:
        speed_mean, speed_factor = data["mean_speed"], data["factor_speed"]
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(config["output_data"])
    data_root.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "source_config_sha256": digest(config["source_config"]),
        "confirm_config_sha256": digest(config["confirm_config"]),
        "frozen_priors_sha256": digest(config["frozen_priors"]),
        "implementation_sha256": digest(__file__),
        "systems": config["systems"], "audit_systems": config["audit_systems"],
    }, indent=2))
    audits = numerical_audit(config, source)
    (output / "numerical_audit.json").write_text(json.dumps(audits, indent=2))
    rows = []
    for system in config["systems"]:
        code_file = Path(config["frozen_confirm_codes"]) / f"system{system}_codes.npz"
        with np.load(code_file) as data:
            spec = json.loads(str(data["spec"]))
            beta256, coordinates = data["speed_beta"], data["speed_coordinates"]
        if spec["spectral_taper_scale"] != config["taper_scale"]:
            raise ValueError("confirmation system has different taper")
        geometry512 = generator_geometry(
            spec, 512, 32, modes=32, quadrature_nodes=160,
        )
        beta512, _ = speed_projection(
            spec, geometry512, resample_generator(speed_mean, 512),
            resample_generator(speed_factor, 512), coordinates, derivative=False,
        )
        ranges = ((np.arange(source["query_range_count"]) + 0.37)
                  / source["query_range_count"] * spec["range_m"])
        depths = np.linspace(0.1, spec["depth_m"] - 0.1, source["query_depth_count"])
        for frequency in config["frequencies_hz"]:
            truth = propagate_code(
                spec, frequency, ranges, config["teacher_cells"],
                config["teacher_modes"], config["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"],
            )
            np.savez_compressed(
                data_root / f"system{system}_f{frequency}.npz",
                modal_coefficients=truth.astype(np.complex64),
                ranges_m=ranges, depths_m=depths, frequency_hz=frequency,
                source_variants=source["source_variants"], spec=json.dumps(spec),
            )
            for variant in config["decoder_variants"]:
                beta = beta256 if variant["cells"] == 256 else beta512
                prediction = propagate_code(
                    spec, frequency, ranges, variant["cells"], variant["modes"],
                    config["teacher_quadrature_nodes"], source["eigh_device"],
                    source["eigh_batch"], beta,
                )
                rows.extend(metric_rows(
                    prediction, truth, ranges, depths, spec, source, system,
                    frequency, variant["name"],
                ))
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    (output / "summary.json").write_text(json.dumps(
        summarize(rows, audits, config), indent=2,
    ))
    files = sorted(data_root.glob("system*_f*.npz"))
    if len(files) != 2 * len(config["systems"]):
        raise ValueError("missing new frequency truth files")
    (output / "data_sha256.json").write_text(json.dumps(
        {str(path): digest(path) for path in files}, indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
