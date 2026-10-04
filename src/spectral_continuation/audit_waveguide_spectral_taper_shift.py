"""Frozen rank-256 waveguide representation under a medium-correlation shift.

The 16x16 cosine basis and perturbation RMS are unchanged. Only the spectral
taper length differs from the training family. This is an OOD diagnostic within
one PDE, not a second physical-domain confirmation or a learned-method claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_continuous_generator import resample_generator
from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_new_systems import context_data
from .audit_waveguide_spectral_context_robustness import digest
from .audit_waveguide_spectral_rank256_inference import infer
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis, spectral_environment


def shifted_environment(seed: int, terms: int, rms: float, taper_scale: float) -> dict:
    """Draw the same cosine family with a wider, matched-RMS spectral taper."""
    if taper_scale <= 0:
        raise ValueError("taper scale must be positive")
    spec = spectral_environment(seed, terms, rms)
    rng = np.random.default_rng(seed + 1_000_000)
    indices = np.arange(terms)
    taper = np.exp(-0.5 * (indices / taper_scale) ** 2)
    coefficients = rng.standard_normal((terms, terms)) * taper[:, None] * taper[None]
    coefficients[0, 0] = 0.0
    samples = np.cos(np.pi * np.linspace(0, 1, 128)[:, None] * indices)
    sample = samples @ coefficients @ samples.T
    coefficients *= rms / np.sqrt(np.mean(sample**2))
    spec["spectral_coefficients_m_s"] = coefficients.tolist()
    spec["spectral_family"] = f"correlated_cosine_{terms}x{terms}_taper{taper_scale:g}"
    spec["spectral_taper_scale"] = float(taper_scale)
    return spec


def numerical_audit(config: dict, source: dict) -> list[dict]:
    rows = []
    for system in config["audit_systems"]:
        spec = shifted_environment(
            system, config["terms"], config["perturbation_rms_m_s"],
            config["target_taper_scale"],
        )
        ranges = (np.arange(513) + 0.37) / 513 * spec["range_m"]
        roi = ranges >= source["roi_min_range_m"]
        for frequency in source["query_frequencies"]:
            low = propagate_code(
                spec, frequency, ranges, source["teacher_cells"],
                source["teacher_modes"], source["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"],
            )
            high = propagate_code(
                spec, frequency, ranges, source["audit_cells"],
                source["audit_modes"], source["audit_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"],
            )
            raw = modal_raw(low[roi], high[roi])
            rows.append({"system": system, "frequency": frequency,
                         "raw_by_source": raw.tolist()})
            print(json.dumps({"numerical_audit": rows[-1]}), flush=True)
            if np.max(raw) >= config["gate_800hz_both_sources"]["numerical_audit_max_raw_below"]:
                raise ValueError("independent numerical truth audit failed")
    return rows


def metric_rows(prediction, truth, ranges, depths, spec, source, system, frequency, method):
    roi = ranges >= source["roi_min_range_m"]
    modal = modal_raw(prediction[roi], truth[roi])
    basis = depth_basis(depths, spec["depth_m"], truth.shape[1])
    target = (basis @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)).reshape(
        len(depths), len(ranges), len(source["source_variants"])
    )
    field = (basis[:, :prediction.shape[1]] @ prediction.transpose(1, 0, 2).reshape(
        prediction.shape[1], -1)).reshape(
            len(depths), len(ranges), len(source["source_variants"])
        )
    return [
        {"system": system, "frequency": frequency, "source": variant,
         "method": method, "modal_parseval_raw": float(modal[index]),
         **complex_metrics(field[:, roi, index], target[:, roi, index])}
        for index, variant in enumerate(source["source_variants"])
    ]


def run(config: dict, output: Path) -> None:
    source = json.loads(Path(config["source_config"]).read_text())
    if config["rank"] != 256 or config["terms"] != 16:
        raise ValueError("fixed rank256 and 16x16 OOD protocol required")
    if source["spectral_terms"] != config["terms"]:
        raise ValueError("target and training cosine basis sizes differ")
    if config["training_taper_scale"] != 5.0 or config["target_taper_scale"] != 10.0:
        raise ValueError("this preregistered diagnostic compares only 5 to 10")
    if set(config["systems"]) & set(source["systems"]):
        raise ValueError("OOD seeds overlap IID confirmation")
    with np.load(config["training_labels"]) as data:
        training = data["target32"].astype(float)
    mean, factor = fit_prior(training, config["rank"])
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(config["output_data"])
    data_root.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "scope": config["scope"],
        "source_config_sha256": digest(config["source_config"]),
        "training_labels_sha256": digest(config["training_labels"]),
        "implementation_sha256": digest(__file__),
        "systems": config["systems"],
    }, indent=2))
    audits = numerical_audit(config, source)
    (output / "numerical_audit.json").write_text(json.dumps(audits, indent=2))
    rows, histories = [], []
    for system in config["systems"]:
        spec = shifted_environment(
            system, config["terms"], config["perturbation_rms_m_s"],
            config["target_taper_scale"],
        )
        ranges_context, observed = context_data(spec, source)
        beta, history = infer(
            spec, observed, ranges_context, source, mean, factor,
            config["coarse_steps"], config["fine_steps"],
        )
        histories.extend({"system": system, **entry} for entry in history)
        coarse = generator_geometry(spec, 64, mean.shape[1], modes=32)
        true_beta = true_projection(spec, coarse)
        coordinates = np.linalg.lstsq(
            factor.reshape(-1, config["rank"]), (true_beta - mean).ravel(), rcond=None
        )[0]
        projected_beta = resample_generator(mean, source["decoder_cells"]) + np.einsum(
            "rdk,k->rd", resample_generator(factor, source["decoder_cells"]), coordinates
        )
        geometry = generator_geometry(
            spec, source["decoder_cells"], mean.shape[1],
            modes=source["decoder_modes"],
            quadrature_nodes=source["teacher_quadrature_nodes"],
        )
        true_beta_fine = true_projection(spec, geometry)
        np.savez_compressed(
            output / f"system{system}_codes.npz", beta=beta,
            context_ranges_m=ranges_context,
            observed=observed.astype(np.complex64), spec=json.dumps(spec),
        )
        ranges = ((np.arange(source["query_range_count"]) + 0.37)
                  / source["query_range_count"] * spec["range_m"])
        depths = np.linspace(0.1, spec["depth_m"] - 0.1, source["query_depth_count"])
        for frequency in source["query_frequencies"]:
            truth = propagate_code(
                spec, frequency, ranges, source["teacher_cells"],
                source["teacher_modes"], source["teacher_quadrature_nodes"],
                source["eigh_device"], source["eigh_batch"],
            )
            np.savez_compressed(
                data_root / f"system{system}_f{frequency}.npz",
                modal_coefficients=truth.astype(np.complex64),
                ranges_m=ranges, depths_m=depths, frequency_hz=frequency,
                source_variants=source["source_variants"], spec=json.dumps(spec),
            )
            for method, candidate in [
                ("rank256_context", beta),
                ("rank256_true_medium_projection_oracle", projected_beta),
                ("true_medium32_oracle", true_beta_fine),
            ]:
                prediction = propagate_code(
                    spec, frequency, ranges, source["decoder_cells"],
                    source["decoder_modes"], source["teacher_quadrature_nodes"],
                    source["eigh_device"], source["eigh_batch"], candidate,
                )
                rows.extend(metric_rows(
                    prediction, truth, ranges, depths, spec, source, system, frequency,
                    method,
                ))
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        print(json.dumps({"system": system, "completed_rows": len(rows)}), flush=True)
    summary = []
    for method in ["rank256_context", "rank256_true_medium_projection_oracle",
                   "true_medium32_oracle"]:
        for frequency in source["query_frequencies"]:
            for variant in source["source_variants"]:
                sample = [r for r in rows if r["method"] == method
                          and r["frequency"] == frequency and r["source"] == variant]
                summary.append({
                    "method": method, "frequency": frequency, "source": variant,
                    "systems": len(sample),
                    "mean_raw": float(np.mean([r["complex_rel_l2"] for r in sample])),
                    "mean_phase_rad": float(np.mean([
                        r["energy_weighted_phase_rmse_rad"] for r in sample
                    ])),
                    "max_raw": float(np.max([r["complex_rel_l2"] for r in sample])),
                    "systems_raw_at_most_0p10": sum(
                        r["complex_rel_l2"] <= 0.1 for r in sample
                    ),
                })
    gate = config["gate_800hz_both_sources"]
    checked = [r for r in summary if r["method"] == "rank256_context"
               and r["frequency"] == 800]
    checks = {
        "mean_raw": all(r["mean_raw"] <= gate["mean_raw_at_most"] for r in checked),
        "mean_phase": all(r["mean_phase_rad"] <= gate["mean_phase_rad_at_most"]
                          for r in checked),
        "systems_below_raw": all(
            r["systems_raw_at_most_0p10"]
            >= gate["systems_raw_at_most_0p10_at_least"] for r in checked
        ),
        "numerical_audit": max(
            max(r["raw_by_source"]) for r in audits
        ) < gate["numerical_audit_max_raw_below"],
    }
    (output / "summary.json").write_text(json.dumps({
        "summary": summary, "gate_checks": checks, "all_gates_pass": all(checks.values()),
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
