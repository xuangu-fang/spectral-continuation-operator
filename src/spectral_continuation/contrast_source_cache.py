"""Sufficient quadratic forms for learning small propagated source spaces.

Targets enter supervised training/evaluation statistics, never the physics-only
coefficient solve or medium feature encoder. Background LU costs are persisted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .audit_contrast_source_rom import background_spec, source_trial
from .audit_phase_reduced_operator import discrete_operator
from .coherent_helmholtz_data import sample_environment
from .evaluate_continuous_phase_rom import reference_carriers


def source_windows(spec, axis, frequency, centers=(0.25, 0.5, 0.75), width=0.2):
    xx, yy = np.meshgrid(axis[1:-1], axis[1:-1])
    speed, _, _ = sample_environment(spec, axis)
    deficit = 1 / speed[1:-1, 1:-1] - 1
    windows, features = [np.ones(xx.size)], []
    sx, sy = spec["source"]
    for cy in centers:
        for cx in centers:
            window = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * width**2))
            weight = window / window.sum()
            mean = (weight * deficit).sum()
            variance = (weight * (deficit - mean) ** 2).sum()
            features.append(
                [
                    cx,
                    cy,
                    cx - sx,
                    cy - sy,
                    mean,
                    np.sqrt(variance),
                    spec["source_width"] * 20,
                    frequency / 10,
                    np.hypot(cx - sx, cy - sy),
                ]
            )
            windows.append(window.ravel())
    return np.stack(windows, axis=1), np.asarray(features)


def quadratic_forms(operator_basis, rhs, response_basis, residual_target, target_norm2):
    return {
        "H": operator_basis.conj().T @ operator_basis,
        "g": operator_basis.conj().T @ rhs,
        "Q": response_basis.conj().T @ response_basis,
        "t": response_basis.conj().T @ residual_target,
        "e": np.asarray(np.vdot(residual_target, residual_target).real),
        "target_norm2": np.asarray(target_norm2),
    }


def projected_coefficients(forms, transform, ridge=1e-12):
    gram = transform.conj().T @ forms["H"] @ transform
    right = transform.conj().T @ forms["g"]
    coefficient = np.linalg.solve(
        gram + ridge * np.trace(gram).real / len(gram) * np.eye(len(gram)), right
    )
    return transform @ coefficient


def quadratic_error(forms, coefficients):
    value = (
        np.vdot(coefficients, forms["Q"] @ coefficients).real
        - 2 * np.vdot(coefficients, forms["t"]).real
        + float(forms["e"])
    )
    return float(np.sqrt(max(value, 0) / float(forms["target_norm2"])))


def base_transform(branches, windows):
    transform = np.zeros((branches * windows, branches), dtype=complex)
    transform[np.arange(branches) * windows, np.arange(branches)] = 1
    return transform


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    systems = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as z:
            axis, frequencies, fields = z["axis"], z["frequencies"], z["field"]
            spec = json.loads(str(z["spec"]))
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        if np.isin(config["query_frequencies"], frequencies[observed]).any():
            raise ValueError("target frequency cannot be observed")
        carrier, _ = reference_carriers(spec, axis, frequencies, True, True)
        snapshots = fields[observed, 1:-1, 1:-1].reshape(len(observed), -1).T
        envelopes = snapshots / carrier[:, observed]
        lap, mass, rhs = discrete_operator(spec, len(axis))
        _, bg_mass, _ = discrete_operator(background_spec(spec), len(axis))
        keep = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
        systems.append(
            {
                "system": system,
                "axis": axis,
                "f": frequencies,
                "fields": fields,
                "spec": spec,
                "carrier": carrier,
                "envelopes": envelopes,
                "mass": mass,
                "rhs": rhs,
                "bg_mass": bg_mass,
                "roi": np.outer(keep, keep).ravel(),
            }
        )
    rows = []
    for frequency in config["query_frequencies"]:
        start = time.perf_counter()
        background_mass = systems[0]["bg_mass"]
        if any(not np.array_equal(s["bg_mass"], background_mass) for s in systems):
            raise ValueError("background not shareable across systems")
        lu = splu(lap + sparse.diags((2 * np.pi * frequency) ** 2 * background_mass, format="csc"))
        factor_seconds = time.perf_counter() - start
        for item in systems:
            path = output / f"system{item['system']}_f{frequency}.npz"
            if path.exists():
                raise FileExistsError(f"refuse to overwrite immutable cache {path}")
            start = time.perf_counter()
            qi = int(np.flatnonzero(item["f"] == frequency)[0])
            u0 = lu.solve(item["rhs"])
            basis = item["envelopes"] * item["carrier"][:, qi, None]
            basis = np.column_stack((basis, basis.conj()))
            windows, features = source_windows(
                item["spec"], item["axis"], frequency, width=config.get("window_width", 0.2)
            )
            expanded = (basis[:, :, None] * windows[:, None]).reshape(len(u0), -1)
            contrast = item["mass"] - background_mass
            inverse_start = time.perf_counter()
            response = source_trial(lu, contrast, expanded, frequency)
            inverse_seconds = time.perf_counter() - inverse_start
            scale = np.linalg.norm(response, axis=0).clip(1e-30)
            response /= scale[None]
            source_basis = contrast[:, None] * expanded / scale[None]
            av = lap @ response + (2 * np.pi * frequency) ** 2 * item["mass"][:, None] * response
            residual_rhs = item["rhs"] - lap @ u0 - (2 * np.pi * frequency) ** 2 * item["mass"] * u0
            roi = item["roi"]
            target = item["fields"][qi, 1:-1, 1:-1].ravel()
            forms = quadratic_forms(
                av,
                residual_rhs,
                response[roi],
                (target - u0)[roi],
                np.vdot(target[roi], target[roi]).real,
            )
            source_gram = source_basis.conj().T @ source_basis
            _, eigenvectors = np.linalg.eigh(source_gram)
            transform = base_transform(basis.shape[1], windows.shape[1])
            transforms = {"base6": transform, "source_pod6": eigenvectors[:, -basis.shape[1] :]}
            scores = {
                name: quadratic_error(forms, projected_coefficients(forms, matrix))
                for name, matrix in transforms.items()
            }
            full_coeff = np.linalg.lstsq(av, residual_rhs, rcond=1e-10)[0]
            scores["full60"] = quadratic_error(forms, full_coeff)
            # Independent direct field check guards subtraction/cancellation mistakes.
            direct_error = np.linalg.norm(
                (u0 + response @ full_coeff - target)[roi]
            ) / np.linalg.norm(target[roi])
            if abs(direct_error - scores["full60"]) > 1e-7:
                raise ValueError("quadratic loss is not equivalent to grid loss")
            record = {
                "system": item["system"],
                "frequency": frequency,
                "scores": scores,
                "columns": expanded.shape[1],
                "background_factor_seconds_shared": factor_seconds,
                "background_inverse_seconds": inverse_seconds,
                "preparation_seconds": time.perf_counter() - start,
                "evidence": "development sufficient statistics; only t/e/target_norm2 depend on target and are used only in supervised loss; never encoder or PDE coefficient solve",
            }
            np.savez_compressed(
                path,
                **forms,
                features=features,
                source_gram=source_gram,
                source_pod=eigenvectors[:, -basis.shape[1] :],
                scales=scale,
                branches=basis.shape[1],
                windows=windows.shape[1],
                metadata=json.dumps(record),
            )
            rows.append(record)
            (output / "summary.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
