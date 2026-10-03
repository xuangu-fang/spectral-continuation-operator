"""Cutoff-aware classical control on independent fractional-frequency truth.

Uses homogeneous water and a pressure-release surface: modal depth shape is
sin(k_z z). This assumption is explicit and Ocean-specific, not a neural result.
Modal strengths and k_z are inferred only from observed fields; unseen queries
are evaluated at new depth/range coordinates directly from these representations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

from .audit_modal_spectral_transport import modal_query, spatial_modes
from .coherent_representation import complex_metrics


def scalar_modes(frequencies, modes, depths, speed):
    states, diagnostics = [], []
    for frequency, (kr, coefficients) in zip(frequencies, modes, strict=True):
        kz2 = np.maximum((2 * np.pi * frequency / speed) ** 2 - kr**2, 0)
        vertical = np.sin(depths[:, None] * np.sqrt(kz2)[None])
        strength = np.sum(vertical * coefficients, axis=0) / np.sum(vertical**2, axis=0)
        reconstructed = vertical * strength[None]
        error = np.linalg.norm(reconstructed - coefficients) / np.linalg.norm(coefficients)
        states.append((kz2, strength))
        diagnostics.append(
            {"frequency": float(frequency), "shape_error": float(error), "modes": len(kr)}
        )
    return states, diagnostics


def scalar_query(observed_f, states, query_f, x, z, speed, birth_rule):
    prediction = np.zeros((len(query_f), len(z), len(x)), dtype=np.complex128)
    metadata = []
    max_count = max(len(k) for k, _ in states)
    for mode in range(max_count):
        ids = np.array([i for i, (k, _) in enumerate(states) if len(k) > mode])
        frequency = observed_f[ids]
        kz2 = np.array([states[i][0][mode] for i in ids])
        strength = np.array([states[i][1][mode] for i in ids])
        if len(ids) == 1:
            kz2_query = np.full(len(query_f), kz2[0])
        else:
            kz2_query = CubicSpline(frequency, kz2)(query_f)
        kz_query = np.sqrt(np.maximum(kz2_query, 0))
        kr_query = np.sqrt(np.maximum((2 * np.pi * query_f / speed) ** 2 - kz2_query, 0))
        born_in_band = ids[0] > 0
        cutoff = None
        if birth_rule == "zero_anchor":
            observed_strength = np.zeros(len(observed_f), dtype=np.complex128)
            observed_strength[ids] = strength
            q = CubicSpline(observed_f, observed_strength)(query_f)
            if born_in_band:
                q[query_f <= observed_f[ids[0] - 1]] = 0
        elif birth_rule == "cutoff_linear":
            if born_in_band:
                low, high = observed_f[ids[0] - 1], frequency[0]
                cutoff = (low + high) / 2
                if len(ids) >= 2:
                    # Use a signed amplitude in the first observed modal phase.
                    phase = strength[0] / max(abs(strength[0]), 1e-30)
                    a0, a1 = (strength[:2] / phase).real
                    if a1 > a0 > 0:
                        cutoff = float(
                            np.clip(high - a0 * (frequency[1] - high) / (a1 - a0), low, high - 1e-8)
                        )
                q = CubicSpline(np.r_[cutoff, frequency], np.r_[0j, strength])(query_f)
                q[query_f <= cutoff] = 0
            else:
                q = CubicSpline(frequency, strength)(query_f)
        else:
            raise ValueError(birth_rule)
        propagating = kz2_query < (2 * np.pi * query_f / speed) ** 2
        q = q * propagating
        prediction += (
            q[:, None, None]
            * np.sin(kz_query[:, None, None] * z[None, :, None])
            * np.exp(1j * kr_query[:, None, None] * x[None, None])
        )
        metadata.append(
            {
                "mode": mode + 1,
                "observed_frequencies": frequency.tolist(),
                "born_in_band": bool(born_in_band),
                "estimated_cutoff": cutoff,
                "single_observation": len(ids) == 1,
            }
        )
    return prediction / np.sqrt(x)[None, None], metadata


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows, diagnostic = [], []
    with np.load(config["ocean_path"]) as archive:
        all_f, old_x, old_z = archive["frequencies_hz"], archive["ranges_m"], archive["depths_m"]
        fields = archive["complex_field"]
        names = list(archive["environment_parameter_names"])
        for system in config["systems"]:
            if archive["split"][system] not in ("train", "val"):
                raise ValueError("development only")
            speed = archive["environment_parameters"][system, names.index("water_speed_mps")]
            raw = fields[system]
            field = raw[..., 0] + 1j * raw[..., 1]
            with np.load(Path(config["query_data"]) / f"system{system}_fine.npz") as z:
                target, query_f, x, depths = z["field"], z["frequencies"], z["x"], z["y"]
            for spacing in config["frequency_spacings"]:
                observed_f = np.unique(np.r_[np.arange(90, 126, spacing), 125.0])
                if np.isin(query_f, observed_f).any():
                    raise ValueError("query frequencies must have no observations")
                ids = np.flatnonzero(np.isin(all_f, observed_f))
                modes = []
                for fi in ids:
                    kr, a, _ = spatial_modes(
                        field[fi],
                        old_x,
                        2 * np.pi * all_f[fi] / speed,
                        config["rank_tolerance"],
                        config["spatial_delays"],
                    )
                    modes.append((kr, a))
                states, fits = scalar_modes(observed_f, modes, old_z, speed)
                diagnostic.append(
                    {"system": system, "spacing": spacing, "observed_shape_fit": fits}
                )
                old_depth_prediction = modal_query(observed_f, modes, query_f, x, speed, False)
                predictions = {
                    "modal_zero_support": CubicSpline(old_z, old_depth_prediction, axis=1)(depths)
                }
                for rule in ("zero_anchor", "cutoff_linear"):
                    p, details = scalar_query(observed_f, states, query_f, x, depths, speed, rule)
                    predictions[f"modal_scalar_{rule}"] = p
                    diagnostic.append(
                        {"system": system, "spacing": spacing, "rule": rule, "tracks": details}
                    )
                carrier_observed = np.exp(
                    2j * np.pi * observed_f[:, None, None] * old_x[None, None] / speed
                )
                envelope = field[ids] / carrier_observed
                envelope_query = CubicSpline(observed_f, envelope, axis=0)(query_f)
                p = []
                for qi, frequency in enumerate(query_f):
                    smooth = RectBivariateSpline(old_z, old_x, envelope_query[qi].real)(
                        depths, x
                    ) + 1j * RectBivariateSpline(old_z, old_x, envelope_query[qi].imag)(depths, x)
                    p.append(smooth * np.exp(2j * np.pi * frequency * x[None] / speed))
                predictions["range_carrier_cubic"] = np.stack(p)
                for method, prediction in predictions.items():
                    errors = []
                    for qi, frequency in enumerate(query_f):
                        row = {
                            "system": system,
                            "frequency_spacing": spacing,
                            "query_frequency": float(frequency),
                            "method": method,
                            "target_frequency_observations": 0,
                            "direct_new_coordinate_truth": True,
                            **complex_metrics(prediction[qi], target[qi]),
                        }
                        rows.append(row)
                        errors.append(row["complex_rel_l2"])
                    print(
                        json.dumps(
                            {
                                "system": system,
                                "spacing": spacing,
                                "method": method,
                                "macro_error": float(np.mean(errors)),
                            }
                        ),
                        flush=True,
                    )
                if system == config["systems"][0]:
                    np.savez_compressed(
                        output / f"system{system}_spacing{spacing}_predictions.npz",
                        target=target,
                        frequencies=query_f,
                        x=x,
                        y=depths,
                        **predictions,
                    )
                (output / "results.json").write_text(json.dumps(rows, indent=2))
                (output / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
