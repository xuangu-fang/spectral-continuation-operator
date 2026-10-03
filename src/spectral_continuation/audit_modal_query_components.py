"""Attribute spectral-query error with explicitly labelled target-truth oracles.

Only the baseline is deployable. Oracle branches substitute target modal phases,
amplitudes or support to diagnose what remains; they are never training results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .audit_modal_spectral_transport import spatial_modes
from .coherent_representation import complex_metrics


def interpolated_states(observed_f, modes, query_f):
    count = max(len(k) for k, _ in modes)
    channels = modes[0][1].shape[0]
    kq = np.full((len(query_f), count), np.nan)
    aq = np.zeros((len(query_f), channels, count), dtype=np.complex128)
    for j in range(count):
        ids = [i for i, (k, _) in enumerate(modes) if len(k) > j]
        if len(ids) < 2:
            continue
        f = np.asarray(observed_f)[ids]
        valid = (query_f >= f[0]) & (query_f <= f[-1])
        kq[valid, j] = CubicSpline(f, [modes[i][0][j] for i in ids])(query_f[valid])
        aq[valid, :, j] = CubicSpline(f, np.stack([modes[i][1][:, j] for i in ids]), axis=0)(
            query_f[valid]
        )
    return kq, aq


def compose(k, a, ranges):
    return (a @ np.exp(1j * np.nan_to_num(k)[:, None] * ranges[None])) / np.sqrt(ranges)[None]


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows, diagnostics = [], []
    with np.load(config["ocean_path"]) as z:
        f, ranges = z["frequencies_hz"], z["ranges_m"]
        all_fields = z["complex_field"]
        names = list(z["environment_parameter_names"])
        band = np.flatnonzero((f >= 90) & (f <= 125))
        for system in config["systems"]:
            if z["split"][system] not in ("train", "val"):
                raise ValueError("development only")
            speed = z["environment_parameters"][system, names.index("water_speed_mps")]
            raw = all_fields[system]
            fields = raw[..., 0] + 1j * raw[..., 1]
            # Target-frequency state is used ONLY below in named oracle diagnostics.
            states = {}
            for fi in band:
                k, a, info = spatial_modes(
                    fields[fi],
                    ranges,
                    2 * np.pi * f[fi] / speed,
                    config["rank_tolerance"],
                    config["spatial_delays"],
                )
                states[fi] = k, a
                diagnostics.append(
                    {
                        "system": system,
                        "frequency": float(f[fi]),
                        "scope": "full-target state fit, oracle diagnostic",
                        **info,
                    }
                )
            for spacing in config["frequency_spacings"]:
                observations = np.unique(np.r_[np.arange(90, 126, spacing), 125.0])
                observed = np.flatnonzero(np.isin(f, observations))
                query = np.array([i for i in band if i not in observed])
                kp, ap = interpolated_states(f[observed], [states[i] for i in observed], f[query])
                for qi, fi in enumerate(query):
                    kt, at = states[fi]
                    rank = max(len(kt), kp.shape[-1])
                    true_k, pred_k = np.zeros(rank), np.zeros(rank)
                    true_a, pred_a = (
                        np.zeros((len(at), rank), complex),
                        np.zeros((len(at), rank), complex),
                    )
                    true_k[: len(kt)], true_a[:, : len(kt)] = kt, at
                    pred_k[: kp.shape[-1]], pred_a[:, : kp.shape[-1]] = kp[qi], ap[qi]
                    supported = np.zeros(rank, dtype=bool)
                    supported[: kp.shape[-1]] = np.isfinite(kp[qi])
                    true_present = np.arange(rank) < len(kt)
                    # Oracle phase includes correct support. Oracle amplitude keeps
                    # predicted phases only where a track exists and uses true phase
                    # for genuinely absent tracks, so it isolates transport error.
                    corrected_k = np.where(supported & true_present, pred_k, true_k)
                    support_a = pred_a.copy()
                    support_a[:, ~true_present] = 0
                    missing = true_present & ~supported
                    support_a[:, missing] = true_a[:, missing]
                    variants = {
                        "deployable_modal_cubic": compose(pred_k, pred_a, ranges),
                        "oracle_true_phase": compose(true_k, pred_a * true_present[None], ranges),
                        "oracle_true_amplitude_and_missing_phase": compose(
                            corrected_k, true_a, ranges
                        ),
                        "oracle_missing_support": compose(corrected_k, support_a, ranges),
                        "oracle_all_modal_state": compose(true_k, true_a, ranges),
                    }
                    missing_field = (
                        compose(true_k[missing], true_a[:, missing], ranges)
                        if missing.any()
                        else np.zeros_like(fields[fi])
                    )
                    missing_energy = float(
                        np.sum(abs(missing_field) ** 2) / np.sum(abs(fields[fi]) ** 2)
                    )
                    for name, p in variants.items():
                        rows.append(
                            {
                                "system": system,
                                "spacing": spacing,
                                "frequency": float(f[fi]),
                                "method": name,
                                "uses_query_truth": name.startswith("oracle"),
                                "target_modes": len(kt),
                                "missing_tracks": int(missing.sum()),
                                "missing_component_energy_ratio": missing_energy,
                                **complex_metrics(p, fields[fi]),
                            }
                        )
                for name in variants:
                    current = [
                        r
                        for r in rows
                        if r["system"] == system and r["spacing"] == spacing and r["method"] == name
                    ]
                    print(
                        json.dumps(
                            {
                                "system": system,
                                "spacing": spacing,
                                "method": name,
                                "macro_error": float(
                                    np.mean([r["complex_rel_l2"] for r in current])
                                ),
                            }
                        ),
                        flush=True,
                    )
                (output / "results.json").write_text(json.dumps(rows, indent=2))
                (output / "target_modal_fit.json").write_text(json.dumps(diagnostics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
