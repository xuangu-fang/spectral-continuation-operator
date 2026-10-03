"""Classical spatial DMD control for continuous-frequency representation.

For range-independent outgoing Ocean fields only. Learns spatial carriers from
observed complex fields, not the simulator's modes. Query frequencies are never
used for fitting, rank selection or mode alignment. This is a mechanism audit,
not a novel learned operator or a claim of general wave propagation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .coherent_representation import complex_metrics


def spatial_modes(field, ranges, k0, tolerance=1e-5, delays=1):
    """Identify outgoing spatial exponentials after cylindrical demodulation."""
    v = np.asarray(field, dtype=np.complex128) * np.sqrt(ranges)[None]
    if delays < 1 or delays >= v.shape[1] // 2:
        raise ValueError("invalid spatial delay count")
    lifted = np.concatenate([v[:, j : j + v.shape[1] - delays + 1] for j in range(delays)], axis=0)
    x, y = lifted[:, :-1], lifted[:, 1:]
    u, singular, vh = np.linalg.svd(x, full_matrices=False)
    rank = max(1, int(np.sum(singular > singular[0] * tolerance)))
    ur, vr = u[:, :rank], vh[:rank].conj().T
    reduced = ur.conj().T @ y @ (vr / singular[:rank])
    eigenvalues = np.linalg.eigvals(reduced)
    step = np.diff(ranges)
    if not np.allclose(step, step[0]):
        raise ValueError("spatial DMD requires uniform observed range coordinates")
    wavevectors = np.angle(eigenvalues) / step[0]
    # Exclude evanescent/numerical modes; no exact seabed or mode count input.
    keep = (wavevectors > 0) & (wavevectors < k0 * 1.001)
    keep &= np.abs(np.log(np.maximum(abs(eigenvalues), 1e-30))) < 0.02
    k = np.sort(wavevectors[keep])[::-1]
    if not len(k):
        raise ValueError("no propagating mode identifiable")
    phi = np.exp(1j * ranges[:, None] * k[None])
    coefficients = np.linalg.lstsq(phi, v.T, rcond=1e-10)[0].T
    reconstruction = (coefficients @ phi.T) / np.sqrt(ranges)[None]
    return (
        k,
        coefficients,
        {
            "svd_rank": rank,
            "propagating_modes": len(k),
            "fit_complex_rel_l2": complex_metrics(reconstruction, field)["complex_rel_l2"],
            "singular_ratio": (singular / singular[0]).tolist(),
        },
    )


def modal_query(observed_f, modes, query_f, ranges, speed, dispersion):
    """Order propagating modes by k_r; interpolate each track on observed data.

    A mode is zero outside its observed support (no invented cutoff response).
    Querying a mode with fewer than two observations uses its fixed amplitude
    only at the same frequency, so it contributes nothing at a withheld query.
    """
    shape = modes[0][1].shape[0]
    prediction = np.zeros((len(query_f), shape, len(ranges)), dtype=np.complex128)
    max_rank = max(len(a[0]) for a in modes)
    for j in range(max_rank):
        indices = [i for i, (k, _) in enumerate(modes) if j < len(k)]
        if len(indices) < 2:
            continue
        freq = np.asarray(observed_f)[indices]
        k = np.asarray([modes[i][0][j] for i in indices])
        a = np.stack([modes[i][1][:, j] for i in indices])
        coordinate = (2 * np.pi * freq / speed) ** 2 - k**2 if dispersion else k
        allowed = (query_f >= freq[0]) & (query_f <= freq[-1])
        fq = query_f[allowed]
        interpolated = CubicSpline(freq, coordinate)(fq)
        kq = (
            np.sqrt(np.maximum((2 * np.pi * fq / speed) ** 2 - interpolated, 0))
            if dispersion
            else interpolated
        )
        aq = CubicSpline(freq, a, axis=0)(fq)
        prediction[allowed] += aq[:, :, None] * np.exp(1j * kq[:, None, None] * ranges[None, None])
    return prediction / np.sqrt(ranges)[None, None]


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows, fits = [], []
    with np.load(config["ocean_path"]) as z:
        f, ranges = z["frequencies_hz"], z["ranges_m"]
        names = list(z["environment_parameter_names"])
        fields = z["complex_field"]
        for index in config["system_indices"]:
            if z["split"][index] not in ("train", "val"):
                raise ValueError("development audit only")
            c = float(z["environment_parameters"][index, names.index("water_speed_mps")])
            raw = fields[index]
            field = raw[..., 0] + 1j * raw[..., 1]
            for spacing in config["frequency_spacings"]:
                observed = np.unique(np.r_[np.arange(90, 126, spacing), 125.0])
                observed_indices = np.flatnonzero(np.isin(f, observed))
                query_indices = np.flatnonzero((f >= 90) & (f <= 125) & ~np.isin(f, observed))
                fo, fq = f[observed_indices], f[query_indices]
                modes = []
                for fi in observed_indices:
                    k, a, info = spatial_modes(
                        field[fi],
                        ranges,
                        2 * np.pi * f[fi] / c,
                        config["rank_tolerance"],
                        config.get("spatial_delays", 1),
                    )
                    fits.append(
                        {
                            "system": index,
                            "spacing": spacing,
                            "frequency": float(f[fi]),
                            "k": k.tolist(),
                            **info,
                        }
                    )
                    modes.append((k, a))
                predictions = {
                    "modal_k_cubic": modal_query(fo, modes, fq, ranges, c, False),
                    "modal_dispersion_cubic": modal_query(fo, modes, fq, ranges, c, True),
                }
                carrier = np.exp(2j * np.pi * f[:, None, None] * ranges[None, None] / c)
                predictions["range_carrier_cubic"] = (
                    CubicSpline(fo, field[observed_indices] / carrier[observed_indices], axis=0)(fq)
                    * carrier[query_indices]
                )
                for method, p in predictions.items():
                    errors = []
                    for qi, fi in enumerate(query_indices):
                        row = {
                            "system": index,
                            "method": method,
                            "frequency_spacing": spacing,
                            "query_frequency": float(f[fi]),
                            "observed_count": len(fo),
                            **complex_metrics(p[qi], field[fi]),
                        }
                        rows.append(row)
                        errors.append(row["complex_rel_l2"])
                    print(
                        json.dumps(
                            {
                                "system": index,
                                "spacing": spacing,
                                "method": method,
                                "macro_error": float(np.mean(errors)),
                            }
                        ),
                        flush=True,
                    )
                np.savez_compressed(
                    output / f"system{index}_spacing{spacing}_predictions.npz",
                    frequencies=fq,
                    ranges=ranges,
                    depths=z["depths_m"],
                    target=field[query_indices],
                    **predictions,
                )
                (output / "results.json").write_text(json.dumps(rows, indent=2))
                (output / "observed_fit.json").write_text(json.dumps(fits, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
