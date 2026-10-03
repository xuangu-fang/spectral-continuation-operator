"""R1 reference audit: separate frequency interpolation from spatial capacity.

All query frequencies are withheld from the interpolator. This reuses development
systems and measures same-system coding, NOT trained operator generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .coherent_representation import complex_metrics


def run(config: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with np.load(config["ocean_path"]) as z:
        fields = z["complex_field"]
        f = z["frequencies_hz"]
        names = list(z["environment_parameter_names"])
        for index in config["system_indices"]:
            if z["split"][index] not in ("train", "val"):
                raise ValueError("development audit only")
            c = z["environment_parameters"][index, names.index("water_speed_mps")]
            raw = fields[index]
            u = raw[..., 0] + 1j * raw[..., 1]
            delay = np.broadcast_to(z["ranges_m"][None, :] / c, u.shape[1:])
            for spacing in config["frequency_spacings"]:
                observed = np.unique(np.r_[np.arange(90, 126, spacing), 125.0])
                fit = np.flatnonzero(np.isin(f, observed))
                query = np.flatnonzero((f >= 90) & (f <= 125) & ~np.isin(f, observed))
                for demodulate in (False, True):
                    carrier = (
                        np.exp(2j * np.pi * f[:, None, None] * delay[None])
                        if demodulate
                        else np.ones_like(u)
                    )
                    coded = u[fit] / carrier[fit]
                    pred = CubicSpline(f[fit], coded, axis=0)(f[query]) * carrier[query]
                    method = "range_carrier_cubic" if demodulate else "raw_complex_cubic"
                    for n, fi in enumerate(query):
                        rows.append(
                            {
                                "system": str(z["environment_ids"][index]),
                                "method": method,
                                "frequency_spacing": spacing,
                                "query_frequency": float(f[fi]),
                                "observed_count": len(fit),
                                **complex_metrics(pred[n], u[fi]),
                            }
                        )
                    # A spectral span diagnostic uses ONLY observed slices.
                    singular = np.linalg.svd(coded.reshape(len(fit), -1), compute_uv=False)
                    energy = np.cumsum(singular**2) / np.sum(singular**2)
                    print(
                        json.dumps(
                            {
                                "system": index,
                                "method": method,
                                "spacing": spacing,
                                "error": float(
                                    np.mean([r["complex_rel_l2"] for r in rows[-len(query) :]])
                                ),
                                "rank99_observed": int(np.searchsorted(energy, 0.99) + 1),
                            }
                        ),
                        flush=True,
                    )
    (output / "results.json").write_text(json.dumps(rows, indent=2))
    (output / "config.json").write_text(json.dumps(config, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
