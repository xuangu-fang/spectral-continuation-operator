"""Off-grid solver truth and wavelength-adaptive sampling stress test.

Same-frequency measurements are supplied. This tests representation transfer,
not unobserved-frequency generation. All systems are development systems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from .coherent_representation import complex_metrics, regular_fit_mask
from .evaluate_coherent_chart_fields import decode_field
from .learned_wave_chart import WaveChartEncoder


def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    model = WaveChartEncoder(**cfg["model"], extent=cfg["data"]["extent"]).cuda()
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    models = {
        "conditional": load_model(config["conditional_checkpoint"]),
        "global_dictionary": load_model(config["global_checkpoint"]),
    }
    fixed = WaveChartEncoder(branches=16, degree=1, extent=1.5, learned=False, ridge=1e-5).cuda()
    angle = torch.linspace(-np.pi / 4, np.pi / 4, 17, device="cuda", dtype=torch.float64)[:-1]
    fixed.base_k.copy_(3 * np.pi * torch.stack((angle.cos(), angle.sin()), -1))
    models["fixed_cone45"] = fixed
    rows = []
    for idx in config["system_indices"]:
        with np.load(Path(config["dataset"]) / f"system{idx}_fine.npz") as z:
            fields, x, y, frequencies = z["field"], z["x"], z["y"], z["frequencies"]
            spec = json.loads(str(z["environment"]))
        speed = spec["water_speed_mps"]
        for fi, freq in enumerate(frequencies):
            field = fields[fi]
            for policy in ("fixed_grid", "four_points_per_wavelength"):
                stride = (
                    4
                    if policy == "fixed_grid"
                    else max(1, min(4, int(speed / freq / (4 * max(x[1] - x[0], y[1] - y[0])))))
                )
                mask = regular_fit_mask(field.shape, stride)
                predictions = {}
                for name, model in models.items():
                    for spreading in (False, True):
                        # Radial geometric spreading is a strong classical Ocean-specific control.
                        adjusted = field * np.sqrt(x[None]) if spreading else field
                        pred, meta = decode_field(model, adjusted, mask, x, y, freq, speed)
                        if spreading:
                            pred = pred / np.sqrt(x[None])
                        label = name + "_sqrt_r" if spreading else name
                        rows.append(
                            {
                                "system": idx,
                                "frequency": float(freq),
                                "policy": policy,
                                "stride": stride,
                                "method": label,
                                **meta,
                                **complex_metrics(pred[~mask], field[~mask]),
                            }
                        )
                        predictions[label] = pred
                iy, ix = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
                observed = field[np.ix_(iy, ix)]
                pred = RectBivariateSpline(y[iy], x[ix], observed.real)(y, x)
                pred = pred + 1j * RectBivariateSpline(y[iy], x[ix], observed.imag)(y, x)
                rows.append(
                    {
                        "system": idx,
                        "frequency": float(freq),
                        "policy": policy,
                        "stride": stride,
                        "method": "cubic_spline",
                        "total_observed_points": int(mask.sum()),
                        **complex_metrics(pred[~mask], field[~mask]),
                    }
                )
                predictions["cubic_spline"] = pred
                (output / "results.json").write_text(json.dumps(rows, indent=2))
                print(
                    json.dumps(
                        {
                            "system": idx,
                            "frequency": float(freq),
                            "policy": policy,
                            "errors": {r["method"]: r["complex_rel_l2"] for r in rows[-7:]},
                        }
                    ),
                    flush=True,
                )
                if idx == config["system_indices"][0] and freq in (125, 200):
                    np.savez_compressed(
                        output / f"system{idx}_freq{freq}_{policy}.npz",
                        target=field,
                        mask=mask,
                        x=x,
                        y=y,
                        **predictions,
                    )
    (output / "config.json").write_text(json.dumps(config, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
