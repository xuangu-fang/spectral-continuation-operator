"""Full physical ROI and direct fine-grid audit, with fixed global sample masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from .coherent_representation import complex_metrics, regular_fit_mask
from .evaluate_coherent_chart_fields import decode_field
from .learned_coherent_kernel import CoherentKernelEncoder
from .learned_wave_chart import build_chart_model


def load_models(config):
    models = {}
    for name, path in config["checkpoints"].items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        model = build_chart_model(cfg["model"], cfg["data"]["extent"]).cuda()
        model.load_state_dict(checkpoint["model"])
        models[name] = (model.eval(), cfg["data"]["extent"])
    for epsilon, ridge in config["gaussian_baselines"]:
        model = CoherentKernelEncoder(mode="global", ridge=ridge, extent=config["extent"]).cuda()
        with torch.no_grad():
            model.encoder["delta"][[2, 3]] = float(np.arctanh(np.log(epsilon) / 2))
        models[f"gaussian_e{epsilon:g}_ridge{ridge:g}"] = (model.eval(), config["extent"])
    return models


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(4)
    models, rows = load_models(config), []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as z:
            all_fields, all_f, axis = z["field"], z["frequencies"], z["axis"]
            spec = json.loads(str(z["spec"]))
        keep = (axis >= config["roi"][0]) & (axis <= config["roi"][1])
        ids, coarse_axis = np.flatnonzero(keep), axis[keep]
        # Identical physical extent for nested evaluation, no extrapolated truth.
        query_axis, fine_fields, fine_f = coarse_axis, None, None
        if config.get("fine_data"):
            with np.load(Path(config["fine_data"]) / f"system_{system}_grid513.npz") as z:
                fine_axis, fine_f = z["axis"], z["frequencies"]
                inside = (fine_axis >= coarse_axis[0]) & (fine_axis <= coarse_axis[-1])
                fine_ids, query_axis = np.flatnonzero(inside), fine_axis[inside]
                fine_fields = z["field"][:, fine_ids][:, :, fine_ids]
        qx, qy = np.meshgrid(query_axis, query_axis)
        source_distance = np.sqrt((qx - spec["source"][0]) ** 2 + (qy - spec["source"][1]) ** 2)
        for frequency in config["frequencies"]:
            fi = int(np.flatnonzero(np.isclose(all_f, frequency))[0])
            field = all_fields[fi][np.ix_(ids, ids)]
            target = (
                field
                if fine_fields is None
                else fine_fields[int(np.flatnonzero(np.isclose(fine_f, frequency))[0])]
            )
            for stride in config["strides"]:
                mask = regular_fit_mask(field.shape, stride)
                query_mask = np.ones(target.shape, dtype=bool)
                factor = (len(query_axis) - 1) // (len(coarse_axis) - 1)
                # Fitted coordinates excluded in both same-grid and independent-grid metrics.
                iy, ix = np.where(mask)
                query_mask[iy * factor, ix * factor] = False
                predictions = {}
                for name, (model, extent) in models.items():
                    pred, meta = decode_field(
                        model,
                        field,
                        mask,
                        coarse_axis,
                        coarse_axis,
                        frequency,
                        1.0,
                        extent=extent,
                        query_x=query_axis,
                        query_y=query_axis,
                    )
                    predictions[name] = pred
                    for region, region_mask in (
                        ("full_roi", query_mask),
                        (
                            "near_source_3sigma",
                            query_mask & (source_distance <= 3 * spec["source_width"]),
                        ),
                        (
                            "outside_source_3sigma",
                            query_mask & (source_distance > 3 * spec["source_width"]),
                        ),
                    ):
                        if not region_mask.any():
                            continue
                        row = {
                            "system": system,
                            "frequency": frequency,
                            "stride": stride,
                            "method": name,
                            "region": region,
                            "independent_grid": fine_fields is not None,
                            "target_energy_fraction": float(
                                np.sum(abs(target[region_mask]) ** 2)
                                / np.sum(abs(target[query_mask]) ** 2)
                            ),
                            **meta,
                            **complex_metrics(pred[region_mask], target[region_mask]),
                        }
                        rows.append(row)
                        if region == "full_roi":
                            print(json.dumps(row), flush=True)
                iy, ix = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
                samples = field[np.ix_(iy, ix)]
                pred = RectBivariateSpline(coarse_axis[iy], coarse_axis[ix], samples.real)(
                    query_axis, query_axis
                ) + 1j * RectBivariateSpline(coarse_axis[iy], coarse_axis[ix], samples.imag)(
                    query_axis, query_axis
                )
                predictions["cubic_spline"] = pred
                for region, region_mask in (
                    ("full_roi", query_mask),
                    (
                        "near_source_3sigma",
                        query_mask & (source_distance <= 3 * spec["source_width"]),
                    ),
                    (
                        "outside_source_3sigma",
                        query_mask & (source_distance > 3 * spec["source_width"]),
                    ),
                ):
                    if region_mask.any():
                        rows.append(
                            {
                                "system": system,
                                "frequency": frequency,
                                "stride": stride,
                                "method": "cubic_spline",
                                "region": region,
                                "independent_grid": fine_fields is not None,
                                "total_observed_points": int(mask.sum()),
                                **complex_metrics(pred[region_mask], target[region_mask]),
                            }
                        )
                if system == config["systems"][0]:
                    np.savez_compressed(
                        output / f"system{system}_frequency{frequency}_stride{stride}.npz",
                        target=target,
                        query_mask=query_mask,
                        x=query_axis,
                        y=query_axis,
                        **predictions,
                    )
                (output / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
