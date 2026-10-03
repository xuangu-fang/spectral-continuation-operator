"""Full-field POU evaluation of shared local encoders, without query labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from .coherent_representation import complex_metrics, regular_fit_mask
from .learned_wave_chart import WaveChartEncoder


@torch.no_grad()
def decode_field(
    model,
    field,
    fit_mask,
    x,
    y,
    frequency,
    speed,
    extent=1.5,
    query_x=None,
    query_y=None,
    max_observed=64,
    seed=43,
):
    query_x = x if query_x is None else query_x
    query_y = y if query_y is None else query_y
    physical_radius = extent * speed / frequency
    lo, hi = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    count = np.maximum(np.ceil((hi - lo) / physical_radius).astype(int), 1)
    spacing = (hi - lo) / count
    cx, cy = [np.linspace(lo[d], hi[d], count[d] + 1) for d in range(2)]
    centers = np.stack(np.meshgrid(cx, cy), -1).reshape(-1, 2)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack((xx, yy), -1)
    observed_x = coords[fit_mask]
    observed_y = field[fit_mask]
    qx, qy = np.meshgrid(query_x, query_y)
    query = np.stack((qx, qy), -1).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    nodes, values, weights, scales, queries, qindices, qweights, counts = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for center in centers:
        dx = observed_x - center
        inside = np.max(np.abs(dx / spacing), axis=-1) < 1 - 1e-10
        ids = np.flatnonzero(inside)
        if len(ids) < 4:
            raise ValueError("insufficient context in full-field chart")
        if len(ids) > max_observed:
            ids = rng.choice(ids, max_observed, replace=False)
        current_x = dx[ids] / physical_radius
        current_y = observed_y[ids]
        scale = max(float(np.sqrt(np.mean(abs(current_y) ** 2))), 1e-20)
        node = np.zeros((max_observed, 2))
        node[: len(ids)] = current_x
        value = np.zeros(max_observed, dtype=np.complex64)
        value[: len(ids)] = current_y / scale
        weight = np.zeros(max_observed)
        weight[: len(ids)] = np.prod(np.maximum(1 - abs(current_x), 0.05), -1)
        delta = query - center
        w = np.prod(np.maximum(1 - abs(delta / spacing), 0), -1)
        qi = np.flatnonzero(w > 1e-12)
        nodes.append(node)
        values.append(value)
        weights.append(weight)
        scales.append(scale)
        queries.append(delta[qi] / physical_radius)
        qindices.append(qi)
        qweights.append(w[qi])
        counts.append(len(ids))
    max_query = max(map(len, queries))
    padded_query = np.zeros((len(queries), max_query, 2))
    for i, q in enumerate(queries):
        padded_query[i, : len(q)] = q
    device = model.base_k.device
    X = torch.tensor(np.asarray(nodes), device=device, dtype=torch.float32)
    Y = torch.tensor(np.asarray(values), device=device, dtype=torch.complex64)
    W = torch.tensor(np.asarray(weights), device=device, dtype=torch.float32)
    Q = torch.tensor(padded_query, device=device, dtype=torch.float32)
    out, total = np.zeros(len(query), dtype=np.complex128), np.zeros(len(query))
    for start in range(0, len(nodes), 16):
        sl = slice(start, start + 16)
        prediction = model(X[sl], Y[sl], Q[sl], W[sl]).cpu().numpy()
        for j in range(len(prediction)):
            i = start + j
            pred = prediction[j, : len(qindices[i])] * scales[i]
            out[qindices[i]] += pred * qweights[i]
            total[qindices[i]] += qweights[i]
    if total.min() < 0.99:
        raise RuntimeError("partition does not cover requested domain")
    return (out / total).reshape(len(query_y), len(query_x)), {
        "charts": len(nodes),
        "minimum_context_per_chart": min(counts),
        "maximum_context_per_chart": max(counts),
        "total_observed_points": int(fit_mask.sum()),
    }


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    learned = WaveChartEncoder(**cfg["model"], extent=cfg["data"]["extent"]).cuda()
    learned.load_state_dict(checkpoint["model"])
    models = {"learned": learned}
    for cone in (180, 45):
        fixed = WaveChartEncoder(
            branches=16, degree=1, extent=1.5, learned=False, ridge=1e-5
        ).cuda()
        a = torch.linspace(
            -cone * np.pi / 180, cone * np.pi / 180, 17, device="cuda", dtype=torch.float64
        )[:-1]
        fixed.base_k.copy_(2 * np.pi * 1.5 * torch.stack((a.cos(), a.sin()), -1))
        models[f"fixed16_affine_cone{cone}"] = fixed
    rows = []
    with np.load(config["ocean_path"]) as z:
        all_fields = z["complex_field"]
        x, y, f = z["ranges_m"], z["depths_m"], z["frequencies_hz"]
        names = list(z["environment_parameter_names"])
        for idx in config["system_indices"]:
            if z["split"][idx] not in ("train", "val"):
                raise ValueError("development only")
            speed = float(z["environment_parameters"][idx, names.index("water_speed_mps")])
            for freq in config["frequencies"]:
                fi = int(np.flatnonzero(np.isclose(f, freq))[0])
                raw = all_fields[idx, fi]
                field = raw[..., 0] + 1j * raw[..., 1]
                mask = regular_fit_mask(field.shape, config["stride"])
                predictions = {}
                for name, model in models.items():
                    pred, meta = decode_field(model, field, mask, x, y, freq, speed)
                    row = {
                        "system": str(z["environment_ids"][idx]),
                        "frequency": freq,
                        "method": name,
                        **meta,
                        **complex_metrics(pred[~mask], field[~mask]),
                    }
                    predictions[name] = pred
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                iy, ix = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
                samples = field[np.ix_(iy, ix)]
                pred = RectBivariateSpline(y[iy], x[ix], samples.real)(y, x)
                pred = pred + 1j * RectBivariateSpline(y[iy], x[ix], samples.imag)(y, x)
                rows.append(
                    {
                        "system": str(z["environment_ids"][idx]),
                        "frequency": freq,
                        "method": "cubic_spline",
                        **complex_metrics(pred[~mask], field[~mask]),
                    }
                )
                predictions["cubic_spline"] = pred
                if idx == config["system_indices"][0]:
                    np.savez_compressed(
                        output / f"system{idx}_frequency{freq}.npz",
                        target=field,
                        fit_mask=mask,
                        x=x,
                        y=y,
                        **predictions,
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
