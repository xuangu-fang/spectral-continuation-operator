"""R1: do spatially accurate wave charts also support unobserved frequencies?

Read ONLY observed-frequency fields when encoding. Query truths are used only
after synthesis for metrics. Contrast independent frames with context-shared
frames and fixed physical frames; all comparisons are development diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from .coherent_representation import complex_metrics
from .evaluate_coherent_fine_ocean import load_model
from .learned_wave_chart import WaveChartEncoder


@torch.no_grad()
def spectral_query(
    model,
    context,
    context_f,
    target_f,
    x,
    y,
    speed,
    shared_frame,
    reference_frequency=125.0,
    spreading=True,
):
    """No target field argument exists in this synthesizer."""
    radius = 1.5 * speed / reference_frequency
    lo, hi = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    counts = np.maximum(np.ceil((hi - lo) / radius).astype(int), 1)
    spacing = (hi - lo) / counts
    axes = [np.linspace(lo[d], hi[d], counts[d] + 1) for d in range(2)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    xx, yy = np.meshgrid(x, y)
    coordinates = np.stack((xx, yy), -1).reshape(-1, 2)
    random = np.random.default_rng(9206)
    physical_x, indices, query_ids, query_weights = [], [], [], []
    for center in centers:
        delta = coordinates - center
        w = np.prod(np.maximum(1 - abs(delta / spacing), 0), -1)
        valid = np.flatnonzero(w > 1e-10)
        ids = random.choice(valid, min(64, len(valid)), replace=False)
        physical_x.append(delta[ids])
        indices.append(ids)
        query_ids.append(valid)
        query_weights.append(w[valid])
    size = max(map(len, indices))
    offsets = np.zeros((len(indices), size, 2))
    observed = np.zeros((len(context_f), len(indices), size), dtype=np.complex128)
    valid_weights = np.zeros((len(indices), size))
    for p, ids in enumerate(indices):
        offsets[p, : len(ids)] = physical_x[p]
        observed[:, p, : len(ids)] = context.reshape(len(context_f), -1)[:, ids]
        if spreading:
            observed[:, p, : len(ids)] *= np.sqrt(coordinates[ids, 0])[None]
        valid_weights[p, : len(ids)] = 1
    device = model.base_k.device
    states, coeffs, inputs = [], [], []
    for j, f in enumerate(context_f):
        normalized_x = offsets / (1.5 * speed / f)
        weight = np.prod(np.maximum(1 - abs(normalized_x), 0.05), -1) * valid_weights
        scale = np.sqrt((abs(observed[j]) ** 2 * valid_weights).sum(-1) / valid_weights.sum(-1))
        scale = np.maximum(scale, 1e-20)
        X = torch.tensor(normalized_x, device=device, dtype=torch.float32)
        Y = torch.tensor(observed[j] / scale[:, None], device=device, dtype=torch.complex64)
        W = torch.tensor(weight, device=device, dtype=torch.float32)
        _, k, a = model(X, Y, X[:, :1], W, return_state=True)
        states.append(k.cpu().numpy())
        coeffs.append(a.cpu().numpy() * scale[:, None])
        inputs.append((X, Y, W, scale))
    states = np.stack(states)
    if shared_frame:
        states[:] = states.mean(0, keepdims=True)
        coeffs = []
        for X, Y, W, scale in inputs:
            k = torch.tensor(states[0], device=device)
            matrix = model.basis(X, k) * W[..., None].sqrt()
            gram = matrix.mH @ matrix
            diag = gram.diagonal(dim1=-2, dim2=-1).real.mean(-1)
            eye = torch.eye(gram.shape[-1], device=device, dtype=gram.dtype)
            a = torch.linalg.solve(
                gram + model.ridge * diag[:, None, None] * eye,
                matrix.mH @ (Y.to(torch.complex128) * W.sqrt())[..., None],
            ).squeeze(-1)
            coeffs.append(a.cpu().numpy() * scale[:, None])
    coeffs = np.stack(coeffs)
    # Common reference travel phase; residual modal delays remain to be interpolated.
    center_phase = np.exp(2j * np.pi * np.asarray(context_f)[:, None] * centers[None, :, 0] / speed)
    gauged = coeffs / center_phase[..., None]
    new_k = CubicSpline(context_f, states, axis=0)(target_f)
    new_a = CubicSpline(context_f, gauged, axis=0)(target_f)
    new_a *= np.exp(2j * np.pi * np.asarray(target_f)[:, None] * centers[None, :, 0] / speed)[
        ..., None
    ]
    qsize = max(map(len, query_ids))
    qoffsets = np.zeros((len(indices), qsize, 2))
    for p, ids in enumerate(query_ids):
        qoffsets[p, : len(ids)] = coordinates[ids] - centers[p]
    predictions = []
    for j, f in enumerate(target_f):
        output = np.zeros(len(coordinates), dtype=np.complex128)
        weights = np.zeros(len(coordinates))
        Q = torch.tensor(qoffsets / (1.5 * speed / f), device=device, dtype=torch.float32)
        K = torch.tensor(new_k[j], device=device)
        A = torch.tensor(new_a[j], device=device)
        for start in range(0, len(indices), 16):
            sl = slice(start, start + 16)
            local = (model.basis(Q[sl], K[sl]) * A[sl, None]).sum(-1).cpu().numpy()
            for b in range(len(local)):
                p = start + b
                ids, w = query_ids[p], query_weights[p]
                output[ids] += local[b, : len(ids)] * w
                weights[ids] += w
        if weights.min() < 0.99:
            raise RuntimeError("query outside chart cover")
        output /= weights
        if spreading:
            output /= np.sqrt(coordinates[:, 0])
        predictions.append(output.reshape(len(y), len(x)))
    return np.stack(predictions)


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    conditional = load_model(config["conditional_checkpoint"])
    global_model = load_model(config["global_checkpoint"])
    fixed = WaveChartEncoder(branches=16, degree=1, learned=False, ridge=1e-5).cuda()
    angles = torch.linspace(-np.pi / 4, np.pi / 4, 17, device="cuda", dtype=torch.float64)[:-1]
    fixed.base_k.copy_(3 * np.pi * torch.stack((angles.cos(), angles.sin()), -1))
    models = [
        ("conditional_independent_frames", conditional, False),
        ("conditional_context_shared_frame", conditional, True),
        ("global_learned_frame", global_model, False),
        ("fixed_cone_frame", fixed, False),
    ]
    rows = []
    with np.load(config["ocean_path"]) as z:
        raw, f = z["complex_field"], z["frequencies_hz"]
        x, y = z["ranges_m"], z["depths_m"]
        names = list(z["environment_parameter_names"])
        for idx in config["system_indices"]:
            c = float(z["environment_parameters"][idx, names.index("water_speed_mps")])
            fields = raw[idx, ..., 0] + 1j * raw[idx, ..., 1]
            for step in config["frequency_spacings"]:
                observed_f = np.unique(np.r_[np.arange(90, 126, step), 125.0])
                fi = np.flatnonzero(np.isin(f, observed_f))
                qi = np.flatnonzero((f >= 90) & (f <= 125) & ~np.isin(f, observed_f))
                context = fields[fi]
                predictions = {}
                for name, model, shared in models:
                    predictions[name] = spectral_query(
                        model, context, f[fi], f[qi], x, y, c, shared
                    )
                carrier = np.exp(2j * np.pi * f[:, None, None] * x[None, None, :] / c)
                predictions["range_carrier_cubic"] = (
                    CubicSpline(f[fi], context / carrier[fi], axis=0)(f[qi]) * carrier[qi]
                )
                for name, prediction in predictions.items():
                    case_rows = [
                        {
                            "system": idx,
                            "frequency_spacing": step,
                            "query_frequency": float(f[q]),
                            "method": name,
                            **complex_metrics(prediction[j], fields[q]),
                        }
                        for j, q in enumerate(qi)
                    ]
                    rows.extend(case_rows)
                    print(
                        json.dumps(
                            {
                                "system": idx,
                                "step": step,
                                "method": name,
                                "error": np.mean([r["complex_rel_l2"] for r in case_rows]),
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
