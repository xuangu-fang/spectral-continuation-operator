"""Observation-conditioned compact spectral codec, without full SSP access.

Known stationary background defines an analytic modal carrier. A shared FNO
predicts only a correction to cubic interpolation of observed modal envelopes.
This information protocol differs from the full-SSP reduced-solver benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import nn

from .audit_waveguide_spectral_transport import background_modal_carrier, encode_field
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


class SpectralConv1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.complex64) / width)

    def forward(self, x):
        spectrum = torch.fft.rfft(x)
        out = torch.zeros_like(spectrum)
        keep = min(self.modes, spectrum.shape[-1])
        out[..., :keep] = torch.einsum(
            "bik,iok->bok", spectrum[..., :keep], self.weight[..., :keep]
        )
        return torch.fft.irfft(out, n=x.shape[-1])


class EnvelopeOperator(nn.Module):
    def __init__(self, channels, outputs=64, width=64, modes=16, layers=4):
        super().__init__()
        self.lift = nn.Conv1d(channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, outputs, 1)
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, features, baseline):
        count = features.shape[-1]
        hidden = nn.functional.pad(self.lift(features), (0, 16))
        for spectral, local in zip(self.spectral, self.local, strict=True):
            hidden = nn.functional.gelu(spectral(hidden) + local(hidden))
        correction = self.readout(hidden[..., :count]).transpose(1, 2)
        return baseline + torch.complex(correction[..., :32], correction[..., 32:])


def prepare(config, systems):
    features, baselines, labels, meta, evaluation = [], [], [], [], []
    for system in systems:
        path = Path(config["data"]) / f"system{system}_m32_r2049.npz"
        with np.load(path) as z:
            fields, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
            spec = json.loads(str(z["spec"]))
        observed = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        take = np.linspace(0, len(ranges) - 1, config["range_samples"], dtype=int)
        coarse_ranges = ranges[take]
        context = []
        for oi in observed:
            kr, vectors = background_modal_carrier(spec, frequencies[oi], 32, "boundary_derivative")
            # Only observed-frequency fields enter features.
            coefficients = encode_field(fields[oi][:, take], depths, spec["depth_m"], 32)
            context.append(
                (coefficients @ vectors) * np.exp(-1j * coarse_ranges[:, None] * kr[None])
            )
        context = np.stack(context)
        scale = max(float(np.sqrt(np.mean(abs(context) ** 2))), 1e-20)
        context /= scale
        interpolator = CubicSpline(frequencies[observed], context, axis=0)
        # Complex values retain their common source phase; no independent branch alignment.
        packed = (
            np.concatenate([context.real, context.imag], axis=-1)
            .transpose(1, 0, 2)
            .reshape(len(take), -1)
        )
        readout = depth_basis(depths, spec["depth_m"], 32)
        roi = ranges >= config["roi_min_range_m"]
        for frequency in config["query_frequencies"]:
            if frequency in config["observed_frequencies"]:
                raise ValueError("observed query frequency")
            fi = int(np.flatnonzero(frequencies == frequency)[0])
            kr, vectors = background_modal_carrier(spec, frequency, 32, "boundary_derivative")
            # Known background dispersion only; range-dependent contrast never enters features.
            constant = np.r_[frequency / 100, kr / (2 * np.pi * 100 / spec["reference_speed_m_s"])]
            x = np.concatenate(
                [
                    packed,
                    (coarse_ranges / spec["range_m"])[:, None],
                    np.tile(constant, (len(take), 1)),
                ],
                axis=-1,
            )
            features.append(x.T)
            baselines.append(interpolator(frequency))
            target_modes = encode_field(fields[fi][:, take], depths, spec["depth_m"], 32)
            labels.append(
                (target_modes @ vectors) * np.exp(-1j * coarse_ranges[:, None] * kr[None]) / scale
            )
            meta.append({"system": system, "frequency": frequency, "scale": scale})
            evaluation.append(
                {
                    "target": fields[fi][:, roi].copy(),
                    "ranges": ranges,
                    "coarse_ranges": coarse_ranges,
                    "roi": roi,
                    "readout": readout @ vectors,
                    "kr": kr,
                    "scale": scale,
                }
            )
    arrays = {
        "features": np.asarray(features, dtype=np.float32),
        "baseline": np.asarray(baselines, dtype=np.complex64),
        "target": np.asarray(labels, dtype=np.complex64),
    }
    return arrays, meta, evaluation


def evaluate_envelopes(envelopes, meta, evaluation):
    rows = []
    for envelope, info, item in zip(envelopes, meta, evaluation, strict=True):
        full = CubicSpline(item["coarse_ranges"], envelope, axis=0)(item["ranges"])
        coefficients = (
            full * np.exp(1j * item["ranges"][:, None] * item["kr"][None]) * item["scale"]
        )
        field = item["readout"] @ coefficients.T
        rows.append({**info, **complex_metrics(field[:, item["roi"]], item["target"])})
    return rows


def device_pool(arrays):
    return {k: torch.as_tensor(v, device="cuda") for k, v in arrays.items()}


@torch.no_grad()
def predict(model, pool, batch=32):
    return np.concatenate(
        [
            model(pool["features"][i : i + batch], pool["baseline"][i : i + batch]).cpu().numpy()
            for i in range(0, len(pool["features"]), batch)
        ]
    )


def kernel_controls(train, val, config):
    """Train-only joint context/frequency RBF regression of the same residual.

    Hyperparameters are selected on development, exactly as the neural model.
    The same background/frequency information is present in both feature arrays.
    """
    x, q = (
        train["features"].reshape(len(train["features"]), -1),
        val["features"].reshape(len(val["features"]), -1),
    )
    # Normalize distances by training feature variance, not query labels.
    centered = x - x.mean(0)
    scale = max(float(np.mean(np.sum(centered**2, axis=-1))), 1e-12)

    def distance(a, b):
        return np.maximum(
            ((a * a).sum(-1)[:, None] + (b * b).sum(-1)[None] - 2 * a @ b.T) / scale, 0
        )

    dxx, dqx = (
        distance(x.astype(float), x.astype(float)),
        distance(q.astype(float), x.astype(float)),
    )
    residual = (train["target"] - train["baseline"]).reshape(len(x), -1)
    for bandwidth in config["bandwidths"]:
        kernel = np.exp(-dxx / bandwidth**2)
        cross = np.exp(-dqx / bandwidth**2)
        for ridge in config["ridges"]:
            coefficient = np.linalg.solve(kernel + ridge * np.eye(len(x)), residual)
            correction = (cross @ coefficient).reshape(val["baseline"].shape)
            yield {"bandwidth": bandwidth, "ridge": ridge}, val["baseline"] + correction


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    if set(config["train_systems"]) & set(config["validation_systems"]):
        raise ValueError("system leakage")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    start = time.perf_counter()
    train, train_meta, train_eval = prepare(config, config["train_systems"])
    val, val_meta, val_eval = prepare(config, config["validation_systems"])
    print(
        json.dumps(
            {
                "data_ready_seconds": time.perf_counter() - start,
                "train_queries": len(train_meta),
                "validation_queries": len(val_meta),
            }
        ),
        flush=True,
    )
    baseline = evaluate_envelopes(val["baseline"], val_meta, val_eval)
    (output / "cubic_baseline.json").write_text(json.dumps(baseline, indent=2))
    best_kernel, kernel_history = float("inf"), []
    for options, envelope in kernel_controls(train, val, config["kernel_control"]):
        rows = evaluate_envelopes(envelope, val_meta, val_eval)
        score = float(np.mean([r["complex_rel_l2"] for r in rows]))
        kernel_history.append({**options, "macro_error": score})
        if score < best_kernel:
            best_kernel = score
            (output / "best_kernel.json").write_text(
                json.dumps({"options": options, "results": rows}, indent=2)
            )
    (output / "kernel_selection.json").write_text(json.dumps(kernel_history, indent=2))
    print(
        json.dumps(
            {
                "baseline": float(np.mean([r["complex_rel_l2"] for r in baseline])),
                "best_kernel": best_kernel,
            }
        ),
        flush=True,
    )
    train_gpu, val_gpu = device_pool(train), device_pool(val)
    model = EnvelopeOperator(train["features"].shape[1], **config["model"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"], eta_min=1e-5)
    best, history = float("inf"), []
    training_start = time.perf_counter()
    for step in range(config["steps"] + 1):
        if step % config["eval_every"] == 0:
            rows = evaluate_envelopes(predict(model, val_gpu), val_meta, val_eval)
            score = float(np.mean([r["complex_rel_l2"] for r in rows]))
            history.append(
                {
                    "step": step,
                    "macro_error": score,
                    "seconds": time.perf_counter() - training_start,
                }
            )
            print(json.dumps(history[-1]), flush=True)
            (output / "history.json").write_text(json.dumps(history, indent=2))
            if score < best:
                best = score
                torch.save(
                    {"model": model.state_dict(), "config": config, "step": step},
                    output / "best.pt",
                )
                (output / "best_validation.json").write_text(json.dumps(rows, indent=2))
        if step == config["steps"]:
            break
        ids = torch.randint(len(train["features"]), (config["batch_size"],), device="cuda")
        prediction = model(train_gpu["features"][ids], train_gpu["baseline"][ids])
        target = train_gpu["target"][ids]
        loss = (
            (prediction - target).abs().square().mean((1, 2))
            / target.abs().square().mean((1, 2)).clamp_min(1e-12)
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    torch.save(
        {"model": model.state_dict(), "config": config, "step": config["steps"]}, output / "last.pt"
    )
    last_train_rows = evaluate_envelopes(predict(model, train_gpu), train_meta, train_eval)
    (output / "last_training.json").write_text(json.dumps(last_train_rows, indent=2))
    state = torch.load(output / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"])
    train_rows = evaluate_envelopes(predict(model, train_gpu), train_meta, train_eval)
    (output / "best_training.json").write_text(json.dumps(train_rows, indent=2))
    summary = {
        "macro_error": best,
        "training_macro_error": float(np.mean([r["complex_rel_l2"] for r in train_rows])),
        "last_training_macro_error": float(np.mean([r["complex_rel_l2"] for r in last_train_rows])),
        "kernel_macro_error": best_kernel,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - start,
        "information": "3 complete observed fields + stationary background; range-dependent SSP is NOT an input",
        "scope": "development shared spectral codec; not a known-full-PDE solver, formal test, or distant-frequency extrapolator",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
