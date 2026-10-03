"""Shared wavelength-normalized chart encoder with coherent analytic readout.

The encoder infers wavevectors from observed complex samples. Ridge variable
projection preserves their absolute phase. No query label is an encoder input.
This is a local representation-transfer POC, not a full system operator yet.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .coherent_representation import complex_metrics


class WaveChartEncoder(nn.Module):
    def __init__(
        self,
        branches=8,
        degree=1,
        width=128,
        extent=1.5,
        ridge=1e-5,
        learned=True,
        analyzer_count=32,
        mode="conditional",
    ):
        super().__init__()
        self.degree, self.ridge, self.learned = degree, ridge, learned
        self.mode = mode
        angle = torch.arange(branches) * 2 * math.pi / branches
        k = 2 * math.pi * extent * torch.stack((angle.cos(), angle.sin()), -1)
        self.register_buffer("base_k", k.double())
        angle = torch.arange(analyzer_count) * 2 * math.pi / analyzer_count
        analyzer = 2 * math.pi * extent * torch.stack((angle.cos(), angle.sin()), -1)
        self.register_buffer("analyzer", analyzer)
        self.encoder = nn.Sequential(
            nn.Linear(analyzer_count * 3 + 6, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 2 * branches),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        if mode == "global":
            self.encoder = nn.ParameterDict({"delta": nn.Parameter(torch.zeros(branches, 2))})
        elif mode != "conditional":
            raise ValueError(mode)

    def infer_k(self, x, y, weights):
        if not self.learned:
            return self.base_k[None].expand(len(x), -1, -1)
        if self.mode == "global":
            k = self.base_k + 6 * torch.tanh(self.encoder["delta"]).double()
            return k[None].expand(len(x), -1, -1)
        features = self.encode_features(x, y, weights)
        delta = 6 * torch.tanh(self.encoder(features)).reshape(len(x), -1, 2).double()
        return self.base_k[None] + delta

    def encode_features(self, x, y, weights):
        # Complex responses are reduced only by one shared phase reference.
        phase = torch.einsum("bnd,jd->bnj", x.float(), self.analyzer)
        projection = (
            torch.exp(-1j * phase) * y[..., None].to(torch.complex64) * weights[..., None].float()
        ).sum(1) / weights.sum(1)[:, None].float()
        strongest = projection.abs().argmax(-1)
        reference = projection.gather(1, strongest[:, None])
        projection = projection * reference.conj() / reference.abs().clamp_min(1e-8)
        # Dimensionless geometry moments support varying sample layouts.
        w = weights.float() / weights.sum(-1, keepdim=True).float()
        mean = (w[..., None] * x.float()).sum(1)
        square = (w[..., None] * x.float().square()).sum(1)
        cross = (w * x[..., 0].float() * x[..., 1].float()).sum(1, keepdim=True)
        count = (weights > 0).sum(-1, keepdim=True).float().clamp_min(1).log() / 5
        return torch.cat(
            (projection.real, projection.imag, projection.abs(), mean, square, cross, count), -1
        )

    def basis(self, x, k):
        phi = torch.einsum("bnd,bjd->bnj", x.double(), k)
        wave = torch.exp(1j * phi)
        if self.degree:
            poly = torch.cat((torch.ones_like(x[..., :1]), x), -1).double()
            wave = (wave[..., None] * poly[..., None, :]).flatten(-2)
        return wave

    def forward(self, x, y, query, weights=None, return_state=False):
        if weights is None:
            weights = torch.ones_like(y.real)
        k = self.infer_k(x, y, weights)
        phi = self.basis(x, k)
        matrix = phi * weights[..., None].sqrt()
        labels = y.to(torch.complex128) * weights.sqrt()
        gram = matrix.mH @ matrix
        diagonal = gram.diagonal(dim1=-2, dim2=-1).real.mean(-1)
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        coeff = torch.linalg.solve(
            gram + self.ridge * diagonal[:, None, None] * eye, matrix.mH @ labels[..., None]
        ).squeeze(-1)
        result = (self.basis(query, k) * coeff[:, None]).sum(-1)
        return (result, k, coeff) if return_state else result


def make_ocean_pool(config, system_indices, count_per_system, seed):
    """Only development fields. Query positions and systems are strictly recorded."""
    rng = np.random.default_rng(seed)
    with np.load(config["path"]) as z:
        raw = z["complex_field"]
        x, y = z["ranges_m"], z["depths_m"]
        all_f = z["frequencies_hz"]
        names = list(z["environment_parameter_names"])
        fit_count, query_count = config["fit_count"], config["query_count"]
        out = {k: [] for k in ("x", "y", "q", "target", "weight", "scale", "system", "frequency")}
        out.update({"center": [], "physical_radius": []})
        for idx in system_indices:
            if z["split"][idx] not in ("train", "val"):
                raise ValueError("sealed systems must not be opened")
            c = float(z["environment_parameters"][idx, names.index("water_speed_mps")])
            for _ in range(count_per_system):
                freq = float(rng.choice(config["frequencies"]))
                fi = int(np.flatnonzero(np.isclose(all_f, freq))[0])
                extent = config["extent"] * c / freq
                # Sample center over whole domain, including truncated edge charts.
                for attempt in range(100):
                    cx, cy = rng.uniform(x[0], x[-1]), rng.uniform(y[0], y[-1])
                    ix, iy = (
                        np.flatnonzero(abs(x - cx) < extent),
                        np.flatnonzero(abs(y - cy) < extent),
                    )
                    if len(ix) * len(iy) >= fit_count + query_count:
                        break
                else:
                    raise ValueError("insufficient physical sampling density")
                xx, yy = np.meshgrid(x[ix], y[iy])
                positions = np.stack(((xx - cx) / extent, (yy - cy) / extent), -1).reshape(-1, 2)
                field = raw[idx, fi][np.ix_(iy, ix)]
                values = (field[..., 0] + 1j * field[..., 1]).ravel()
                take = rng.choice(len(values), fit_count + query_count, replace=False)
                fit, query = take[:fit_count], take[fit_count:]
                scale = max(float(np.sqrt(np.mean(abs(values[fit]) ** 2))), 1e-20)
                out["x"].append(positions[fit])
                out["q"].append(positions[query])
                out["y"].append(values[fit] / scale)
                out["target"].append(values[query] / scale)
                out["weight"].append(np.prod(np.maximum(1 - abs(positions[fit]), 0.05), -1))
                out["scale"].append(scale)
                out["system"].append(idx)
                out["frequency"].append(freq)
                out["center"].append([cx, cy])
                out["physical_radius"].append(extent)
    return {k: np.asarray(v) for k, v in out.items()}


def make_helmholtz_pool(config, system_indices, count_per_system, seed):
    """Direct solver labels inside the prespecified non-sponge physical ROI."""
    rng = np.random.default_rng(seed)
    fit_count, query_count = config["fit_count"], config["query_count"]
    out = {k: [] for k in ("x", "y", "q", "target", "weight", "scale", "system", "frequency")}
    for idx in system_indices:
        path = Path(config["path"]) / f"system_{idx}_grid{config['grid']}.npz"
        with np.load(path) as z:
            axis, fields, all_f = z["axis"], z["field"], z["frequencies"]
        lo, hi = config["roi"]
        allowed = (axis >= lo) & (axis <= hi)
        for _ in range(count_per_system):
            freq = float(rng.choice(config["frequencies"]))
            fi = int(np.flatnonzero(np.isclose(all_f, freq))[0])
            extent = config["extent"] / freq
            for attempt in range(100):
                cx, cy = rng.uniform(lo, hi, 2)
                ix = np.flatnonzero((abs(axis - cx) < extent) & allowed)
                iy = np.flatnonzero((abs(axis - cy) < extent) & allowed)
                if len(ix) * len(iy) >= fit_count + query_count:
                    break
            else:
                raise ValueError("insufficient sampling density")
            xx, yy = np.meshgrid(axis[ix], axis[iy])
            positions = np.stack(((xx - cx) / extent, (yy - cy) / extent), -1).reshape(-1, 2)
            values = fields[fi][np.ix_(iy, ix)].ravel()
            take = rng.choice(len(values), fit_count + query_count, replace=False)
            fit, query = take[:fit_count], take[fit_count:]
            scale = max(float(np.sqrt(np.mean(abs(values[fit]) ** 2))), 1e-20)
            out["x"].append(positions[fit])
            out["q"].append(positions[query])
            out["y"].append(values[fit] / scale)
            out["target"].append(values[query] / scale)
            out["weight"].append(np.prod(np.maximum(1 - abs(positions[fit]), 0.05), -1))
            out["scale"].append(scale)
            out["system"].append(idx)
            out["frequency"].append(freq)
    return {k: np.asarray(v) for k, v in out.items()}


def on_device(pool, device):
    return {
        k: torch.tensor(
            v, device=device, dtype=torch.complex64 if np.iscomplexobj(v) else torch.float32
        )
        for k, v in pool.items()
        if k not in ("system", "frequency")
    }


@torch.no_grad()
def evaluate(model, pool, arrays, batch=128):
    predictions = []
    model.eval()
    for i in range(0, len(pool["x"]), batch):
        sl = slice(i, i + batch)
        p = model(pool["x"][sl], pool["y"][sl], pool["q"][sl], pool["weight"][sl])
        predictions.append(p.cpu().numpy())
    predictions = np.concatenate(predictions)
    target = arrays["target"]
    rows = []
    for system in np.unique(arrays["system"]):
        for freq in np.unique(arrays["frequency"]):
            mask = (arrays["system"] == system) & (arrays["frequency"] == freq)
            if not mask.any():
                continue
            scale = arrays["scale"][mask, None]
            rows.append(
                {
                    "system": int(system),
                    "frequency": float(freq),
                    **complex_metrics(predictions[mask] * scale, target[mask] * scale),
                }
            )
    model.train()
    return rows, predictions


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    start = time.perf_counter()
    data = config["data"]
    train_ids, val_ids = data["train_ids"], data["validation_ids"]
    if set(train_ids) & set(val_ids):
        raise ValueError("system leakage")
    train_config = {**data, "frequencies": data["train_frequencies"]}
    val_config = {**data, "frequencies": data["validation_frequencies"]}
    loader = make_helmholtz_pool if data.get("domain") == "helmholtz" else make_ocean_pool
    arrays_train = loader(train_config, train_ids, data["train_patches_per_system"], config["seed"])
    arrays_val = loader(val_config, val_ids, data["validation_patches_per_system"], 970031)
    np.savez_compressed(output / "development_query_pool.npz", **arrays_val)
    train, val = on_device(arrays_train, "cuda"), on_device(arrays_val, "cuda")
    print(
        f"pools ready train={len(train['x'])} val={len(val['x'])} seconds={time.perf_counter() - start:.1f}",
        flush=True,
    )
    baseline = {}
    for branches, degree in ((8, 0), (8, 1), (16, 0), (32, 0), (16, 1)):
        name = f"fixed_pw{branches}_degree{degree}"
        model = WaveChartEncoder(
            branches=branches,
            degree=degree,
            extent=data["extent"],
            learned=False,
            ridge=config["model"]["ridge"],
        ).cuda()
        rows, _ = evaluate(model, val, arrays_val)
        baseline[name] = rows
        print(name, np.mean([r["complex_rel_l2"] for r in rows]), flush=True)
    (output / "baselines.json").write_text(json.dumps(baseline, indent=2))
    model = build_chart_model(config["model"], data["extent"]).cuda()
    if config.get("initial_checkpoint"):
        initial = torch.load(config["initial_checkpoint"], map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(initial["model"], strict=False)
        if unexpected or set(missing) - {"step_logits"}:
            raise ValueError(f"initial checkpoint mismatch: {missing}, {unexpected}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"], eta_min=1e-5)
    best, history = float("inf"), []
    training_start = time.perf_counter()
    for step in range(config["steps"] + 1):
        if step % config["eval_every"] == 0:
            rows, _ = evaluate(model, val, arrays_val)
            score = float(np.mean([r["complex_rel_l2"] for r in rows]))
            entry = {
                "step": step,
                "validation_macro_rel_l2": score,
                "seconds": time.perf_counter() - training_start,
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
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
        ids = torch.randint(len(train["x"]), (config["batch_size"],), device="cuda")
        optimizer.zero_grad()
        pred = model(train["x"][ids], train["y"][ids], train["q"][ids], train["weight"][ids])
        error = (pred - train["target"][ids]).abs().square().mean(-1)
        denominator = train["target"][ids].abs().square().mean(-1).clamp_min(1e-8)
        loss = (error / denominator).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    torch.save({"model": model.state_dict(), "config": config}, output / "last.pt")
    best_state = torch.load(output / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(best_state["model"])
    train_rows, _ = evaluate(model, train, arrays_train)
    (output / "best_training.json").write_text(json.dumps(train_rows, indent=2))
    summary = {
        "best_validation_macro_rel_l2": best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_macro_rel_l2": float(np.mean([r["complex_rel_l2"] for r in train_rows])),
        "seconds": time.perf_counter() - start,
        "scope": "system-held-out local reconstruction development; not full-field or spectral query",
        "checkpoint_selection": "development validation only; no formal test has been generated",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def build_chart_model(config, extent):
    options = dict(config)
    family = options.pop("family", "plane_wave")
    if family == "kernel":
        from .learned_coherent_kernel import CoherentKernelEncoder

        return CoherentKernelEncoder(**options, extent=extent)
    if family == "unrolled_kernel":
        from .unrolled_coherent_kernel import UnrolledCoherentKernel

        return UnrolledCoherentKernel(**options, extent=extent)
    if family != "plane_wave":
        raise ValueError(family)
    return WaveChartEncoder(**options, extent=extent)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("conditional", "global"))
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--no-carrier", action="store_true")
    p.add_argument("--no-metric", action="store_true")
    p.add_argument("--curvature", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--encoder-kind", choices=["spectral_moments", "set_attention"])
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode:
        config["model"]["mode"] = args.mode
        config["ablation"] = "Only change conditional encoder to shared global dictionary"
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
        config["optimizer_audit"] = "Permit global dictionary a separately tuned learning rate"
    if args.no_carrier:
        config["model"]["carrier"] = False
        config["carrier_ablation"] = "Zero phase carrier; learned real envelope metric only"
    if args.no_metric:
        config["model"]["metric"] = False
        config["metric_ablation"] = "Fixed Gaussian metric; learned phase carrier only"
    if args.curvature:
        config["model"]["curvature"] = True
        config["curvature_hypothesis"] = (
            "R0 quadratic phase improved heterogeneous capacity; add only three curvature coordinates to the kernel, retaining the same data and objective"
        )
    if args.seed is not None:
        config["seed"] = args.seed
    if args.encoder_kind:
        config["model"]["encoder_kind"] = args.encoder_kind
        config["encoder_hypothesis"] = (
            "Target-truth geometry fit ~0.00128 vs moment encoder ~0.01196 and context ML ~0.00725: test whether keeping full sample-set geometry helps amortized inference. Only encoder changes; same kernel, supervision and splits."
        )
    run(config, args.output)


if __name__ == "__main__":
    main()
