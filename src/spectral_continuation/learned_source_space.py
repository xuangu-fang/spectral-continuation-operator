"""Learn six continuous source channels; retain a physics-only complex readout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def source_transform(raw, scales, bounded=True):
    batch, patches, doubled_branches = raw.shape
    branches = doubled_branches // 2
    raw = (0.75 * torch.tanh(raw) if bounded else raw).reshape(batch, patches, branches, 2)
    weights = torch.complex(raw[..., 0], raw[..., 1]).transpose(1, 2)
    weights = torch.cat((torch.ones_like(weights[:, :, :1]), weights), dim=-1)
    scales = scales.reshape(batch, branches, patches + 1)
    weights = weights * scales / scales[:, :, :1]
    identity = torch.eye(branches, dtype=weights.dtype, device=weights.device)
    return (weights[..., None] * identity[None, :, None, :]).reshape(batch, -1, branches)


class SourceSpaceEncoder(nn.Module):
    def __init__(self, kind="conditional", branches=6, patches=9, width=64):
        super().__init__()
        self.kind, self.branches, self.patches = kind, branches, patches
        if kind == "global":
            self.raw = nn.Parameter(torch.zeros(patches, branches * 2))
        elif kind == "conditional":
            self.encoder = nn.Sequential(
                nn.Linear(9, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh()
            )
            self.decoder = nn.Sequential(
                nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, branches * 2)
            )
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)
        else:
            raise ValueError(kind)

    def forward(self, features, scales):
        batch = features.shape[0]
        if self.kind == "global":
            raw = self.raw[None].expand(batch, -1, -1)
        else:
            local = self.encoder(features)
            pooled = local.mean(dim=1, keepdim=True).expand_as(local)
            raw = self.decoder(torch.cat((local, pooled), dim=-1))
        # The cache normalizes propagated columns. Undo their relative scaling so
        # these coefficients still describe physical source-window modulation.
        return source_transform(raw, scales)


def physics_coefficients(h, g, transform, ridge=1e-12):
    gram = transform.mH @ h @ transform
    scale = gram.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
    matrix = gram + ridge * scale[:, None, None] * torch.eye(
        gram.shape[-1], dtype=gram.dtype, device=gram.device
    )
    coefficients = torch.linalg.solve(matrix, (transform.mH @ g[..., None])).squeeze(-1)
    return (transform @ coefficients[..., None]).squeeze(-1)


def squared_error(item, coefficients):
    quadratic = (
        (coefficients.conj() * (item["Q"] @ coefficients[..., None]).squeeze(-1)).sum(-1).real
    )
    linear = (coefficients.conj() * item["t"]).sum(-1).real
    return (quadratic - 2 * linear + item["e"]).clamp_min(0) / item["target_norm2"]


def load_cache(config, systems, device):
    values = []
    metadata = []
    for system in systems:
        for frequency in config["query_frequencies"]:
            path = Path(config["cache"]) / f"system{system}_f{frequency}.npz"
            with np.load(path) as z:
                values.append(
                    {
                        key: np.asarray(z[key])
                        for key in ["H", "g", "Q", "t", "e", "target_norm2", "features", "scales"]
                    }
                )
                metadata.append(json.loads(str(z["metadata"])))
    return {
        key: torch.tensor(np.stack([v[key] for v in values]), device=device) for key in values[0]
    }, metadata


@torch.no_grad()
def evaluate(model, data, metadata, ridge):
    transform = model(data["features"], data["scales"])
    coefficients = physics_coefficients(data["H"], data["g"], transform, ridge)
    error = squared_error(data, coefficients).sqrt().cpu().numpy()
    return [
        {
            "system": record["system"],
            "frequency": record["frequency"],
            "complex_rel_l2": float(e),
            "baseline": record["scores"],
            "target_frequency_observations": 0,
        }
        for record, e in zip(metadata, error, strict=True)
    ]


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    if set(config["train_systems"]) & set(config["validation_systems"]):
        raise ValueError("train/development system overlap")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    device = torch.device(config.get("device", "cuda"))
    started = time.perf_counter()
    train, train_meta = load_cache(config, config["train_systems"], device)
    val, val_meta = load_cache(config, config["validation_systems"], device)
    model = SourceSpaceEncoder(config["kind"], width=config.get("width", 64)).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    initial = evaluate(model, val, val_meta, config["ridge"])
    (output / "initial_validation.json").write_text(json.dumps(initial, indent=2))
    best = float(np.mean([r["complex_rel_l2"] for r in initial]))
    torch.save({"model": model.state_dict(), "config": config, "step": 0}, output / "best.pt")
    history = []
    for step in range(1, config["steps"] + 1):
        index = torch.randint(len(train_meta), (config["batch_size"],), device=device)
        batch = {key: value[index] for key, value in train.items()}
        transform = model(batch["features"], batch["scales"])
        coefficients = physics_coefficients(batch["H"], batch["g"], transform, config["ridge"])
        loss = squared_error(batch, coefficients).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite source-space loss")
        optimizer.zero_grad()
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config["evaluate_every"] == 0:
            rows = evaluate(model, val, val_meta, config["ridge"])
            score = float(np.mean([r["complex_rel_l2"] for r in rows]))
            record = {
                "step": step,
                "train_batch_rms_error": float(loss.detach().sqrt()),
                "validation_macro_error": score,
                "gradient_norm": float(grad),
                "seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            (output / "history.json").write_text(json.dumps(history, indent=2))
            if score < best:
                best = score
                torch.save(
                    {"model": model.state_dict(), "config": config, "step": step},
                    output / "best.pt",
                )
    torch.save(
        {"model": model.state_dict(), "config": config, "step": config["steps"]}, output / "last.pt"
    )
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_rows = evaluate(model, val, val_meta, config["ridge"])
    train_rows = evaluate(model, train, train_meta, config["ridge"])
    (output / "validation.json").write_text(json.dumps(val_rows, indent=2))
    (output / "training.json").write_text(json.dumps(train_rows, indent=2))
    summary = {
        "best_step": checkpoint["step"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation_macro_error": float(np.mean([r["complex_rel_l2"] for r in val_rows])),
        "training_macro_error": float(np.mean([r["complex_rel_l2"] for r in train_rows])),
        "validation_baselines": {
            name: float(np.mean([r["baseline"][name] for r in val_rows]))
            for name in val_rows[0]["baseline"]
        },
        "training_seconds_excludes_explicitly_recorded_cache_preparation": time.perf_counter()
        - started,
        "evidence": "single-seed development; six source columns; physics coefficients; quadratic training loss exactly equals full-grid complex loss",
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
