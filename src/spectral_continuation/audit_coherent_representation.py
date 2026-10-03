"""R0 fitting audit with disjoint fitted/queried coordinates on existing data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline, RectBivariateSpline

from .coherent_representation import (
    ComplexINR,
    LocalWaveCharts,
    build_patches,
    complex_metrics,
    regular_fit_mask,
)


def load_cases(config: dict):
    for domain, cfg in config["data"].items():
        if domain == "helmholtz":
            for idx in cfg["system_indices"]:
                path = Path(cfg["path"]) / f"system_{idx}_grid{cfg['grid']}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    axis = archive["axis"]
                    keep = (axis >= cfg["roi"][0]) & (axis <= cfg["roi"][1])
                    ids = np.flatnonzero(keep)
                    for freq in cfg["frequencies"]:
                        fi = int(np.flatnonzero(np.isclose(archive["frequencies"], freq))[0])
                        yield (
                            domain,
                            str(idx),
                            freq,
                            axis[keep],
                            axis[keep],
                            archive["field"][fi][np.ix_(ids, ids)],
                            1.0,
                        )
            continue
        with np.load(cfg["path"], allow_pickle=False) as archive:
            if domain == "ocean":
                f = archive["frequencies_hz"]
                x, y = archive["ranges_m"], archive["depths_m"]
                # This archive contains development only, no strict field bytes.
                fields = archive["complex_field"]
                names = list(archive["environment_parameter_names"])
                for idx in cfg["system_indices"]:
                    if archive["split"][idx] not in ("train", "val"):
                        raise ValueError("R0 must not read sealed test systems")
                    c = float(
                        archive["environment_parameters"][idx, names.index("water_speed_mps")]
                    )
                    for freq in cfg["frequencies"]:
                        fi = int(np.flatnonzero(np.isclose(f, freq))[0])
                        raw = fields[idx, fi]
                        field = raw[..., 0] + 1j * raw[..., 1]
                        yield domain, str(archive["environment_ids"][idx]), freq, x, y, field, c
            else:
                fields = archive["field"]
                axis = np.arange(fields.shape[-1]) * float(archive["spacing"])
                for idx in cfg["system_indices"]:
                    if archive["split"][idx] == "test":
                        raise ValueError("R0 must not open historical test systems")
                    for fi in cfg["frequency_indices"]:
                        freq = float(archive["frequencies"][fi])
                        yield (
                            domain,
                            str(idx),
                            freq,
                            axis,
                            axis,
                            fields[idx, cfg.get("source_index", 0), fi],
                            1.0,
                        )


def run(config: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = config.get("device", "cuda")
    rows = []
    (output / "config.json").write_text(json.dumps(config, indent=2))
    for domain, system, freq, x, y, field, speed in load_cases(config):
        case_id = f"{domain}_{system}_{freq:.3f}"
        xx, yy = np.meshgrid(x, y, indexing="xy")
        coords = np.stack((xx, yy), -1)
        fit = regular_fit_mask(field.shape, config["spatial_stride"])
        scale = np.sqrt(np.mean(np.abs(field[fit]) ** 2))
        target = field / scale
        predictions = {}

        def save(
            name: str,
            pred: np.ndarray,
            elapsed: float,
            scalars: int,
            train_loss=None,
            domain=domain,
            system=system,
            freq=freq,
            fit=fit,
            target=target,
            predictions=predictions,
        ):
            row = {
                "domain": domain,
                "system": system,
                "frequency": freq,
                "method": name,
                "fit_points": int(fit.sum()),
                "query_points": int((~fit).sum()),
                "stored_real_scalars": scalars,
                "seconds": elapsed,
                "train_rel_l2": complex_metrics(pred[fit], target[fit])["complex_rel_l2"],
                **complex_metrics(pred[~fit], target[~fit]),
            }
            if train_loss is not None:
                row["local_objective"] = train_loss
            rows.append(row)
            predictions[name] = pred.astype(np.complex64)
            print(json.dumps(row), flush=True)
            (output / "results.json").write_text(json.dumps(rows, indent=2))

        for order in (1, 3):
            start = time.perf_counter()
            iy, ix = np.where(fit.any(1))[0], np.where(fit.any(0))[0]
            samples = target[np.ix_(iy, ix)]
            pred = RectBivariateSpline(y[iy], x[ix], samples.real, kx=order, ky=order)(y, x)
            pred = pred + 1j * RectBivariateSpline(y[iy], x[ix], samples.imag, kx=order, ky=order)(
                y, x
            )
            save(f"spline_{order}", pred, time.perf_counter() - start, 2 * int(fit.sum()))

        for rank in config.get("tensor_ranks", []):
            start = time.perf_counter()
            iy, ix = np.where(fit.any(1))[0], np.where(fit.any(0))[0]
            samples = target[np.ix_(iy, ix)]
            u, singular, vh = np.linalg.svd(samples, full_matrices=False)
            rank = min(rank, len(singular))
            left, right = u[:, :rank] * singular[:rank], vh[:rank]
            left_query = CubicSpline(y[iy], left, axis=0)(y)
            right_query = CubicSpline(x[ix], right, axis=1)(x)
            pred = left_query @ right_query
            save(
                f"functional_svd_rank{rank}",
                pred,
                time.perf_counter() - start,
                2 * rank * (len(iy) + len(ix)),
            )

        for cfg in config["charts"]:
            start = time.perf_counter()
            torch.manual_seed(config["seed"])
            spacing = cfg["wavelength_spacing"] * speed / freq
            patches = build_patches(coords, target, fit, spacing, device)
            model = LocalWaveCharts(
                patches,
                2 * np.pi * freq / speed,
                cfg["branches"],
                degree=cfg.get("degree", 0),
                learnable=cfg.get("learnable", False),
                curvature=cfg.get("curvature", False),
                ridge=cfg.get("ridge", 1e-7),
            )
            initial = model.set_coefficients(patches)
            history = [{"step": 0, "loss": initial}]
            if cfg.get("learnable", False):
                optimizer = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], lr=cfg.get("lr", 0.02)
                )
                best, best_state = (
                    initial,
                    {k: v.detach().clone() for k, v in model.state_dict().items()},
                )
                for step in range(1, cfg["steps"] + 1):
                    optimizer.zero_grad()
                    loss, _ = model.project(patches)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite variable-projection loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if step % 25 == 0 or step == cfg["steps"]:
                        current = model.set_coefficients(patches)
                        history.append({"step": step, "loss": current})
                        if current < best:
                            best, best_state = (
                                current,
                                {k: v.detach().clone() for k, v in model.state_dict().items()},
                            )
                model.load_state_dict(best_state)
            final = model.set_coefficients(patches)
            pred = model.predict(coords)
            torch.cuda.synchronize() if device == "cuda" else None
            save(cfg["name"], pred, time.perf_counter() - start, model.stored_real_scalars(), final)
            torch.save(
                {"state": model.state_dict(), "config": cfg, "history": history},
                output / f"{case_id}_{cfg['name']}.pt",
            )
            del model

        for cfg in config.get("inrs", []):
            start = time.perf_counter()
            torch.manual_seed(config["seed"])
            scaled = (
                2 * (coords - coords.reshape(-1, 2).min(0)) / np.ptp(coords.reshape(-1, 2), axis=0)
                - 1
            )
            X = torch.as_tensor(scaled.reshape(-1, 2), device=device, dtype=torch.float32)
            Y = torch.as_tensor(target.ravel(), device=device, dtype=torch.complex64)
            fit_idx = torch.as_tensor(np.flatnonzero(fit.ravel()), device=device)
            model = ComplexINR(cfg["kind"], cfg["width"], bandwidth=cfg["bandwidth"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, cfg["steps"], eta_min=1e-5
            )
            for step in range(cfg["steps"]):
                ids = fit_idx[
                    torch.randperm(len(fit_idx), device=device)[: cfg.get("batch_size", 4096)]
                ]
                optimizer.zero_grad()
                loss = (model(X[ids]) - Y[ids]).abs().square().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
            with torch.no_grad():
                pred = (
                    torch.cat([model(a) for a in X.split(4096)]).cpu().numpy().reshape(field.shape)
                )
            torch.cuda.synchronize() if device == "cuda" else None
            save(
                cfg["name"],
                pred,
                time.perf_counter() - start,
                sum(p.numel() for p in model.parameters())
                + sum(b.numel() for b in model.buffers()),
            )
            torch.save(model.state_dict(), output / f"{case_id}_{cfg['name']}.pt")
        np.savez_compressed(
            output / f"{case_id}_predictions.npz",
            target=target,
            fit_mask=fit,
            x=x,
            y=y,
            **predictions,
        )
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (output / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "protocol": "R0 per-field fit; not operator generalization",
        "seed": config["seed"],
        "fit_coordinate_rule": "regular stride with domain endpoints",
        "data": config["data"],
        "normalization": "RMS of fitted points only",
        "selection": "chart train loss only; fixed INR training budget; never query loss",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
