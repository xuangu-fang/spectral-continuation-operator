"""A minimal shared phase-coordinate learner with a physics-only ROM readout.

Known medium/source/PDE and complete context-frequency fields are inputs. The
target field is used for training supervision, never for the inference solve.
This is an exploratory learned representation, not a claim to invent PPRKS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit_fundamental_solution_rom import gaussian_reference
from .audit_phase_reduced_operator import discrete_operator, travel_coordinates
from .coherent_representation import complex_metrics


def phase_features(spec, axis, coordinates, source_regularized=False):
    xx, yy = np.meshgrid(axis[1:-1], axis[1:-1])
    sx, sy = spec["source"]
    dx, dy = xx - sx, yy - sy
    radius = np.hypot(dx, dy)
    speed = np.ones_like(xx)
    gx, gy = np.zeros_like(xx), np.zeros_like(xx)
    for center, width, contrast in zip(
        spec["centers"], spec["widths"], spec["contrasts"], strict=True
    ):
        qx, qy = xx - center[0], yy - center[1]
        bump = contrast * np.exp(-(qx * qx + qy * qy) / (2 * width * width))
        speed -= bump
        gx += bump * qx / width**2
        gy += bump * qy / width**2
    excess = (coordinates["straight_ray_medium_phase"] - coordinates["reference_phase"]).reshape(
        radius.shape
    )
    safe_r = np.maximum(radius, 1e-8)
    path_excess = np.where(radius > 1e-8, excess / safe_r, 1 / speed - 1)
    feature_r = np.sqrt(radius**2 + spec["source_width"] ** 2) if source_regularized else radius
    direction_r = feature_r if source_regularized else safe_r
    features = np.stack(
        [
            feature_r,
            dx / direction_r,
            dy / direction_r,
            10 * path_excess,
            1 / speed - 1,
            gx * feature_r,
            gy * feature_r,
            xx,
            yy,
        ],
        axis=-1,
    )
    return features.reshape(-1, features.shape[-1])


class PhaseCoordinate(nn.Module):
    def __init__(self, kind="neural", width=32):
        super().__init__()
        self.kind = kind
        if kind == "scalar":
            self.raw = nn.Parameter(torch.zeros(()))
        elif kind == "neural":
            self.network = nn.Sequential(
                nn.Linear(9, width),
                nn.Tanh(),
                nn.Linear(width, width),
                nn.Tanh(),
                nn.Linear(width, 1),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        else:
            raise ValueError(kind)

    def forward(self, features, reference, straight):
        raw = self.raw if self.kind == "scalar" else self.network(features).squeeze(-1)
        alpha = 1 + 0.75 * torch.tanh(raw)
        return reference + alpha.to(reference.dtype) * (straight - reference)


class EnvelopeCoordinate(nn.Module):
    """Learn smooth amplitude factors; retain the known straight-ray phase."""

    def __init__(self, kind="envelope_neural", width=32):
        super().__init__()
        self.kind = kind
        if kind in ("envelope_scalar", "envelope_complex_global"):
            self.raw = nn.Parameter(torch.zeros(4 if kind == "envelope_complex_global" else 2))
        elif kind in ("envelope_neural", "envelope_complex_neural"):
            self.network = nn.Sequential(
                nn.Linear(9, width),
                nn.Tanh(),
                nn.Linear(width, width),
                nn.Tanh(),
                nn.Linear(width, 4 if kind == "envelope_complex_neural" else 2),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        else:
            raise ValueError(kind)

    @staticmethod
    def fixed_factors(item):
        radius = item["coordinates"]["reference_phase"]
        first = (
            torch.sqrt(radius**2 + item["source_width"] ** 2)
            if item.get("source_regularized", False)
            else radius
        )
        return torch.stack(
            (
                torch.ones_like(radius),
                first,
                item["source_width"] / torch.sqrt(radius**2 + item["source_width"] ** 2),
            ),
            dim=1,
        )

    def factors(self, item, frequency=None):
        fixed = self.fixed_factors(item)
        if self.kind in ("envelope_scalar", "envelope_complex_global"):
            powers = 1 + 0.5 * torch.tanh(self.raw[:2])
            tail = fixed[:, 1:].clamp_min(1e-12) ** powers[None]
            if self.kind == "envelope_complex_global":
                phase = fixed[:, 1, None] * np.pi * torch.tanh(self.raw[2:])[None]
        else:
            raw = self.network(item["features"])
            modulation = torch.exp(0.75 * torch.tanh(raw[:, :2]))
            tail = fixed[:, 1:] * modulation
            if self.kind == "envelope_complex_neural":
                phase = np.pi * torch.tanh(raw[:, 2:])
        if "complex" in self.kind:
            reference_frequency = item["observed_f"].mean()
            ratio = 1.0 if frequency is None else frequency / reference_frequency
            tail = tail * torch.exp(1j * ratio * phase)
            return torch.cat((fixed[:, :1].to(tail.dtype), tail), dim=1)
        return torch.cat((fixed[:, :1], tail), dim=1)


def build_representation(config):
    cls = EnvelopeCoordinate if config["kind"].startswith("envelope_") else PhaseCoordinate
    return cls(config["kind"], config.get("width", 32))


def torch_projection(lap, mass, rhs, basis, frequency, ridge=1e-12):
    # Column scaling avoids source amplitude determining the numerical ridge.
    basis = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True).clamp_min(1e-30)
    av = torch.sparse.mm(lap, basis) + (2 * np.pi * frequency) ** 2 * mass[:, None] * basis
    gram = av.mH @ av
    scale = gram.diag().real.mean()
    coefficients = torch.linalg.solve(
        gram + ridge * scale * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device),
        av.mH @ rhs,
    )
    return basis @ coefficients


def predict(item, tau, frequency, config, factors=None, carrier_family="phase"):
    if carrier_family == "phase":
        basis = item["snapshots"] * torch.exp(
            2j * np.pi * tau[:, None] * (frequency - item["observed_f"][None])
        )
    elif carrier_family == "source_aware":
        basis = (
            item["snapshots"]
            * item["reference_carriers"][float(frequency)][:, None]
            / item["observed_reference_carriers"]
        )
    else:
        raise ValueError(carrier_family)
    if config.get("conjugate_enrichment", True):
        basis = torch.cat((basis, basis.conj()), dim=1)
    if factors is not None:
        basis = (basis[:, :, None] * factors[:, None]).reshape(basis.shape[0], -1)
    return torch_projection(
        item["lap"], item["mass"], item["rhs"], basis, frequency, config.get("ridge", 1e-12)
    )


def load_system(config, system, device):
    with np.load(Path(config["data"]) / f"system_{system}_grid257.npz") as z:
        fields, all_f, axis = z["field"], z["frequencies"], z["axis"]
        spec = json.loads(str(z["spec"]))
    observed = np.flatnonzero(np.isin(all_f, config["observed_frequencies"]))
    query = np.flatnonzero(np.isin(all_f, config["query_frequencies"]))
    if set(observed).intersection(query):
        raise ValueError("target-frequency observations are forbidden")
    lap, mass, rhs = discrete_operator(spec, len(axis))
    coo = lap.tocoo()
    sparse_lap = torch.sparse_coo_tensor(
        np.stack((coo.row, coo.col)),
        coo.data.astype(np.complex128),
        coo.shape,
        device=device,
        dtype=torch.complex128,
    ).coalesce()
    coords = travel_coordinates(spec, axis, config.get("include_eikonal", True))
    if config.get("source_regularized", False):
        radius = coords["reference_phase"]
        coords["source_regularized_phase"] = (
            coords["straight_ray_medium_phase"]
            * radius
            / np.sqrt(radius**2 + spec["source_width"] ** 2)
        )
    roi1 = (axis[1:-1] >= config["roi"][0]) & (axis[1:-1] <= config["roi"][1])
    item = {
        "system": system,
        "features": torch.tensor(
            phase_features(spec, axis, coords, config.get("source_regularized", False)),
            dtype=getattr(torch, config.get("feature_dtype", "float32")),
            device=device,
        ),
        "source_width": spec["source_width"],
        "source_regularized": config.get("source_regularized", False),
        "coordinates": {k: torch.tensor(v, device=device) for k, v in coords.items()},
        "lap": sparse_lap,
        "mass": torch.tensor(mass, device=device),
        "rhs": torch.tensor(rhs, device=device),
        "snapshots": torch.tensor(
            fields[observed, 1:-1, 1:-1].reshape(len(observed), -1).T.copy(),
            dtype=torch.complex128,
            device=device,
        ),
        "observed_f": torch.tensor(all_f[observed].astype(float), device=device),
        "targets": torch.tensor(fields[query, 1:-1, 1:-1].reshape(len(query), -1), device=device),
        "query_f": all_f[query].astype(float).tolist(),
        "roi": torch.tensor(np.outer(roi1, roi1).ravel(), device=device),
    }
    if config.get("carrier_family", "phase") == "source_aware":
        if not config["kind"].startswith("envelope_"):
            raise ValueError("source-aware branch currently learns envelope only")
        radius = coords["reference_phase"]
        excess = coords["straight_ray_medium_phase"] - radius
        if config.get("source_regularized", False):
            excess = excess * radius / np.sqrt(radius**2 + spec["source_width"] ** 2)
        item["reference_carriers"] = {
            float(f): torch.tensor(
                gaussian_reference(radius, f, spec["source_width"], loss=spec["background_loss"])
                * np.exp(2j * np.pi * f * excess),
                device=device,
            )
            for f in all_f
        }
        item["observed_reference_carriers"] = torch.stack(
            [item["reference_carriers"][float(f)] for f in all_f[observed]], dim=1
        )
    return item


def learned_tau(model, item):
    if isinstance(model, EnvelopeCoordinate):
        key = (
            "source_regularized_phase"
            if item.get("source_regularized", False)
            else "straight_ray_medium_phase"
        )
        return item["coordinates"][key]
    return model(
        item["features"],
        item["coordinates"]["reference_phase"],
        item["coordinates"]["straight_ray_medium_phase"],
    )


@torch.no_grad()
def evaluate(model, dataset, config, include_baselines=False):
    model.eval()
    rows = []
    for item in dataset:
        envelope = isinstance(model, EnvelopeCoordinate)
        coordinates = {
            f"learned_{model.kind}": (
                learned_tau(model, item),
                model.factors(item) if envelope else None,
                config.get("carrier_family", "phase"),
            )
        }
        if include_baselines:
            suffix = "_radial" if envelope else ""
            coordinates.update(
                {
                    k + suffix: (v, model.fixed_factors(item) if envelope else None, "phase")
                    for k, v in item["coordinates"].items()
                }
            )
            if config.get("carrier_family", "phase") == "source_aware":
                coordinates["source_aware_fixed_radial"] = (
                    learned_tau(model, item),
                    model.fixed_factors(item),
                    "source_aware",
                )
                coordinates["source_aware_base6"] = (learned_tau(model, item), None, "source_aware")
        for name, (tau, factors, carrier_family) in coordinates.items():
            for qi, frequency in enumerate(item["query_f"]):
                if name.startswith("learned_") and envelope:
                    factors = model.factors(item, frequency)
                prediction = predict(item, tau, frequency, config, factors, carrier_family)
                row = {
                    "system": item["system"],
                    "query_frequency": frequency,
                    "method": name,
                    "target_frequency_observations": 0,
                    "carrier_family": carrier_family,
                    "basis_columns": len(item["observed_f"])
                    * (2 if config.get("conjugate_enrichment", True) else 1)
                    * (factors.shape[-1] if factors is not None else 1),
                    **complex_metrics(
                        prediction[item["roi"]].cpu().numpy(),
                        item["targets"][qi, item["roi"]].cpu().numpy(),
                    ),
                }
                rows.append(row)
    model.train()
    return rows


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(config.get("cpu_threads", 2))
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    device = torch.device(config.get("device", "cuda"))
    started = time.perf_counter()
    train = [load_system(config, s, device) for s in config["train_systems"]]
    val = [load_system(config, s, device) for s in config["validation_systems"]]
    if set(config["train_systems"]) & set(config["validation_systems"]):
        raise ValueError("systems must be disjoint")
    model = build_representation(config).to(
        device=device, dtype=getattr(torch, config.get("feature_dtype", "float32"))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    initial = evaluate(model, val, config, include_baselines=True)
    (output / "initial_validation.json").write_text(json.dumps(initial, indent=2))
    best = float(
        np.mean([r["complex_rel_l2"] for r in initial if r["method"].startswith("learned_")])
    )
    torch.save({"model": model.state_dict(), "config": config, "step": 0}, output / "best.pt")
    history = []
    for step in range(1, config["steps"] + 1):
        item = train[int(rng.integers(len(train)))]
        qi = int(rng.integers(len(item["query_f"])))
        tau = learned_tau(model, item)
        factors = (
            model.factors(item, item["query_f"][qi])
            if isinstance(model, EnvelopeCoordinate)
            else None
        )
        prediction = predict(
            item, tau, item["query_f"][qi], config, factors, config.get("carrier_family", "phase")
        )
        error = prediction[item["roi"]] - item["targets"][qi, item["roi"]]
        loss = error.abs().square().sum() / item["targets"][qi, item["roi"]].abs().square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite complex loss")
        optimizer.zero_grad()
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % config["evaluate_every"] == 0 or step == 1:
            rows = evaluate(model, val, config)
            score = float(np.mean([r["complex_rel_l2"] for r in rows]))
            record = {
                "step": step,
                "train_single_rel_l2": float(loss.detach().sqrt()),
                "validation_macro_rel_l2": score,
                "gradient_norm": float(gradient),
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
    final_training = evaluate(model, train, config)
    (output / "last_training.json").write_text(json.dumps(final_training, indent=2))
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    rows = evaluate(model, val, config, include_baselines=True)
    training_rows = evaluate(model, train, config)
    (output / "validation.json").write_text(json.dumps(rows, indent=2))
    (output / "training.json").write_text(json.dumps(training_rows, indent=2))
    summary = {
        "best_step": checkpoint["step"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds_including_data_and_baselines": time.perf_counter() - started,
        "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else 0,
        "validation_macro_by_method": {
            method: float(np.mean([r["complex_rel_l2"] for r in rows if r["method"] == method]))
            for method in sorted({r["method"] for r in rows})
        },
        "training_macro_rel_l2": float(np.mean([r["complex_rel_l2"] for r in training_rows])),
        "last_training_macro_rel_l2": float(np.mean([r["complex_rel_l2"] for r in final_training])),
        "evidence": "development; known PDE; learned phase OR envelope representation, physics coefficients; not formal test",
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
