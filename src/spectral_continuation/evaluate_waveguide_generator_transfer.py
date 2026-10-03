"""Frozen generator source/frequency/continuous-coordinate capability audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_generator_identification import latent_from_coefficients
from .coarse_waveguide_operator import query_field
from .coherent_representation import complex_metrics
from .learned_waveguide_generator import GeneratorEncoder, prepare


def frozen_coefficients(config):
    directory = Path(config["coefficient_run"])
    setting = json.loads((directory / "config.json").read_text())
    val, metadata = prepare(setting, setting["validation_systems"], False)
    with np.load(directory / "training_statistics.npz") as z:
        mean, std = z["mean"], z["std"]
        x = z["features"].reshape(len(z["features"]), -1).astype(float)
        residual = (z["target"] - z["baseline"]).reshape(len(x), -1)
    features = (val["features"] - mean) / std
    result = {"local_inverse": val["baseline"], "true_medium_oracle": val["target"]}
    stronger, _ = prepare(
        {**setting, "interpolation_order": 5, "range_penalty": 1},
        setting["validation_systems"],
        False,
    )
    result["regularized_inverse"] = stronger["baseline"]
    options = json.loads((directory / "best_kernel.json").read_text())["options"]
    q = features.reshape(len(features), -1).astype(float)
    scale = max(float(np.mean(np.sum((x - x.mean(0)) ** 2, axis=-1))), 1e-12)

    def distance(a, b):
        return np.maximum(
            ((a * a).sum(-1)[:, None] + (b * b).sum(-1)[None] - 2 * a @ b.T) / scale, 0
        )

    weight = np.linalg.solve(
        np.exp(-distance(x, x) / options["bandwidth"] ** 2) + options["ridge"] * np.eye(len(x)),
        residual,
    )
    result["kernel_generator"] = val["baseline"] + (
        np.exp(-distance(q, x) / options["bandwidth"] ** 2) @ weight
    ).reshape(val["baseline"].shape)
    for name, path in [
        ("coefficient_neural", directory / "best.pt"),
        ("field_neural", Path(config["field_run"]) / "best.pt"),
    ]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = GeneratorEncoder(features.shape[1], **setting["model"]).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            result[name] = (
                model(
                    torch.tensor(features, device="cuda"),
                    torch.tensor(val["baseline"], device="cuda"),
                )
                .cpu()
                .numpy()
            )
    for step in [10, 100]:
        with np.load(Path(config["context_fit_run"]) / f"step{step}_coefficients.npz") as z:
            result[f"nonlinear_context_{step}"] = z["beta"]
    return result, metadata, setting


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    candidates, metadata, setting = frozen_coefficients(config)
    np.savez_compressed(
        output / "frozen_coefficients.npz", **candidates, systems=setting["validation_systems"]
    )
    rows = []
    for si, item in enumerate(metadata):
        for variant in config["source_variants"]:
            with np.load(Path(config["data"]) / f"system{item['system']}_{variant}.npz") as z:
                spec = json.loads(str(z["spec"]))
                frequencies, ranges, depths, target = (
                    z["frequencies_hz"],
                    z["ranges_m"],
                    z["depths_m"],
                    z["field"],
                )
            geometry = item["geometry"]
            # Known requested source, not a target-field or hidden-medium readout.
            initial = geometry["basis"].T @ (
                geometry["weights"]
                * np.exp(
                    -0.5 * ((geometry["z"] - spec["source_depth_m"]) / spec["source_width_m"]) ** 2
                )
            )
            roi = ranges >= 50
            for method, coefficient in candidates.items():
                latent, clip = latent_from_coefficients(coefficient[si], initial, geometry)
                for fi, frequency in enumerate(frequencies):
                    prediction = query_field(latent, frequency, ranges, depths)
                    rows.append(
                        {
                            "system": item["system"],
                            "source_variant": variant,
                            "method": method,
                            "frequency": float(frequency),
                            "frequency_scope": "in_band"
                            if frequency <= 140
                            else "extrapolation_stress",
                            "clipping_fraction": clip,
                            "evidence": "paired development; encoder frozen, zero new-frequency or new-source field samples",
                            **complex_metrics(prediction[:, roi], target[fi][:, roi]),
                        }
                    )
            (output / "results.json").write_text(json.dumps(rows, indent=2))
            print(
                json.dumps({"system": item["system"], "source_variant": variant, "complete": True}),
                flush=True,
            )
    summary = []
    for method in candidates:
        for variant in config["source_variants"]:
            for scope in ["in_band", "extrapolation_stress"]:
                selected = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["source_variant"] == variant
                    and r["frequency_scope"] == scope
                ]
                summary.append(
                    {
                        "method": method,
                        "source_variant": variant,
                        "frequency_scope": scope,
                        "raw": float(np.mean([r["complex_rel_l2"] for r in selected])),
                        "phase_rmse_rad": float(
                            np.mean([r["energy_weighted_phase_rmse_rad"] for r in selected])
                        ),
                    }
                )
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
