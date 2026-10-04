"""Change decoder resolution, not encoder weights or real generator latent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_generator_identification import generator_geometry, latent_from_coefficients
from .coarse_waveguide_operator import query_field
from .coherent_representation import complex_metrics
from .learned_waveguide_generator import GeneratorEncoder, prepare


def run(config, output):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    with np.load(config["frozen_coefficients"]) as z:
        systems = z["systems"].tolist()
        codes = {method: z[method] for method in config["methods"]}
    for name, path in config.get("extra_neural_runs", {}).items():
        checkpoint = torch.load(Path(path) / "best.pt", map_location="cpu", weights_only=False)
        setting = checkpoint["config"]
        val, _ = prepare(setting, systems, False)
        features = (val["features"] - checkpoint["mean"]) / checkpoint["std"]
        model = GeneratorEncoder(features.shape[1], **setting["model"]).eval()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            codes[name] = model(torch.tensor(features), torch.tensor(val["baseline"])).numpy()
    for name, path in config.get("extra_coefficients", {}).items():
        with np.load(path) as z:
            beta = z["coefficients"] if "coefficients" in z else z["beta"]
            if "systems" in z and z["systems"].tolist() != systems:
                raise ValueError("extra coefficient system identities/order changed")
            if beta.shape != codes[config["methods"][0]].shape:
                raise ValueError("extra coefficients must follow the frozen system order/shape")
            codes[name] = beta
    aliases = config.get("method_aliases", {})
    codes = {aliases.get(name, name): value for name, value in codes.items()}
    if len(codes) != len(
        set(config["methods"])
        | set(config.get("extra_neural_runs", {}))
        | set(config.get("extra_coefficients", {}))
    ):
        raise ValueError("method aliases must not merge distinct candidates")
    rows = []
    for system in config["systems"]:
        si = systems.index(system)
        variant = config.get("source_variant", "original")
        candidates = [
            Path(root) / f"system{system}_{variant}.npz"
            for root in [config["data"], *config.get("additional_data", [])]
        ]
        paths = [path for path in candidates if path.exists()]
        if len(paths) != 1:
            raise ValueError(f"need exactly one direct truth per system: {paths}")
        with np.load(paths[0]) as z:
            spec = json.loads(str(z["spec"]))
            target, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
        roi = ranges >= 50
        for modes in config["decoder_modes"]:
            geometry = generator_geometry(spec, 64, 16, modes=modes)
            initial = geometry["basis"].T @ (
                geometry["weights"]
                * np.exp(
                    -0.5 * ((geometry["z"] - spec["source_depth_m"]) / spec["source_width_m"]) ** 2
                )
            )
            for method, all_codes in codes.items():
                latent, clipping = latent_from_coefficients(all_codes[si], initial, geometry)
                for fi, frequency in enumerate(frequencies):
                    prediction = query_field(latent, frequency, ranges, depths)
                    row = {
                        "system": system,
                        "source_variant": variant,
                        "method": method,
                        "decoder_modes": modes,
                        "frequency": float(frequency),
                        "latent_real_coefficients": 1024,
                        "clipping_fraction": clipping,
                        **complex_metrics(prediction[:, roi], target[fi][:, roi]),
                    }
                    rows.append(row)
                print(
                    json.dumps(
                        {
                            "system": system,
                            "method": method,
                            "decoder_modes": modes,
                            "mean_raw": float(
                                np.mean([r["complex_rel_l2"] for r in rows[-len(frequencies) :]])
                            ),
                        }
                    ),
                    flush=True,
                )
            (output / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
