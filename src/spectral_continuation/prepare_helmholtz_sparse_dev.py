"""Materialize DEV-only sparse complex observations with a strict public schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_waveguide_spectral_context_robustness import digest

PUBLIC_SPEC_KEYS = (
    "source", "source_width", "sponge_width", "sponge_strength",
    "background_loss",
)


def extract_public_observations(data: dict, config: dict) -> dict:
    """Do not return arbitrary NPZ members or private fields in ``spec``."""
    field = np.asarray(data["field"])
    frequencies = np.asarray(data["frequencies"])
    spec = json.loads(str(data["spec"]))
    if field.shape != (len(frequencies), 257, 257):
        raise ValueError("unexpected teacher shape")
    if len(frequencies) != len(np.unique(frequencies)):
        raise ValueError("duplicate frequency")
    if set(PUBLIC_SPEC_KEYS) - set(spec):
        raise ValueError("missing public source or boundary metadata")
    x = np.asarray(config["x_indices"], dtype=int)
    y_spec = config["y_indices"]
    y = y_spec["start"] + y_spec["step"] * np.arange(y_spec["count"])
    if len(x) != 4 or len(y) != 65 or np.any(x <= 0) or np.any(x >= 256):
        raise ValueError("unexpected receiver geometry")
    if np.any(y <= 0) or np.any(y >= 256) or len(np.unique(x)) != 4:
        raise ValueError("unexpected receiver geometry")
    context_f = np.asarray(config["context_frequencies_hz"], dtype=float)
    if len(context_f) != 3 or np.intersect1d(
        context_f, config["reserved_query_frequencies_hz"]
    ).size:
        raise ValueError("context and query frequency leak")
    indices = []
    for f in context_f:
        found = np.flatnonzero(np.isclose(frequencies, f, atol=0, rtol=0))
        if len(found) != 1:
            raise ValueError(f"missing or duplicate context frequency {f}")
        indices.append(int(found[0]))
    samples = field[np.asarray(indices)[:, None, None], y[None, :, None], x[None, None, :]]
    if samples.shape != (3, 65, 4) or not np.isfinite(samples).all():
        raise ValueError("invalid complex observations")
    public_spec = {key: spec[key] for key in PUBLIC_SPEC_KEYS}
    return {
        "samples": samples.astype(np.complex64, copy=False),
        "frequencies": context_f,
        "x": x / 256,
        "y": y / 256,
        "public_spec": json.dumps(public_spec, sort_keys=True),
    }


def run(config: dict, output: Path) -> None:
    expected = (list(range(20269600, 20269608)), 257,
                [88, 128, 168, 208], {"start": 32, "step": 3, "count": 65},
                [6.0, 10.0, 14.0], [11.3, 17.0])
    actual = (config["systems"], config["teacher_grid"], config["x_indices"],
              config["y_indices"], config["context_frequencies_hz"],
              config["reserved_query_frequencies_hz"])
    if actual != expected:
        raise ValueError("frozen DEV sparse-context setting changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    manifest = {"scope": config["scope"], "implementation_sha256": digest(__file__),
                "source_sha256": {}, "public_sha256": {}}
    for ordinal, system in enumerate(config["systems"]):
        source = Path(config["source_data"]) / f"system_{system}_grid257.npz"
        manifest["source_sha256"][str(source)] = digest(source)
        with np.load(source) as data:
            if json.loads(str(data["spec"]))["seed"] != system:
                raise ValueError("system seed mismatch")
            public = extract_public_observations(data, config)
        destination = output / f"dev_{ordinal:02d}_four_lines.npz"
        np.savez_compressed(destination, **public)
        manifest["public_sha256"][destination.name] = digest(destination)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
