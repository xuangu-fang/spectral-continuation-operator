"""Register a held-out confirmation protocol before generating any target truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generation = json.loads(args.generation_config.read_text())
    inference = json.loads(args.inference_config.read_text())
    train = set(range(generation["train_seed_start"], generation["train_seed_start"] +
                      generation["train_count"]))
    new = set(generation["systems"])
    audit = set(generation["audit_systems"])
    old = set(range(20264600, 20264632))
    if not len(new) == 32 or train & (new | audit) or new & (audit | old):
        raise ValueError("confirmation seeds are incomplete or overlap old data")
    if new != set(inference["systems"]):
        raise ValueError("generation and inference systems differ")
    if Path(generation["output_data"]).exists() or Path(inference["source_codes"]).exists():
        raise ValueError("confirmation data already exists; no pre-truth registration")
    if args.output.exists():
        raise ValueError("registration already exists")
    paths = [
        args.generation_config,
        args.inference_config,
        Path(inference["training_labels"]),
        Path("src/spectral_continuation/audit_waveguide_spectral_medium.py"),
        Path("src/spectral_continuation/audit_waveguide_spectral_rank256_inference.py"),
        Path("src/spectral_continuation/coherent_waveguide_data.py"),
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "registration": "BEFORE new independent validation and numerical-audit truth; rank256 selected on old first8 only; all new32 systems fixed",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "systems": generation["systems"],
        "audit_systems": generation["audit_systems"],
        "confirm_gate": inference["confirm_gate"],
        "tracked_sha256": {str(path): digest(path) for path in paths},
    }
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"registered_utc": record["registered_utc"], "systems": len(new)}))


if __name__ == "__main__":
    main()
