"""Predeclared Gaussian MAP controls; no query-driven stopping or selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_waveguide_trajectory_newton import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    options = json.loads(args.config.read_text())
    strengths = options.pop("prior_strengths")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(args.config.read_text())
    rows = []
    for strength in strengths:
        directory = args.output / f"prior_{strength:g}"
        run({**options, "prior_strength": strength}, directory)
        rows.extend(
            {"prior_strength": strength, **row}
            for row in json.loads((directory / "summary.json").read_text())
        )
        (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
