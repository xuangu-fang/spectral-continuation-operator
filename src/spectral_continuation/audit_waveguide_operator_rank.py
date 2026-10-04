"""Best fixed response-span limits, not general latent-dimension lower bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rank_statistics(response, ranks):
    matrix = response.reshape(-1, response.shape[-1])
    energy = np.maximum(np.linalg.eigvalsh(matrix.conj().T @ matrix)[::-1], 0)
    if not np.isfinite(energy).all() or energy.sum() <= 0:
        raise ValueError("finite nonzero response energy required")
    return {
        "normalized_singular_values": np.sqrt(energy / energy[0]).tolist(),
        "effective_participation_rank": float(energy.sum() ** 2 / np.sum(energy**2)),
        "best_fixed_span_relative_frobenius": {
            str(rank): float(np.sqrt(energy[rank:].sum() / energy.sum())) for rank in ranks
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    rows, hashes = [], {}
    for system in config["systems"]:
        path = Path(config["data"]) / f"system{system}_operator.npz"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:
            matrix, ranges = data["transfer"].astype(complex), data["ranges_m"]
        selections = {
            "range_ge50": ranges >= 50,
            "range_ge1000": ranges >= 1000,
            "last_profile": np.arange(len(ranges)) == len(ranges) - 1,
        }
        for name, selection in selections.items():
            rows.append(
                {
                    "system": system,
                    "reception": name,
                    **rank_statistics(matrix[selection], config["ranks"]),
                }
            )
    summary = []
    for name in selections:
        selected = [row for row in rows if row["reception"] == name]
        summary.append(
            {
                "reception": name,
                "systems": len(selected),
                "minimum_normalized_singular_value": min(
                    min(r["normalized_singular_values"]) for r in selected
                ),
                "mean_best_fixed_span_errors": {
                    str(rank): float(
                        np.mean(
                            [r["best_fixed_span_relative_frobenius"][str(rank)] for r in selected]
                        )
                    )
                    for rank in config["ranks"]
                },
            }
        )
    (args.output / "results.json").write_text(json.dumps(rows, indent=2))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "source_sha256.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
