"""Paired environment statistics for two three-seed representation runs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from .metrics import paired_bootstrap


def _runs(pattern: str) -> dict[int, Path]:
    result = {}
    for raw_path in glob.glob(pattern):
        path = Path(raw_path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        seed = int(summary["args"]["seed"])
        if seed in result:
            raise RuntimeError(f"duplicate seed {seed} in {pattern!r}")
        result[seed] = path.parent
    if not result:
        raise RuntimeError(f"no summaries matched {pattern!r}")
    return result


def _vectors(run: Path, prefix: str, frequencies: tuple[float, ...]) -> np.ndarray:
    with np.load(run / "validation_metrics.npz", allow_pickle=False) as archive:
        return np.stack(
            [archive[f"{prefix}_{frequency:g}_complex_rel_l2"] for frequency in frequencies]
        )


def run(args) -> None:
    candidate = _runs(args.candidate_glob)
    baseline = _runs(args.baseline_glob)
    if candidate.keys() != baseline.keys():
        raise RuntimeError("candidate and baseline seeds differ")
    output = {
        "candidate": args.candidate_label,
        "baseline": args.baseline_label,
        "seeds": sorted(candidate),
        "independent_unit": "physical validation environment after averaging seeds",
    }
    for split, prefix, raw_frequencies in (
        ("selection", "selection", args.selection_frequencies),
        ("extrapolation", "test", args.extrapolation_frequencies),
    ):
        frequencies = tuple(map(float, raw_frequencies.split(",")))
        candidate_values = np.stack(
            [_vectors(candidate[seed], prefix, frequencies) for seed in sorted(candidate)]
        ).mean(axis=0)
        baseline_values = np.stack(
            [_vectors(baseline[seed], prefix, frequencies) for seed in sorted(baseline)]
        ).mean(axis=0)
        output[split] = {
            "macro": paired_bootstrap(
                baseline_values.mean(axis=0), candidate_values.mean(axis=0)
            ),
            "per_frequency": {
                f"{frequency:g}": paired_bootstrap(
                    baseline_values[index], candidate_values[index]
                )
                for index, frequency in enumerate(frequencies)
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-glob", required=True)
    parser.add_argument("--baseline-glob", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--selection-frequencies", required=True)
    parser.add_argument("--extrapolation-frequencies", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
