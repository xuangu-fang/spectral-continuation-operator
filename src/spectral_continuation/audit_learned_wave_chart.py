"""Challenge a local learned chart with stronger classical direction dictionaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .learned_wave_chart import WaveChartEncoder, evaluate, on_device


def run(run_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = torch.load(run_path / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    with np.load(run_path / "development_query_pool.npz") as z:
        arrays = {k: z[k] for k in z.files}
    pool = on_device(arrays, "cuda")
    model = WaveChartEncoder(**config["model"], extent=config["data"]["extent"]).cuda()
    model.load_state_dict(checkpoint["model"])
    rows, _ = evaluate(model, pool, arrays)
    results = {"learned": rows}
    summaries = {"learned": float(np.mean([r["complex_rel_l2"] for r in rows]))}
    for branches, degree in ((16, 1), (32, 0), (32, 1), (64, 0)):
        for cone in (180, 90, 60, 45):
            for ridge in (1e-7, 1e-5, 1e-3):
                baseline = WaveChartEncoder(
                    branches=branches,
                    degree=degree,
                    learned=False,
                    ridge=ridge,
                    extent=config["data"]["extent"],
                ).cuda()
                # Forward cones exploit only known source direction, not query labels.
                angles = torch.linspace(
                    -cone * math.pi / 180,
                    cone * math.pi / 180,
                    branches + 1,
                    device="cuda",
                    dtype=torch.float64,
                )[:-1]
                baseline.base_k.copy_(
                    2
                    * math.pi
                    * config["data"]["extent"]
                    * torch.stack((angles.cos(), angles.sin()), -1)
                )
                key = f"fixed_pw{branches}_degree{degree}_cone{cone}_ridge{ridge}"
                rows, _ = evaluate(baseline, pool, arrays)
                results[key] = rows
                summaries[key] = float(np.mean([r["complex_rel_l2"] for r in rows]))
                print(key, summaries[key], flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2))
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    best = min((k for k in summaries if k != "learned"), key=summaries.get)
    print(
        json.dumps(
            {
                "learned": summaries["learned"],
                "best_classical": best,
                "best_error": summaries[best],
                "development_only": True,
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.run, args.output)


if __name__ == "__main__":
    main()
