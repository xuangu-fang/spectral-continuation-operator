"""Strong target-free nonlinear generator fit to exactly the observed fields.

This is a classical inverse solve, not a neural learner. Query truth is never
used in its objective or stopping rule. Fixed optimization checkpoints are
reported, including regression. Batched timings are not deployment latency.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_generator_identification import (
    generator_geometry,
    identify,
    normal_statistics,
)
from .audit_waveguide_spectral_transport import encode_field
from .learned_waveguide_generator import error, evaluate
from .train_waveguide_generator_field_loss import DifferentiableGenerator


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    initial, beta, background, contexts, metadata = [], [], [], [], []
    start = time.perf_counter()
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system{system}_m32_r2049.npz") as z:
            spec = json.loads(str(z["spec"]))
            fields, frequencies, ranges, depths = (
                z["field"],
                z["frequencies_hz"],
                z["ranges_m"],
                z["depths_m"],
            )
        obs = np.flatnonzero(np.isin(frequencies, config["observed_frequencies"]))
        take = np.linspace(0, len(ranges) - 1, config["cells"] + 1, dtype=int)
        coefficients = np.stack(
            [encode_field(fields[i][:, take], depths, spec["depth_m"], 32) for i in obs]
        )
        geometry = generator_geometry(spec, config["cells"], config["depth_terms"])
        gram, rhs = normal_statistics(
            spec,
            coefficients,
            frequencies[obs],
            ranges[take],
            geometry,
            config["interpolation_order"],
        )
        beta.append(identify(gram, rhs, config["ridge"], config["range_penalty"]))
        initial.append(coefficients[0, 0])
        contexts.append(coefficients)
        background.append(geometry["background_q"])
        metadata.append(
            {
                "system": system,
                "geometry": geometry,
                "initial": coefficients[0, 0],
                "fields": fields,
                "depths": depths,
                "ranges": ranges,
                "frequencies": frequencies,
            }
        )
    physical_setup_seconds = time.perf_counter() - start
    parameter = torch.nn.Parameter(
        torch.tensor(np.array(beta) / 0.02, device="cuda", dtype=torch.float64)
    )
    initial = torch.tensor(np.array(initial), device="cuda")
    background = torch.tensor(np.array(background), device="cuda")
    observed = torch.tensor(np.array(contexts), device="cuda")
    decoder = DifferentiableGenerator(geometry).cuda()
    frequencies = torch.tensor(config["observed_frequencies"], device="cuda", dtype=torch.float64)
    roi = torch.tensor(
        np.linspace(0, spec["range_m"], config["cells"] + 1) >= config["roi_min_range_m"],
        device="cuda",
    )
    optimizer = torch.optim.LBFGS(
        [parameter],
        lr=1,
        max_iter=1,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )
    calls = 0
    last_loss = None

    def closure():
        nonlocal calls, last_loss
        optimizer.zero_grad()
        b = (parameter * 0.02).repeat_interleave(len(frequencies), dim=0)
        prediction = decoder(
            b,
            background.repeat_interleave(len(frequencies), 0),
            initial.repeat_interleave(len(frequencies), 0),
            frequencies.repeat(len(initial)),
        )[:, roi]
        target = observed.reshape(-1, observed.shape[-2], observed.shape[-1])[:, roi]
        loss = (
            (prediction - target).abs().square().sum((1, 2)) / target.abs().square().sum((1, 2))
        ).mean()
        loss.backward()
        calls += 1
        last_loss = float(loss.detach())
        return loss

    history = []
    optimization_seconds = 0.0
    for step in range(max(config["checkpoints"]) + 1):
        if step in config["checkpoints"]:
            rows = evaluate((parameter.detach() * 0.02).cpu().numpy(), metadata, config)
            entry = {
                "step": step,
                "query_macro_error": error(rows),
                "context_objective": last_loss,
                "forward_backward_calls": calls,
                "optimization_seconds": optimization_seconds,
                "setup_seconds": physical_setup_seconds,
                "scope": "batched classical inverse; fixed steps, no target-based stopping",
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
            (output / f"step{step}_results.json").write_text(json.dumps(rows, indent=2))
            (output / "history.json").write_text(json.dumps(history, indent=2))
            np.savez_compressed(
                output / f"step{step}_coefficients.npz",
                beta=(parameter.detach() * 0.02).cpu().numpy(),
            )
        if step == max(config["checkpoints"]):
            break
        start = time.perf_counter()
        optimizer.step(closure)
        torch.cuda.synchronize()
        optimization_seconds += time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
