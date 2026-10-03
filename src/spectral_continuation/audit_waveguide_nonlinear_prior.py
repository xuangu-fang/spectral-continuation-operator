"""Classical empirical Gaussian generator prior with nonlinear observation fit.

Unlike a prior warm start followed by unconstrained fitting, every iterate stays
in the training-derived subspace. Projection of true development coefficients is
separately evaluated as an ORACLE and never used to initialize inference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_generator_prior import conditional_mean, fit_prior, unpack_statistics
from .audit_waveguide_spectral_transport import encode_field
from .learned_waveguide_generator import error, evaluate, prepare
from .train_waveguide_generator_field_loss import DifferentiableGenerator


def run(options, output):
    directory = Path(options["training_run"])
    config = json.loads((directory / "config.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({**options, "base": config}, indent=2))
    torch.set_num_threads(2)
    with np.load(directory / "training_statistics.npz") as z:
        target = z["target"].astype(float)
    mean, factor = fit_prior(target, options["rank"])
    flat = factor.reshape(-1, factor.shape[-1])
    val, metadata = prepare(config, config["validation_systems"], False)
    init = []
    for features in val["features"]:
        gram, rhs = unpack_statistics(features, config["depth_terms"])
        beta = conditional_mean(gram, rhs, mean, factor, options["initial_prior_precision"])
        init.append(np.linalg.lstsq(flat, (beta - mean).ravel(), rcond=None)[0])
    # ORACLE only: capacity of the training-derived real coefficient subspace.
    truth_coordinates = np.linalg.lstsq(
        flat, (val["target"] - mean).reshape(len(metadata), -1).T, rcond=None
    )[0].T
    projected = mean + np.einsum("rdk,bk->brd", factor, truth_coordinates)
    oracle = evaluate(projected, metadata, config)
    (output / "projection_oracle.json").write_text(json.dumps(oracle, indent=2))
    print(json.dumps({"rank": options["rank"], "projection_oracle_raw": error(oracle)}), flush=True)
    contexts = []
    for item in metadata:
        take = np.linspace(0, len(item["ranges"]) - 1, config["cells"] + 1, dtype=int)
        contexts.append(
            np.array(
                [
                    encode_field(
                        item["fields"][int(np.flatnonzero(item["frequencies"] == f)[0])][:, take],
                        item["depths"],
                        item["geometry"]["depth"],
                        32,
                    )
                    for f in config["observed_frequencies"]
                ]
            )
        )
    geometry = metadata[0]["geometry"]
    decoder = DifferentiableGenerator(geometry).cuda()
    observed = torch.tensor(np.array(contexts), device="cuda")
    background = torch.tensor(
        np.array([m["geometry"]["background_q"] for m in metadata]), device="cuda"
    )
    initial = torch.tensor(np.array([m["initial"] for m in metadata]), device="cuda")
    frequencies = torch.tensor(config["observed_frequencies"], dtype=torch.float64, device="cuda")
    prior_mean = torch.tensor(mean, device="cuda")
    prior_factor = torch.tensor(factor, device="cuda")
    roi = torch.tensor(geometry["edges"] >= config["roi_min_range_m"], device="cuda")
    history = []
    for strength in options["prior_strengths"]:
        parameter = torch.nn.Parameter(
            torch.tensor(np.array(init), device="cuda", dtype=torch.float64)
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
        last_data = None
        last_prior = None

        def coefficients(parameter=parameter):
            return prior_mean + torch.einsum("rdk,bk->brd", prior_factor, parameter)

        def closure(
            optimizer=optimizer, parameter=parameter, strength=strength, coefficients=coefficients
        ):
            nonlocal calls, last_data, last_prior
            optimizer.zero_grad()
            predicted = decoder(
                coefficients().repeat_interleave(len(frequencies), 0),
                background.repeat_interleave(len(frequencies), 0),
                initial.repeat_interleave(len(frequencies), 0),
                frequencies.repeat(len(initial)),
            )[:, roi]
            truth = observed.reshape(-1, observed.shape[-2], observed.shape[-1])[:, roi]
            data = (
                (predicted - truth).abs().square().sum((1, 2)) / truth.abs().square().sum((1, 2))
            ).mean()
            prior = parameter.square().mean() * strength
            loss = data + prior
            loss.backward()
            calls += 1
            last_data, last_prior = float(data.detach()), float(prior.detach())
            return loss

        seconds = 0.0
        for step in range(max(options["checkpoints"]) + 1):
            if step in options["checkpoints"]:
                beta = coefficients().detach().cpu().numpy()
                rows = evaluate(beta, metadata, config)
                row = {
                    "prior_strength": strength,
                    "step": step,
                    "macro_error": error(rows),
                    "context_objective": last_data,
                    "prior_penalty": last_prior,
                    "forward_backward_calls": calls,
                    "optimization_seconds": seconds,
                }
                history.append(row)
                prefix = f"prior{strength}_step{step}"
                (output / f"{prefix}_results.json").write_text(json.dumps(rows, indent=2))
                np.savez_compressed(
                    output / f"{prefix}_coefficients.npz",
                    coefficients=beta,
                    systems=config["validation_systems"],
                )
                (output / "history.json").write_text(json.dumps(history, indent=2))
                print(json.dumps(row), flush=True)
            if step == max(options["checkpoints"]):
                break
            start = time.perf_counter()
            optimizer.step(closure)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
