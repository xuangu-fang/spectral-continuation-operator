"""Warm end-to-end error/cost pilot; includes context encoding and queries.

No data IO, offline model loading/training or metric computation in latency.
Both CPUs and GPU are shared, not exclusive. Independent-instance nonlinear
fits are stronger timing/accuracy controls than the earlier batched audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .audit_waveguide_generator_identification import (
    generator_geometry,
    identify,
    latent_from_coefficients,
    normal_statistics,
)
from .audit_waveguide_generator_prior import conditional_mean, fit_prior
from .audit_waveguide_spectral_transport import encode_field
from .coarse_waveguide_operator import query_field
from .coherent_representation import complex_metrics
from .learned_waveguide_generator import GeneratorEncoder, context_features
from .train_waveguide_generator_field_loss import DifferentiableGenerator


def encode(spec, observed, depths, ranges, order, penalty):
    geometry = generator_geometry(spec, 64, 16)
    coefficients = np.stack(
        [encode_field(field, depths, spec["depth_m"], 32) for field in observed]
    )
    gram, rhs = normal_statistics(spec, coefficients, [60, 100, 140], ranges, geometry, order)
    beta = identify(gram, rhs, 1e-6, penalty)
    return beta, geometry, coefficients, gram, rhs


def fit_context(beta, geometry, coefficients, checkpoints):
    start = time.perf_counter()
    parameter = torch.nn.Parameter(
        torch.tensor(beta[None] / 0.02, device="cuda", dtype=torch.float64)
    )
    decoder = DifferentiableGenerator(geometry).cuda()
    source = torch.tensor(coefficients[0, 0][None], device="cuda").repeat(3, 1)
    background = torch.tensor(geometry["background_q"][None], device="cuda").repeat(3, 1)
    observed = torch.tensor(coefficients, device="cuda")
    frequency = torch.tensor([60, 100, 140], device="cuda")
    roi = torch.tensor(geometry["edges"] >= 50, device="cuda")
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

    def closure():
        nonlocal calls
        optimizer.zero_grad()
        prediction = decoder((parameter * 0.02).repeat(3, 1, 1), background, source, frequency)[
            :, roi
        ]
        target = observed[:, roi]
        loss = (
            (prediction - target).abs().square().sum((1, 2)) / target.abs().square().sum((1, 2))
        ).mean()
        loss.backward()
        calls += 1
        return loss

    for step in range(1, max(checkpoints) + 1):
        optimizer.step(closure)
        if step in checkpoints:
            torch.cuda.synchronize()
            result = (parameter.detach() * 0.02).cpu().numpy()[0]
            elapsed = time.perf_counter() - start
            yield step, result, elapsed, calls
            # Exclude caller's query readout and metric time from optimization.
            start = time.perf_counter() - elapsed


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(2)
    checkpoint = torch.load(
        Path(config["field_run"]) / "best.pt", map_location="cpu", weights_only=False
    )
    model = (
        GeneratorEncoder(checkpoint["mean"].shape[1], **checkpoint["config"]["model"]).cuda().eval()
    )
    model.load_state_dict(checkpoint["state_dict"])
    mean, std = checkpoint["mean"], checkpoint["std"]
    with np.load(Path(config["coefficient_run"]) / "training_statistics.npz") as z:
        x = z["features"].reshape(len(z["features"]), -1).astype(float)
        y = (z["target"] - z["baseline"]).reshape(len(x), -1)
        prior_targets = z["target"].copy() if config.get("gaussian_prior_run") else None
    prior = None
    if prior_targets is not None:
        prior_options = json.loads(
            (Path(config["gaussian_prior_run"]) / "best_results.json").read_text()
        )["options"]
        prior_mean, prior_factor = fit_prior(prior_targets, prior_options["rank"])
        prior = (prior_mean, prior_factor, prior_options["prior_precision"])
    variance = max(float(np.mean(np.sum((x - x.mean(0)) ** 2, axis=-1))), 1e-12)
    options = json.loads((Path(config["coefficient_run"]) / "best_kernel.json").read_text())[
        "options"
    ]

    def distance(a, b):
        return np.maximum(
            ((a * a).sum(-1)[:, None] + (b * b).sum(-1)[None] - 2 * a @ b.T) / variance, 0
        )

    weights = np.linalg.solve(
        np.exp(-distance(x, x) / options["bandwidth"] ** 2) + options["ridge"] * np.eye(len(x)), y
    )
    resource = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": 2,
        "gpu": torch.cuda.get_device_name(),
        "gpu_processes_at_start": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "scope": "shared-machine warm latency pilot, not certified exclusive throughput; disk IO/offline fitting excluded for all",
        "kernel_inference_array_bytes": int(x.nbytes + weights.nbytes),
        "neural_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "gaussian_prior_array_bytes": int(prior[0].nbytes + prior[1].nbytes) if prior else None,
        "latent_real_entries": 64 * 16,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path(__file__).with_name("coherent_waveguide_data.py"),
                Path(__file__).with_name("audit_waveguide_generator_identification.py"),
            ]
        },
        "quadrature_caching": "geometry-independent Gauss-Legendre rule shared by ALL pipelines; no system or frequency solution cached",
    }
    (output / "resources.json").write_text(json.dumps(resource, indent=2))
    rows = []
    for system in config["systems"]:
        with np.load(Path(config["data"]) / f"system{system}_m32_r2049.npz") as z:
            spec = json.loads(str(z["spec"]))
            fields, depths, ranges, frequencies = (
                z["field"],
                z["depths_m"],
                z["ranges_m"],
                z["frequencies_hz"],
            )
        take = np.linspace(0, len(ranges) - 1, 65, dtype=int)
        observed = fields[np.flatnonzero(np.isin(frequencies, [60, 100, 140]))][:, :, take].astype(
            complex
        )
        roi = ranges >= 50

        def record(
            method,
            beta,
            geometry,
            initial,
            encoding,
            repeat,
            extra=None,
            ranges=ranges,
            depths=depths,
            frequencies=frequencies,
            roi=roi,
            fields=fields,
            system=system,
        ):
            start = time.perf_counter()
            latent, _ = latent_from_coefficients(beta, initial, geometry)
            predictions = [
                query_field(latent, f, ranges, depths) for f in config["query_frequencies"]
            ]
            query_seconds = time.perf_counter() - start
            errors = []
            for frequency, prediction in zip(config["query_frequencies"], predictions, strict=True):
                fi = int(np.flatnonzero(frequencies == frequency)[0])
                errors.append(complex_metrics(prediction[:, roi], fields[fi][:, roi]))
            rows.append(
                {
                    "system": system,
                    "method": method,
                    "repeat": repeat,
                    "encoding_seconds": encoding,
                    "six_queries_seconds": query_seconds,
                    "total_seconds": encoding + query_seconds,
                    "raw": float(np.mean([r["complex_rel_l2"] for r in errors])),
                    "phase_rmse_rad": float(
                        np.mean([r["energy_weighted_phase_rmse_rad"] for r in errors])
                    ),
                    **(extra or {}),
                }
            )

        # Warm network libraries with actual context before recorded repetitions.
        b, g, _, h, t = encode(spec, observed, depths, ranges[take], 3, 0)
        f = (context_features(h, t, b, g)[None] - mean) / std
        for _ in range(3):
            with torch.inference_mode():
                model(
                    torch.tensor(f, device="cuda", dtype=torch.float32),
                    torch.tensor(b[None], device="cuda", dtype=torch.float32),
                ).cpu().numpy()
        for repeat in range(config["repeats"]):
            methods = ["regularized_inverse", "kernel_generator", "field_neural"]
            if prior is not None:
                methods.append("gaussian_generator")
            for method in methods:
                start = time.perf_counter()
                order, penalty = (5, 1) if method == "regularized_inverse" else (3, 0)
                beta, geometry, coefficients, gram, rhs = encode(
                    spec, observed, depths, ranges[take], order, penalty
                )
                if method == "gaussian_generator":
                    beta = conditional_mean(gram, rhs, *prior)
                elif method != "regularized_inverse":
                    features = (context_features(gram, rhs, beta, geometry)[None] - mean) / std
                    if method == "kernel_generator":
                        correction = (
                            np.exp(
                                -distance(features.reshape(1, -1), x) / options["bandwidth"] ** 2
                            )
                            @ weights
                        )
                        beta = beta + correction.reshape(beta.shape)
                    else:
                        with torch.inference_mode():
                            beta = (
                                model(
                                    torch.tensor(features, device="cuda", dtype=torch.float32),
                                    torch.tensor(beta[None], device="cuda", dtype=torch.float32),
                                )
                                .cpu()
                                .numpy()[0]
                            )
                encoding = time.perf_counter() - start
                record(method, beta, geometry, coefficients[0, 0], encoding, repeat)
                if method in config.get("refine_initializations", []):
                    for step, refined, elapsed, calls in fit_context(
                        beta, geometry, coefficients, config["nonlinear_steps"]
                    ):
                        record(
                            f"{method}_refine_{step}",
                            refined,
                            geometry,
                            coefficients[0, 0],
                            encoding + elapsed,
                            repeat,
                            {"forward_backward_calls": calls},
                        )
            start = time.perf_counter()
            beta, geometry, coefficients, _, _ = encode(spec, observed, depths, ranges[take], 5, 1)
            encoding = time.perf_counter() - start
            for step, refined, elapsed, calls in fit_context(
                beta, geometry, coefficients, config["nonlinear_steps"]
            ):
                record(
                    f"nonlinear_{step}",
                    refined,
                    geometry,
                    coefficients[0, 0],
                    encoding + elapsed,
                    repeat,
                    {"forward_backward_calls": calls},
                )
        (output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"system": system, "complete": True}), flush=True)
    summary = []
    for method in sorted({r["method"] for r in rows}):
        grouped = []
        for system in config["systems"]:
            selected = [r for r in rows if r["system"] == system and r["method"] == method]
            grouped.append(
                {
                    key: float(np.median([r[key] for r in selected]))
                    for key in [
                        "raw",
                        "phase_rmse_rad",
                        "encoding_seconds",
                        "six_queries_seconds",
                        "total_seconds",
                    ]
                }
            )
        summary.append(
            {
                "method": method,
                **{key: float(np.mean([r[key] for r in grouped])) for key in grouped[0]},
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field-run", type=Path)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.field_run is not None:
        config["field_run"] = str(args.field_run)
    if args.repeats is not None:
        config["repeats"] = args.repeats
    run(config, args.output)


if __name__ == "__main__":
    main()
