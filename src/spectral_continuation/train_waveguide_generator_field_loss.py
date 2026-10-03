"""Single raw-complex field loss through a differentiable generator decoder.

No added phase/PDE/medium losses. The from-classical option uses no medium
labels for training; the coefficient-pretrained option explicitly does.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit_waveguide_generator_identification import generator_geometry
from .learned_waveguide_generator import GeneratorEncoder, error, evaluate, prepare


class DifferentiableGenerator(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        for key in ["basis", "weights", "psi", "kz2"]:
            self.register_buffer(key, torch.tensor(geometry[key], dtype=torch.float64))
        self.speed = geometry["reference_speed"]
        self.attenuation = geometry["attenuation"]
        self.step = float(np.diff(geometry["edges"])[0])

    def forward(self, beta, background_q, initial, frequency):
        total = background_q[:, None] + beta.double() @ self.psi.T
        total = torch.clamp(total, 0.65, 1.5)
        mass = torch.einsum(
            "zm,brz,zn->brmn", self.basis, total * self.weights / self.speed**2, self.basis
        )
        vertical = (2 * torch.pi * frequency.double())[
            :, None, None, None
        ] ** 2 * mass - torch.diag(self.kz2)
        eigenvalues, eigenvectors = torch.linalg.eigh(vertical)
        rate = 1j * torch.sqrt(eigenvalues.to(torch.complex128)) - self.attenuation
        phase = torch.exp(rate * self.step)
        vectors = eigenvectors.to(torch.complex128)
        current = initial.to(torch.complex128)
        fields = [current]
        for cell in range(beta.shape[1]):
            projection = torch.einsum("bmn,bm->bn", vectors[:, cell], current)
            current = torch.einsum("bmn,bn->bm", vectors[:, cell], phase[:, cell] * projection)
            fields.append(current)
        return torch.stack(fields, dim=1)


def load_field_supervision(config):
    backgrounds, initials, targets = [], [], []
    for system in config["train_systems"]:
        with np.load(Path(config["training_modal_cache"]) / f"system{system}_modal.npz") as z:
            spec = json.loads(str(z["spec"]))
            freqs = z["frequencies_hz"]
            query = [int(np.flatnonzero(freqs == f)[0]) for f in config["query_frequencies"]]
            coefficients = z["modal_coefficients"]
            step = (coefficients.shape[1] - 1) // config["cells"]
            if step * config["cells"] != coefficients.shape[1] - 1:
                raise ValueError("teacher and decoder coordinates must coincide")
            targets.append(coefficients[query, ::step])
            initials.append(coefficients[0, 0])
            geometry = generator_geometry(spec, config["cells"], config["depth_terms"])
            backgrounds.append(geometry["background_q"])
    return {
        "background": np.array(backgrounds),
        "initial": np.array(initials),
        "target_field": np.array(targets),
    }, geometry


def sampled_gaussian_initial(geometry, count, options, generator, device="cuda"):
    """Known training query source, not read from an unobserved response."""
    draw = torch.rand((count, 2), device=device, dtype=torch.float64, generator=generator)
    bounds = options["source_depth_bounds_m"]
    depths = bounds[0] + (bounds[1] - bounds[0]) * draw[:, 0]
    bounds = options["source_width_bounds_m"]
    widths = bounds[0] + (bounds[1] - bounds[0]) * draw[:, 1]
    z = torch.tensor(geometry["z"], device=device)
    weight = torch.tensor(geometry["weights"], device=device)
    basis = torch.tensor(geometry["basis"], device=device)
    return (torch.exp(-0.5 * ((z[None] - depths[:, None]) / widths[:, None]) ** 2) * weight) @ basis


def run(options, output):
    source = Path(options["coefficient_run"])
    config = json.loads((source / "config.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    full_config = {**config, "field_loss_options": options}
    (output / "config.json").write_text(json.dumps(full_config, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(options["seed"])
    start = time.perf_counter()
    with np.load(source / "training_statistics.npz") as z:
        # The default actual-field protocol deliberately never loads medium labels.
        train = {k: z[k] for k in ["features", "baseline"]}
        if options.get("target_kind", "cached_actual_field") == "projected_generator":
            train["target_generator"] = z["target"]
        mean, std = z["mean"], z["std"]
    val, validation = prepare(config, config["validation_systems"], False)
    val["features"] = (val["features"] - mean) / std
    supervision, geometry = load_field_supervision(config)
    model = GeneratorEncoder(train["features"].shape[1], **config["model"]).cuda()
    if options["initialization"] == "coefficient_pretrained":
        checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    elif options["initialization"] != "classical":
        raise ValueError(options["initialization"])
    decoder = DifferentiableGenerator(geometry).cuda()
    train_gpu = {k: torch.as_tensor(v, device="cuda") for k, v in {**train, **supervision}.items()}
    val_gpu = {k: torch.as_tensor(val[k], device="cuda") for k in ["features", "baseline"]}
    frequencies = torch.tensor(config["query_frequencies"], device="cuda", dtype=torch.float64)
    ranges = np.linspace(0, geometry["edges"][-1], config["cells"] + 1)
    roi = torch.tensor(ranges >= config["roi_min_range_m"], device="cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options["learning_rate"], weight_decay=0.00001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, options["steps"])
    source_rng = torch.Generator(device="cuda").manual_seed(options["seed"] + 10000)
    best, history = float("inf"), []
    batch_losses = []
    print(
        json.dumps(
            {
                "data_ready_seconds": time.perf_counter() - start,
                "initialization": options["initialization"],
            }
        ),
        flush=True,
    )
    for step in range(options["steps"] + 1):
        if step % options["eval_every"] == 0 or step == options["steps"]:
            with torch.no_grad():
                prediction = model(val_gpu["features"], val_gpu["baseline"]).cpu().numpy()
            rows = evaluate(prediction, validation, config)
            record = {
                "step": step,
                "macro_error": error(rows),
                "train_batch_raw_error": float(np.mean(batch_losses)) if batch_losses else None,
                "seconds": time.perf_counter() - start,
            }
            history.append(record)
            batch_losses = []
            print(json.dumps(record), flush=True)
            if error(rows) < best:
                best = error(rows)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": full_config,
                        "mean": mean,
                        "std": std,
                        "step": step,
                    },
                    output / "best.pt",
                )
                (output / "best_results.json").write_text(json.dumps(rows, indent=2))
            (output / "history.json").write_text(json.dumps(history, indent=2))
        if step == options["steps"]:
            break
        index = torch.randint(len(train["features"]), (options["batch_size"],), device="cuda")
        fi = torch.randint(len(frequencies), (options["batch_size"],), device="cuda")
        beta = model(train_gpu["features"][index], train_gpu["baseline"][index])
        initial = train_gpu["initial"][index]
        if options.get("source_sampling", "original") == "random_gaussian":
            if options.get("target_kind") != "projected_generator":
                raise ValueError("new source targets require explicit training-medium supervision")
            initial = sampled_gaussian_initial(geometry, options["batch_size"], options, source_rng)
        field = decoder(beta, train_gpu["background"][index], initial, frequencies[fi])[:, roi]
        if options.get("target_kind", "cached_actual_field") == "projected_generator":
            with torch.no_grad():
                target = decoder(
                    train_gpu["target_generator"][index],
                    train_gpu["background"][index],
                    initial,
                    frequencies[fi],
                )[:, roi]
        else:
            target = train_gpu["target_field"][index, fi][:, roi]
        relative = (field - target).abs().square().sum((1, 2)) / target.abs().square().sum((1, 2))
        # Constant scaling changes numerical conditioning, not the objective.
        loss = relative.mean() * 10000
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite field objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        batch_losses.append(float(relative.detach().sqrt().mean()))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": full_config,
            "mean": mean,
            "std": std,
            "step": options["steps"],
        },
        output / "last.pt",
    )
    summary = {
        "best_macro_error": best,
        "initialization": options["initialization"],
        "medium_label_supervision": options["initialization"] == "coefficient_pretrained"
        or options.get("target_kind") == "projected_generator",
        "target_kind": options.get("target_kind", "cached_actual_field"),
        "source_sampling": options.get("source_sampling", "original"),
        "single_loss": "raw complex modal relative squared error; no added losses",
        "scope": "single-seed development; same 65-profile information and 64-cell decoder",
        "seconds": time.perf_counter() - start,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization", choices=["classical", "coefficient_pretrained"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--coefficient-run", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.initialization:
        config["initialization"] = args.initialization
    if args.steps is not None:
        config["steps"] = args.steps
    if args.coefficient_run is not None:
        config["coefficient_run"] = str(args.coefficient_run)
    if args.seed is not None:
        config["seed"] = args.seed
    run(config, args.output)


if __name__ == "__main__":
    main()
