"""Train the oscillatory-field representation POC on synthetic or Ocean data."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import metric_vectors
from .model import parameter_count
from .representation import (
    CarrierGaugedPoleOperator,
    ComplexDelayOperator,
    EquivariantSpectralGenerator,
    ImageQueryOperator,
    LocalSpanQueryOperator,
    PhaseEquivariantNeuralODE,
    StableSpectralStateOperator,
    phase_equivariance_error,
)
from .representation_data import (
    FrequencyBandDataset,
    FrequencySetDataset,
    fit_representation_normalization,
    load_representation_bundle,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complex_relative_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, power: int = 2
) -> torch.Tensor:
    numerator = (prediction - target).abs().square().flatten(1).sum(1)
    denominator = target.abs().square().flatten(1).sum(1).clamp_min(1e-8)
    ratio = numerator / denominator
    if power == 1:
        ratio = ratio.clamp_min(1e-12).sqrt()
    elif power != 2:
        raise ValueError("relative loss power must be one or two")
    return ratio.mean()


def coordinates(
    environment: torch.Tensor,
    query: torch.Tensor,
    frequency_scale: float,
) -> torch.Tensor:
    batch, _, height, width = environment.shape
    y = torch.linspace(-1.0, 1.0, height, device=environment.device)
    x = torch.linspace(-1.0, 1.0, width, device=environment.device)
    yy = y.view(1, 1, height, 1).expand(batch, 1, height, width)
    xx = x.view(1, 1, 1, width).expand(batch, 1, height, width)
    q = (query / frequency_scale).view(batch, 1, 1, 1).expand(batch, 1, height, width)
    return torch.cat((environment, yy, xx, q), dim=1)


@torch.no_grad()
def evaluate(model, loader, queries, normalization, device):
    model.eval()
    predictions = [[] for _ in queries]
    targets = [[] for _ in queries]
    equivariance = []
    for batch_index, batch in enumerate(loader):
        grouped = len(batch) == 3
        if grouped:
            fields, environment, target = batch
            query = torch.full(
                (fields.shape[0],), queries[0], dtype=fields.real.dtype
            )
            query_index = None
        else:
            fields, environment, target, query, query_index = batch
        fields = fields.to(device)
        environment = environment.to(device)
        target = target.to(device)
        query = query.to(device)
        context = coordinates(environment, query, normalization.frequency_scale)
        if grouped and hasattr(model, "forward_queries"):
            output = model.forward_queries(fields, context, queries)
        else:
            output = model(fields, context, query)
        if batch_index == 0:
            equivariance.append(phase_equivariance_error(model, fields, context, query))
        for index in range(len(queries)):
            if grouped:
                predictions[index].append(output[:, index].cpu())
                targets[index].append(target[:, index].cpu())
            else:
                selected = query_index == index
                if selected.any():
                    predictions[index].append(output[selected].cpu())
                    targets[index].append(target[selected].cpu())
    result, vectors = {}, {}
    for index, frequency in enumerate(queries):
        prediction = torch.cat(predictions[index])
        target = torch.cat(targets[index])
        stacked_prediction = torch.stack((prediction.real, prediction.imag), dim=1)
        stacked_target = torch.stack((target.real, target.imag), dim=1)
        current = metric_vectors(stacked_prediction, stacked_target)
        key = f"{frequency:g}"
        result[key] = {name: float(values.mean()) for name, values in current.items()}
        vectors[key] = current
    result["macro_average"] = {
        metric: float(np.mean([result[f"{frequency:g}"][metric] for frequency in queries]))
        for metric in next(iter(result.values()))
    }
    result["phase_equivariance_relative_error"] = float(np.mean(equivariance))
    return result, vectors


def build_model(args, visible, environment_channels, frequency_scale):
    if args.model == "image_unet":
        return ImageQueryOperator(
            len(visible),
            environment_channels,
            width=args.width,
            frequency_scale=frequency_scale,
        )
    if args.model == "complex_delay":
        return ComplexDelayOperator(
            visible,
            environment_channels,
            modes=args.modes,
            blocks=args.blocks,
            max_delay=args.max_delay,
            frequency_scale=frequency_scale,
        )
    if args.model == "local_span":
        return LocalSpanQueryOperator(
            visible,
            environment_channels,
            token_grid=tuple(args.token_grid),
            width=args.state_width,
            blocks=args.blocks,
            frequency_scale=args.token_frequency_scale,
            weight_bound=args.token_weight_bound,
            joint_mask_fraction=args.joint_mask_fraction,
            carrier_mode=args.token_carrier,
        )
    if args.model == "carrier_poles":
        return CarrierGaugedPoleOperator(
            visible,
            environment_channels,
            poles=getattr(args, "poles", 4),
            width=args.state_width,
            blocks=args.blocks,
            max_growth_per_hz=getattr(args, "max_growth_per_hz", 0.025),
            max_residual_angle_per_hz=getattr(
                args, "max_residual_angle_per_hz", math.pi
            ),
            residue_scale=getattr(args, "residue_scale", 1.0),
            state_backbone=getattr(args, "state_backbone", "dilated"),
            residue_mode=getattr(args, "residue_mode", "predicted"),
            residue_ridge=getattr(args, "residue_ridge", 1e-3),
            pole_growth=getattr(args, "pole_growth", "signed"),
            learn_carrier_correction=getattr(
                args, "learn_carrier_correction", False
            ),
            max_carrier_correction_per_hz=getattr(
                args, "max_carrier_correction_per_hz", 0.15
            ),
            state_normalization=getattr(args, "state_normalization", False),
            pole_anchor=getattr(args, "pole_anchor", "reference_residual"),
            carrier_estimator=getattr(args, "carrier_estimator", "band_ols"),
            max_curvature_per_hz2=getattr(args, "max_curvature_per_hz2", 0.0),
            global_latent_channels=getattr(args, "global_latent_channels", 0),
            hyper_carrier_correction=getattr(args, "hyper_carrier_correction", False),
            max_hyper_slope_per_hz=getattr(args, "max_hyper_slope_per_hz", 0.08),
            max_hyper_curvature_per_hz2=getattr(
                args, "max_hyper_curvature_per_hz2", 0.005
            ),
            local_carrier_token_grid=(
                tuple(args.local_carrier_token_grid)
                if args.local_carrier_token_grid is not None
                else None
            ),
            local_carrier_token_width=args.local_carrier_token_width,
            max_local_token_slope_per_hz=args.max_local_token_slope_per_hz,
            max_local_token_curvature_per_hz2=(
                args.max_local_token_curvature_per_hz2
            ),
        )
    if args.model == "component_generator":
        return EquivariantSpectralGenerator(
            visible,
            environment_channels,
            components=getattr(args, "components", 8),
            complex_width=getattr(args, "complex_width", 24),
            state_width=args.state_width,
            blocks=args.blocks,
            max_growth_per_hz=getattr(args, "max_growth_per_hz", 0.025),
            max_angle_per_hz=getattr(args, "max_residual_angle_per_hz", math.pi),
            max_curvature_per_hz2=getattr(args, "max_curvature_per_hz2", 0.01),
            carrier_mode=getattr(args, "generator_carrier", "endpoint"),
            carrier_kernel=getattr(args, "carrier_kernel", 5),
            parameter_smoothing=getattr(args, "parameter_smoothing", 1),
            stable_growth=getattr(args, "pole_growth", "stable") == "stable",
        )
    if args.model == "carrier_ode":
        return PhaseEquivariantNeuralODE(
            visible,
            environment_channels,
            width=args.width,
            frequency_scale=frequency_scale,
            max_update_rate=getattr(args, "max_update_rate", 1.0),
            carrier_mode=getattr(args, "generator_carrier", "endpoint"),
            carrier_kernel=getattr(args, "carrier_kernel", 5),
        )
    return StableSpectralStateOperator(
        visible,
        environment_channels,
        order=args.order,
        width=args.state_width,
        blocks=args.blocks,
        stability_radius=args.stability_radius,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    started = time.perf_counter()
    visible = tuple(map(float, args.visible.split(",")))
    train_queries = tuple(map(float, args.train_queries.split(",")))
    test_queries = tuple(map(float, args.test_queries.split(",")))
    all_frequencies = tuple(sorted(set(visible + train_queries + test_queries)))
    bundle = load_representation_bundle(
        args.domain,
        args.dataset,
        train_count=args.synthetic_train,
        validation_count=args.synthetic_validation,
        height=args.synthetic_height,
        width=args.synthetic_width,
        frequencies=all_frequencies,
        seed=args.data_seed,
    )
    train_positions = np.flatnonzero(bundle.split == "train")
    val_positions = np.flatnonzero(bundle.split == "val")
    normalization = fit_representation_normalization(
        bundle, train_positions, visible, train_queries
    )
    train_dataset_type = FrequencyBandDataset if args.grouped_training else FrequencySetDataset
    train_dataset = train_dataset_type(
        bundle,
        train_positions,
        visible,
        train_queries,
        normalization,
        field_only=args.field_only,
    )
    evaluation_dataset_type = (
        FrequencyBandDataset
        if args.grouped_training
        and args.model
        in {
            "carrier_poles",
            "stable_recurrence",
            "component_generator",
            "carrier_ode",
            "local_span",
        }
        else FrequencySetDataset
    )
    selection_dataset = evaluation_dataset_type(
        bundle,
        val_positions,
        visible,
        train_queries,
        normalization,
        field_only=args.field_only,
    )
    test_dataset = evaluation_dataset_type(
        bundle,
        val_positions,
        visible,
        test_queries,
        normalization,
        field_only=args.field_only,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    seed_all(args.seed)
    device = torch.device(args.device)
    model = build_model(
        args,
        visible,
        train_dataset.environment_channels,
        normalization.frequency_scale,
    ).to(device)
    checkpoint_load = None
    if args.initial_checkpoint is not None:
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(
            initial["model"], strict=not args.allow_partial_initial_checkpoint
        )
        checkpoint_load = {
            "missing_keys": incompatible.missing_keys,
            "unexpected_keys": incompatible.unexpected_keys,
        }
    if args.freeze_initial_backbone:
        if args.initial_checkpoint is None:
            raise ValueError("freezing the initial backbone requires --initial-checkpoint")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("local_carrier_")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("the model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    selection, _ = evaluate(
        model, selection_loader, train_queries, normalization, device
    )
    best_value = selection["macro_average"]["complex_rel_l2"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    rows = [{"epoch": 0, "train_loss": None, "selection_complex_rel_l2": best_value}]
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            grouped = len(batch) == 3
            if grouped:
                fields, environment, target = batch
                query = torch.full(
                    (fields.shape[0],), train_queries[0], dtype=fields.real.dtype
                )
            else:
                fields, environment, target, query, _ = batch
            fields = fields.to(device)
            environment = environment.to(device)
            target = target.to(device)
            query = query.to(device)
            context = coordinates(environment, query, normalization.frequency_scale)
            optimizer.zero_grad(set_to_none=True)
            if (
                grouped
                and isinstance(model, PhaseEquivariantNeuralODE)
                and args.ode_teacher_forcing
            ):
                static = model.static_features(fields)
                sequence = torch.cat((fields[:, -2:], target), dim=1)
                predictions = []
                for index in range(len(train_queries)):
                    predictions.append(
                        model.step(
                            fields,
                            static,
                            sequence[:, index],
                            sequence[:, index + 1],
                            context,
                            model.reference_frequency + index,
                            model.frequency_step,
                        )
                    )
                prediction = torch.stack(predictions, dim=1)
                query_losses = [
                    complex_relative_loss(
                        prediction[:, index],
                        target[:, index],
                        power=args.relative_loss_power,
                    )
                    for index in range(len(train_queries))
                ]
                loss = torch.stack(query_losses).mean()
            elif grouped and hasattr(model, "forward_queries"):
                prediction = model.forward_queries(fields, context, train_queries)
                query_losses = [
                    complex_relative_loss(
                        prediction[:, index],
                        target[:, index],
                        power=args.relative_loss_power,
                    )
                    for index in range(len(train_queries))
                ]
                loss = torch.stack(query_losses).mean()
            else:
                prediction = model(fields, context, query)
                loss = complex_relative_loss(
                    prediction, target, power=args.relative_loss_power
                )
            if isinstance(
                model, (StableSpectralStateOperator, CarrierGaugedPoleOperator)
            ) and args.history_weight > 0:
                loss = loss + args.history_weight * model.visible_reconstruction_loss(
                    fields, context
                )
            if isinstance(model, LocalSpanQueryOperator) and args.mask_weight > 0:
                loss = loss + args.mask_weight * model.masked_reconstruction_loss(
                    fields, context, args.mask_pretraining
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        selection, _ = evaluate(
            model, selection_loader, train_queries, normalization, device
        )
        current = selection["macro_average"]["complex_rel_l2"]
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "selection_complex_rel_l2": current,
            }
        )
        if current < best_value:
            best_value = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:03d}/{args.epochs} train={np.mean(losses):.5f} "
            f"select={current:.5f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    selection, selection_vectors = evaluate(
        model, selection_loader, train_queries, normalization, device
    )
    extrapolation, extrapolation_vectors = evaluate(
        model, test_loader, test_queries, normalization, device
    )
    cross_resolution = None
    if args.domain == "synthetic" and args.cross_resolution_factor > 1:
        high_bundle = load_representation_bundle(
            "synthetic",
            None,
            train_count=args.synthetic_train,
            validation_count=args.synthetic_validation,
            height=args.synthetic_height * args.cross_resolution_factor,
            width=args.synthetic_width * args.cross_resolution_factor,
            frequencies=all_frequencies,
            seed=args.data_seed,
        )
        high_dataset = FrequencySetDataset(
            high_bundle,
            val_positions,
            visible,
            test_queries,
            normalization,
            field_only=args.field_only,
        )
        high_loader = DataLoader(
            high_dataset,
            batch_size=max(1, args.eval_batch_size // args.cross_resolution_factor),
            shuffle=False,
            num_workers=args.workers,
        )
        cross_resolution, _ = evaluate(
            model, high_loader, test_queries, normalization, device
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    torch.save(
        {"model": best_state, "normalization": normalization, "visible": visible},
        checkpoint_path,
    )
    with (args.output_dir / "curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    vector_payload = {}
    for prefix, source in (("selection", selection_vectors), ("test", extrapolation_vectors)):
        for frequency, metrics in source.items():
            for metric, values in metrics.items():
                vector_payload[f"{prefix}_{frequency}_{metric}"] = values
    np.savez_compressed(args.output_dir / "validation_metrics.npz", **vector_payload)
    summary = {
        "schema_version": 1,
        "status": "representation POC; development data only",
        "domain": args.domain,
        "model": args.model,
        "visible_frequencies_hz": visible,
        "selection_query_frequencies_hz": train_queries,
        "test_query_frequencies_hz": test_queries,
        "parameters": parameter_count(model),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "selected_epoch": best_epoch,
        "selection": selection,
        "extrapolation": extrapolation,
        "cross_resolution_extrapolation": cross_resolution,
        "learned_delays_seconds": (
            model.delays.detach().cpu().tolist()
            if isinstance(model, ComplexDelayOperator)
            else None
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "initial_checkpoint_load": checkpoint_load,
        "lineage": bundle.lineage,
        "elapsed_seconds": time.perf_counter() - started,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("synthetic", "ocean"), required=True)
    parser.add_argument(
        "--model",
        choices=(
            "image_unet",
            "complex_delay",
            "stable_recurrence",
            "carrier_poles",
            "component_generator",
            "carrier_ode",
            "local_span",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--visible", required=True)
    parser.add_argument("--train-queries", required=True)
    parser.add_argument("--test-queries", required=True)
    parser.add_argument("--field-only", action="store_true")
    parser.add_argument("--grouped-training", action="store_true")
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--poles", type=int, default=4)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--complex-width", type=int, default=24)
    parser.add_argument("--state-width", type=int, default=64)
    parser.add_argument("--stability-radius", type=float, default=1.0)
    parser.add_argument("--history-weight", type=float, default=0.0)
    parser.add_argument("--relative-loss-power", type=int, choices=(1, 2), default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--max-delay", type=float, default=0.65)
    parser.add_argument("--max-growth-per-hz", type=float, default=0.025)
    parser.add_argument("--max-residual-angle-per-hz", type=float, default=math.pi)
    parser.add_argument("--max-curvature-per-hz2", type=float, default=0.0)
    parser.add_argument("--residue-scale", type=float, default=1.0)
    parser.add_argument("--state-backbone", choices=("dilated", "unet"), default="dilated")
    parser.add_argument(
        "--residue-mode", choices=("predicted", "least_squares"), default="predicted"
    )
    parser.add_argument("--residue-ridge", type=float, default=1e-3)
    parser.add_argument("--pole-growth", choices=("signed", "stable"), default="signed")
    parser.add_argument("--learn-carrier-correction", action="store_true")
    parser.add_argument("--max-carrier-correction-per-hz", type=float, default=0.15)
    parser.add_argument("--state-normalization", action="store_true")
    parser.add_argument(
        "--pole-anchor",
        choices=("reference_residual", "residue_sum"),
        default="reference_residual",
    )
    parser.add_argument(
        "--carrier-estimator",
        choices=("band_ols", "endpoint_increment"),
        default="band_ols",
    )
    parser.add_argument(
        "--generator-carrier",
        choices=("none", "endpoint", "coherent_endpoint"),
        default="endpoint",
    )
    parser.add_argument("--carrier-kernel", type=int, default=5)
    parser.add_argument("--parameter-smoothing", type=int, default=1)
    parser.add_argument("--max-update-rate", type=float, default=1.0)
    parser.add_argument("--global-latent-channels", type=int, default=0)
    parser.add_argument("--hyper-carrier-correction", action="store_true")
    parser.add_argument("--max-hyper-slope-per-hz", type=float, default=0.08)
    parser.add_argument("--max-hyper-curvature-per-hz2", type=float, default=0.005)
    parser.add_argument("--local-carrier-token-grid", type=int, nargs=2)
    parser.add_argument("--local-carrier-token-width", type=int, default=32)
    parser.add_argument("--max-local-token-slope-per-hz", type=float, default=0.08)
    parser.add_argument("--max-local-token-curvature-per-hz2", type=float, default=0.005)
    parser.add_argument("--ode-teacher-forcing", action="store_true")
    parser.add_argument("--token-grid", type=int, nargs=2, default=(4, 8))
    parser.add_argument("--token-frequency-scale", type=float, default=10.0)
    parser.add_argument("--token-weight-bound", type=float, default=2.0)
    parser.add_argument("--token-carrier", choices=("raw", "endpoint"), default="raw")
    parser.add_argument(
        "--mask-pretraining", choices=("frequency", "joint"), default="frequency"
    )
    parser.add_argument("--mask-weight", type=float, default=0.0)
    parser.add_argument("--joint-mask-fraction", type=float, default=0.5)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--allow-partial-initial-checkpoint", action="store_true")
    parser.add_argument("--freeze-initial-backbone", action="store_true")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=20260915)
    parser.add_argument("--synthetic-train", type=int, default=512)
    parser.add_argument("--synthetic-validation", type=int, default=128)
    parser.add_argument("--synthetic-height", type=int, default=32)
    parser.add_argument("--synthetic-width", type=int, default=64)
    parser.add_argument("--cross-resolution-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
