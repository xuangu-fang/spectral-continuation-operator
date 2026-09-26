"""Generate a system-held-out structural lattice wave-response benchmark.

The simulator solves the damped frequency-domain mass--spring equation

    (K - omega^2 M + i omega C) u = f

on a heterogeneous square lattice.  It is intentionally independent of the
acoustic time-domain and normal-mode solvers used by the other Q3 domains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cross_domain_response_data import (
    CrossDomainResponseArchive,
    save_archive,
)


def perimeter_nodes(grid: int) -> np.ndarray:
    nodes = list(range(grid))
    for row in range(1, grid):
        nodes.append(row * grid + grid - 1)
    for column in range(grid - 2, -1, -1):
        nodes.append((grid - 1) * grid + column)
    for row in range(grid - 2, 0, -1):
        nodes.append(row * grid)
    return np.asarray(nodes, dtype=np.int64)


def correlated_properties(
    count: int, grid: int, seed: int, ood: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, grid),
        np.linspace(-1.0, 1.0, grid),
        indexing="ij",
    )
    density = np.empty((count, grid, grid), dtype=np.float64)
    stiffness = np.empty_like(density)
    anisotropy = np.empty((count, 2), dtype=np.float64)
    for sample in range(count):
        phase = rng.uniform(0.0, 2.0 * np.pi, size=6)
        smooth_mass = (
            np.sin(np.pi * x + phase[0])
            + 0.7 * np.cos(np.pi * y + phase[1])
            + 0.35 * np.sin(np.pi * (x + y) + phase[2])
        )
        smooth_spring = (
            np.cos(np.pi * x + phase[3])
            + 0.7 * np.sin(np.pi * y + phase[4])
            + 0.35 * np.cos(np.pi * (x - y) + phase[5])
        )
        mass = np.exp(0.10 * smooth_mass)
        spring = np.exp(0.22 * smooth_spring)
        defect_count = int(rng.integers(2, 5) if ood else rng.integers(0, 3))
        for _ in range(defect_count):
            center_x, center_y = rng.uniform(-0.65, 0.65, size=2)
            radius = rng.uniform(0.12, 0.28 if ood else 0.22)
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
            if ood:
                spring[mask] *= rng.uniform(0.10, 0.40)
                mass[mask] *= rng.uniform(1.25, 1.80)
            else:
                spring[mask] *= rng.uniform(0.45, 0.80)
                mass[mask] *= rng.uniform(1.05, 1.35)
        density[sample] = mass
        stiffness[sample] = spring
        if ood:
            anisotropy[sample] = rng.uniform((0.55, 1.15), (0.85, 1.55))
        else:
            anisotropy[sample] = rng.uniform((0.85, 0.85), (1.15, 1.15))
    return density, stiffness, anisotropy


def assemble_matrices(
    density: torch.Tensor,
    stiffness: torch.Tensor,
    anisotropy: torch.Tensor,
    boundary_anchor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, grid, _ = density.shape
    nodes = grid * grid
    mass = torch.diag_embed(density.reshape(batch, nodes))
    spring = torch.zeros(
        (batch, nodes, nodes), dtype=torch.float64, device=density.device
    )
    edges_i = []
    edges_j = []
    directions = []
    for row in range(grid):
        for column in range(grid):
            node = row * grid + column
            if column + 1 < grid:
                edges_i.append(node)
                edges_j.append(node + 1)
                directions.append(0)
            if row + 1 < grid:
                edges_i.append(node)
                edges_j.append(node + grid)
                directions.append(1)
    i = torch.as_tensor(edges_i, dtype=torch.long, device=density.device)
    j = torch.as_tensor(edges_j, dtype=torch.long, device=density.device)
    direction = torch.as_tensor(directions, dtype=torch.long, device=density.device)
    flat = stiffness.reshape(batch, nodes)
    weight = 0.5 * (flat[:, i] + flat[:, j]) * anisotropy[:, direction]
    diagonal = torch.zeros((batch, nodes), dtype=torch.float64, device=density.device)
    diagonal.scatter_add_(1, i[None].expand(batch, -1), weight)
    diagonal.scatter_add_(1, j[None].expand(batch, -1), weight)
    node = torch.arange(nodes, device=density.device)
    spring[:, node, node] = diagonal
    spring[:, i, j] = -weight
    spring[:, j, i] = -weight
    boundary = torch.as_tensor(perimeter_nodes(grid), dtype=torch.long, device=density.device)
    spring[:, boundary, boundary] += boundary_anchor
    return mass, spring


@torch.inference_mode()
def simulate(
    density: np.ndarray,
    stiffness: np.ndarray,
    anisotropy: np.ndarray,
    frequencies: np.ndarray,
    source_nodes: np.ndarray,
    receiver_nodes: np.ndarray,
    damping_mass: float,
    damping_stiffness: float,
    boundary_anchor: float,
    batch_size: int,
    device_name: str,
) -> np.ndarray:
    device = torch.device(device_name)
    systems, grid, _ = density.shape
    responses = np.empty(
        (systems, len(source_nodes), len(frequencies), len(receiver_nodes)),
        dtype=np.complex64,
    )
    source = torch.zeros(
        (grid * grid, len(source_nodes)), dtype=torch.complex128, device=device
    )
    source[
        torch.as_tensor(source_nodes, dtype=torch.long, device=device),
        torch.arange(len(source_nodes), device=device),
    ] = 1.0
    receiver = torch.as_tensor(receiver_nodes, dtype=torch.long, device=device)
    identity_frequency = torch.as_tensor(frequencies, dtype=torch.float64, device=device)
    for start in range(0, systems, batch_size):
        stop = min(start + batch_size, systems)
        mass, spring = assemble_matrices(
            torch.as_tensor(density[start:stop], device=device),
            torch.as_tensor(stiffness[start:stop], device=device),
            torch.as_tensor(anisotropy[start:stop], device=device),
            boundary_anchor,
        )
        damping = damping_mass * mass + damping_stiffness * spring
        frequency_response = []
        for omega in identity_frequency:
            dynamic = (
                spring
                - omega.square() * mass
                + 1j * omega * damping
            ).to(torch.complex128)
            displacement = torch.linalg.solve(
                dynamic, source[None].expand(stop - start, -1, -1)
            )
            frequency_response.append(displacement[:, receiver].permute(0, 2, 1))
        responses[start:stop] = (
            torch.stack(frequency_response, dim=2).cpu().numpy().astype(np.complex64)
        )
    return responses


def run(config_path: Path, output: Path, device: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    grid = int(config["grid"])
    counts = config["splits"]
    frequencies = np.asarray(config["angular_frequencies"], dtype=np.float64)
    perimeter = perimeter_nodes(grid)
    receiver_positions = np.linspace(
        0, len(perimeter), int(config["receivers"]), endpoint=False, dtype=np.int64
    )
    receiver_nodes = perimeter[receiver_positions]
    source_receiver_index = np.linspace(
        0, len(receiver_nodes), int(config["sources"]), endpoint=False, dtype=np.int64
    )
    source_nodes = receiver_nodes[source_receiver_index]
    all_responses = []
    all_splits = []
    all_ids = []
    property_summary = {}
    seed = int(config["seed"])
    for split_index, (split, count) in enumerate(counts.items()):
        ood = split == "ood"
        density, stiffness, anisotropy = correlated_properties(
            int(count), grid, seed + 10_003 * split_index, ood
        )
        response = simulate(
            density,
            stiffness,
            anisotropy,
            frequencies,
            source_nodes,
            receiver_nodes,
            float(config["damping_mass"]),
            float(config["damping_stiffness"]),
            float(config["boundary_anchor"]),
            int(config["batch_size"]),
            device,
        )
        all_responses.append(response)
        all_splits.extend([split] * int(count))
        all_ids.extend([f"structural_lattice_{split}_{index:04d}" for index in range(int(count))])
        property_summary[split] = {
            "density_range": [float(density.min()), float(density.max())],
            "stiffness_range": [float(stiffness.min()), float(stiffness.max())],
            "anisotropy_range": [float(anisotropy.min()), float(anisotropy.max())],
        }
    coordinates = np.stack(np.unravel_index(np.arange(grid * grid), (grid, grid)), axis=1)
    coordinates = coordinates[:, ::-1].astype(np.float32) / float(grid - 1)
    archive = CrossDomainResponseArchive(
        responses=np.concatenate(all_responses),
        sources=coordinates[source_nodes],
        receivers=coordinates[receiver_nodes],
        frequencies=frequencies.astype(np.float32),
        source_receiver_index=source_receiver_index,
        split=np.asarray(all_splits),
        system_ids=np.asarray(all_ids),
        valid_response_mask=np.ones((len(source_nodes), len(receiver_nodes)), dtype=bool),
        provenance={
            "domain": "heterogeneous damped structural lattice waves",
            "equation": "(K - omega^2 M + i omega (alpha M + beta K)) u = f",
            "solver": "batched dense complex128 direct solve",
            "grid": grid,
            "source_nodes": source_nodes.tolist(),
            "receiver_nodes": receiver_nodes.tolist(),
            "property_summary": property_summary,
            "config": config,
            "evidence_role": "independent structural-wave strict-Q3 validation domain",
        },
    )
    manifest = save_archive(output, archive)
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
