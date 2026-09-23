"""Oracle audit for multi-branch local phase-space wave representations.

The audit deliberately precedes a learned model.  It asks whether several
locally fitted complex plane-wave atoms predict held-out pixels better than a
single atom.  Atom selection and coefficient fitting use only a fixed subset
of pixels in each patch, so adding branches is not evaluated on its fitting
points.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass(frozen=True)
class DictionaryConfig:
    patch_size: int
    frequency_grid_size: int
    max_cycles_per_pixel: float
    train_fraction: float
    branch_counts: tuple[int, ...]
    nms_bins: int
    ridge: float


@dataclass(frozen=True)
class OMPResult:
    errors: dict[int, np.ndarray]
    selected_indices: np.ndarray
    coefficients: np.ndarray
    multimodality: np.ndarray


def complex_dictionary(
    patch_size: int,
    frequency_grid_size: int,
    max_cycles_per_pixel: float,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complex plane-wave atoms and their ``(ky, kx)`` coordinates."""

    spatial = torch.arange(patch_size, dtype=torch.float32, device=device)
    spatial = spatial - 0.5 * (patch_size - 1)
    yy, xx = torch.meshgrid(spatial, spatial, indexing="ij")
    frequency = torch.linspace(
        -max_cycles_per_pixel,
        max_cycles_per_pixel,
        frequency_grid_size,
        dtype=torch.float32,
        device=device,
    )
    ky, kx = torch.meshgrid(frequency, frequency, indexing="ij")
    wavevectors = torch.stack((ky.reshape(-1), kx.reshape(-1)), dim=1)
    phase = 2.0 * torch.pi * (
        wavevectors[:, 0, None] * yy.reshape(1, -1)
        + wavevectors[:, 1, None] * xx.reshape(1, -1)
    )
    dictionary = torch.exp(1j * phase).to(torch.complex64)
    return dictionary, wavevectors


def heldout_mask(
    patch_size: int, train_fraction: float, *, seed: int, device: torch.device
) -> torch.Tensor:
    """Create one deterministic, spatially dispersed train mask."""

    generator = np.random.default_rng(seed)
    count = patch_size * patch_size
    order = generator.permutation(count)
    train_count = round(train_fraction * count)
    mask = torch.zeros(count, dtype=torch.bool, device=device)
    mask[torch.as_tensor(order[:train_count], device=device)] = True
    return mask


def _solve_coefficients(
    atoms: torch.Tensor,
    values: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Batched ridge solve for complex coefficients.

    ``atoms`` is ``[batch, pixels, branches]`` and ``values`` is
    ``[batch, pixels]``.
    """

    adjoint = atoms.conj().transpose(1, 2)
    gram = adjoint @ atoms
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    gram = gram + ridge * identity[None]
    right = (adjoint @ values[:, :, None]).squeeze(-1)
    return torch.linalg.solve(gram, right)


def sparse_plane_wave_fit(
    patches: torch.Tensor,
    config: DictionaryConfig,
    *,
    device: torch.device,
    mask_seed: int = 20260923,
) -> OMPResult:
    """Fit local branches with batched orthogonal matching pursuit."""

    if patches.ndim != 3 or patches.shape[-2:] != (config.patch_size, config.patch_size):
        raise ValueError("patches do not match configured patch size")
    maximum_branches = max(config.branch_counts)
    dictionary, wavevectors = complex_dictionary(
        config.patch_size,
        config.frequency_grid_size,
        config.max_cycles_per_pixel,
        device=device,
    )
    train_mask = heldout_mask(
        config.patch_size, config.train_fraction, seed=mask_seed, device=device
    )
    validation_mask = ~train_mask
    train_dictionary = dictionary[:, train_mask]
    validation_dictionary = dictionary[:, validation_mask]
    train_norm = torch.sqrt(train_dictionary.abs().square().sum(dim=1)).clamp_min(1e-8)

    flattened = patches.to(device).reshape(len(patches), -1).to(torch.complex64)
    patch_rms = torch.sqrt(flattened.abs().square().mean(dim=1, keepdim=True)).clamp_min(1e-8)
    flattened = flattened / patch_rms
    train_values = flattened[:, train_mask]
    validation_values = flattened[:, validation_mask]

    residual = train_values.clone()
    blocked = torch.zeros(
        len(patches), dictionary.shape[0], dtype=torch.bool, device=device
    )
    selected_atoms: list[torch.Tensor] = []
    selected_indices: list[torch.Tensor] = []
    errors: dict[int, np.ndarray] = {}
    coefficient_by_step: list[torch.Tensor] = []
    frequency_step = 2.0 * config.max_cycles_per_pixel / (config.frequency_grid_size - 1)
    suppression_radius = (config.nms_bins + 0.25) * frequency_step

    for branch in range(1, maximum_branches + 1):
        correlation = torch.abs(residual @ train_dictionary.conj().transpose(0, 1))
        correlation = correlation / train_norm[None]
        correlation = correlation.masked_fill(blocked, -torch.inf)
        choice = torch.argmax(correlation, dim=1)
        selected_indices.append(choice)
        selected_atoms.append(train_dictionary[choice])
        selected = torch.stack(selected_atoms, dim=2)
        coefficients = _solve_coefficients(selected, train_values, config.ridge)
        coefficient_by_step.append(coefficients)
        residual = train_values - torch.einsum("bpk,bk->bp", selected, coefficients)

        chosen_wavevectors = wavevectors[choice]
        distance = torch.linalg.vector_norm(
            wavevectors[None] - chosen_wavevectors[:, None], dim=2
        )
        blocked |= distance <= suppression_radius

        if branch in config.branch_counts:
            validation_atoms = torch.stack(
                [validation_dictionary[index] for index in selected_indices], dim=2
            )
            prediction = torch.einsum("bpk,bk->bp", validation_atoms, coefficients)
            numerator = (prediction - validation_values).abs().square().sum(dim=1)
            denominator = validation_values.abs().square().sum(dim=1).clamp_min(1e-10)
            errors[branch] = torch.sqrt(numerator / denominator).cpu().numpy()

    final_coefficients = coefficient_by_step[-1]
    energy = final_coefficients.abs().square()
    multimodality = 1.0 - energy.max(dim=1).values / energy.sum(dim=1).clamp_min(1e-10)
    return OMPResult(
        errors=errors,
        selected_indices=torch.stack(selected_indices, dim=1).cpu().numpy(),
        coefficients=final_coefficients.cpu().numpy(),
        multimodality=multimodality.cpu().numpy(),
    )


def _sample_separated_wavevectors(
    generator: np.random.Generator, count: int
) -> np.ndarray:
    selected: list[np.ndarray] = []
    attempts = 0
    while len(selected) < count:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not sample separated wavevectors")
        radius = generator.uniform(0.10, 0.34)
        angle = generator.uniform(-np.pi, np.pi)
        candidate = radius * np.asarray([np.sin(angle), np.cos(angle)])
        if all(np.linalg.norm(candidate - previous) >= 0.10 for previous in selected):
            selected.append(candidate)
    return np.stack(selected)


def controlled_mixtures(
    *,
    samples: int,
    patch_size: int,
    branch_count: int,
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate off-grid coherent mixtures with mild amplitude modulation."""

    generator = np.random.default_rng(seed)
    spatial = np.arange(patch_size) - 0.5 * (patch_size - 1)
    yy, xx = np.meshgrid(spatial, spatial, indexing="ij")
    patches = []
    truth = []
    for _ in range(samples):
        wavevectors = _sample_separated_wavevectors(generator, branch_count)
        value = np.zeros((patch_size, patch_size), dtype=np.complex128)
        for ky, kx in wavevectors:
            amplitude = generator.uniform(0.45, 1.0)
            phase0 = generator.uniform(-np.pi, np.pi)
            slope_y, slope_x = generator.normal(0.0, 0.035, size=2)
            envelope = np.exp((slope_y * yy + slope_x * xx) / patch_size)
            value += amplitude * envelope * np.exp(
                1j * (2.0 * np.pi * (ky * yy + kx * xx) + phase0)
            )
        noise_rms = np.sqrt(np.mean(np.abs(value) ** 2)) * 10.0 ** (-snr_db / 20.0)
        noise = (generator.normal(size=value.shape) + 1j * generator.normal(size=value.shape))
        value += noise_rms * noise / np.sqrt(2.0)
        patches.append(value.astype(np.complex64))
        truth.append(wavevectors.astype(np.float32))
    return np.stack(patches), np.stack(truth)


def _assignment_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    count = len(truth)
    return min(
        float(np.mean([np.linalg.norm(prediction[index] - truth[target]) for index, target in enumerate(order)]))
        for order in itertools.permutations(range(count))
    )


def controlled_audit(
    config: dict,
    dictionary: dict,
    *,
    device: torch.device,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    phase_stability = []
    for branch_count in config["branch_counts"]:
        patches, truth = controlled_mixtures(
            samples=int(config["samples_per_branch_count"]),
            patch_size=17,
            branch_count=int(branch_count),
            snr_db=float(config["snr_db"]),
            seed=20260923 + int(branch_count),
        )
        fit_config = DictionaryConfig(
            patch_size=17,
            frequency_grid_size=int(dictionary["frequency_grid_size"]),
            max_cycles_per_pixel=float(dictionary["max_cycles_per_pixel"]),
            train_fraction=float(dictionary["train_fraction"]),
            branch_counts=tuple(int(value) for value in dictionary["branch_counts"]),
            nms_bins=int(dictionary["nms_bins"]),
            ridge=float(dictionary["ridge"]),
        )
        result = sparse_plane_wave_fit(
            torch.from_numpy(patches), fit_config, device=device
        )
        _, wavevectors = complex_dictionary(
            fit_config.patch_size,
            fit_config.frequency_grid_size,
            fit_config.max_cycles_per_pixel,
            device=device,
        )
        selected_wavevectors = wavevectors[result.selected_indices[:, :branch_count]].cpu().numpy()
        branch_error = np.asarray(
            [
                _assignment_error(prediction, target)
                for prediction, target in zip(selected_wavevectors, truth, strict=True)
            ]
        )
        for model_branches, values in result.errors.items():
            records.append(
                {
                    "condition": "controlled",
                    "truth_branches": int(branch_count),
                    "model_branches": int(model_branches),
                    "mean_complex_rel_l2": float(np.mean(values)),
                    "median_complex_rel_l2": float(np.median(values)),
                    "mean_wavevector_error": (
                        float(branch_error.mean()) if model_branches == branch_count else ""
                    ),
                    "patches": len(values),
                }
            )

        rotation = np.exp(1j * 1.2345).astype(np.complex64)
        rotated = sparse_plane_wave_fit(
            torch.from_numpy(patches * rotation), fit_config, device=device
        )
        phase_stability.append(
            float(np.mean(rotated.selected_indices == result.selected_indices))
        )
    summary = {
        "mean_selected_index_stability_under_global_phase": float(np.mean(phase_stability)),
        "per_truth_branch_count": phase_stability,
    }
    return records, summary


def extract_random_patches(
    fields: np.ndarray,
    *,
    patch_size: int,
    patches_per_field: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract deterministic uniform patches and return field-local indices."""

    generator = np.random.default_rng(seed)
    radius = patch_size // 2
    height, width = fields.shape[-2:]
    if patch_size % 2 != 1 or patch_size > min(height, width):
        raise ValueError("patch size must be odd and fit inside the field")
    patches = []
    field_indices = []
    for field_index, field in enumerate(fields):
        for _ in range(patches_per_field):
            row = int(generator.integers(radius, height - radius))
            column = int(generator.integers(radius, width - radius))
            patches.append(
                field[
                    row - radius : row + radius + 1,
                    column - radius : column + radius + 1,
                ]
            )
            field_indices.append(field_index)
    return np.stack(patches).astype(np.complex64), np.asarray(field_indices)


def _bootstrap_interval(
    paired_improvement: np.ndarray, *, seed: int, draws: int = 5000
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    count = len(paired_improvement)
    sampled = generator.integers(0, count, size=(draws, count))
    means = paired_improvement[sampled].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def evaluate_patch_collection(
    patches: np.ndarray,
    system_ids: np.ndarray,
    labels: np.ndarray,
    dictionary: dict,
    *,
    patch_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], dict]:
    fit_config = DictionaryConfig(
        patch_size=patch_size,
        frequency_grid_size=int(dictionary["frequency_grid_size"]),
        max_cycles_per_pixel=float(dictionary["max_cycles_per_pixel"]),
        train_fraction=float(dictionary["train_fraction"]),
        branch_counts=tuple(int(value) for value in dictionary["branch_counts"]),
        nms_bins=int(dictionary["nms_bins"]),
        ridge=float(dictionary["ridge"]),
    )
    error_parts = {count: [] for count in fit_config.branch_counts}
    multimodality_parts = []
    for start in range(0, len(patches), batch_size):
        result = sparse_plane_wave_fit(
            torch.from_numpy(patches[start : start + batch_size]),
            fit_config,
            device=device,
            mask_seed=20260923 + patch_size,
        )
        for count in fit_config.branch_counts:
            error_parts[count].append(result.errors[count])
        multimodality_parts.append(result.multimodality)
    errors = {count: np.concatenate(parts) for count, parts in error_parts.items()}
    multimodality = np.concatenate(multimodality_parts)

    patch_records = []
    for index in range(len(patches)):
        record = {
            "system_id": int(system_ids[index]),
            "label": str(labels[index]),
            "multimodality": float(multimodality[index]),
        }
        for count in fit_config.branch_counts:
            record[f"error_k{count}"] = float(errors[count][index])
        patch_records.append(record)

    system_records = []
    string_labels = labels.astype(str)
    for label in sorted({str(value) for value in labels}):
        for system_id in sorted(set(system_ids[string_labels == label].tolist())):
            keep = (system_ids == system_id) & (string_labels == label)
            record = {"system_id": int(system_id), "label": label, "patches": int(keep.sum())}
            for count in fit_config.branch_counts:
                record[f"error_k{count}"] = float(errors[count][keep].mean())
            record["multimodality"] = float(multimodality[keep].mean())
            system_records.append(record)

    summaries = {}
    first = min(fit_config.branch_counts)
    last = max(fit_config.branch_counts)
    for label in sorted({record["label"] for record in system_records}):
        subset = [record for record in system_records if record["label"] == label]
        baseline = np.asarray([record[f"error_k{first}"] for record in subset])
        improved = np.asarray([record[f"error_k{last}"] for record in subset])
        paired = (baseline - improved) / np.maximum(baseline, 1e-12)
        interval = _bootstrap_interval(paired, seed=20260923 + len(label))
        summaries[label] = {
            "systems": len(subset),
            "patches": int(sum(record["patches"] for record in subset)),
            **{
                f"mean_error_k{count}": float(
                    np.mean([record[f"error_k{count}"] for record in subset])
                )
                for count in fit_config.branch_counts
            },
            f"paired_relative_improvement_k{last}_vs_k{first}": float(paired.mean()),
            "paired_improvement_95ci": list(interval),
            "mean_multimodality": float(
                np.mean([record["multimodality"] for record in subset])
            ),
        }

    gain = (errors[first] - errors[last]) / np.maximum(errors[first], 1e-12)
    correlation = float(np.corrcoef(multimodality, gain)[0, 1])
    summaries["mechanism"] = {
        "patch_multimodality_gain_correlation": correlation,
    }
    return patch_records, system_records, summaries


def load_real_collections(config: dict) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    real = config["real_fields"]
    output = {}
    for name in ("iid", "boundary"):
        with np.load(config["datasets"][name], allow_pickle=False) as data:
            # NPZ members are compressed streams.  Materialize each member once;
            # repeatedly indexing ``data["field"]`` would decompress the full
            # array for every selected view.
            fields = data["field"]
            if name == "iid":
                split = data["split"]
                available = np.flatnonzero(split == "test")
                selected_systems = available[: int(real["max_iid_systems"])]
                labels_by_system = np.asarray(["iid"] * len(selected_systems))
                patch_size = int(real["iid_patch_size"])
            else:
                selected_systems = np.arange(
                    min(int(real["max_boundary_systems"]), len(data["field"]))
                )
                labels_by_system = data["boundary"][selected_systems].astype(str)
                patch_size = int(real["boundary_patch_size"])
            views = []
            view_systems = []
            view_labels = []
            for local_index, system_index in enumerate(selected_systems):
                for source in real["source_indices"]:
                    for frequency in real["frequency_indices"]:
                        views.append(fields[system_index, source, frequency])
                        view_systems.append(int(system_index))
                        view_labels.append(str(labels_by_system[local_index]))
            patches, field_indices = extract_random_patches(
                np.stack(views),
                patch_size=patch_size,
                patches_per_field=int(real["patches_per_view"]),
                seed=20260923 + patch_size,
            )
            systems = np.asarray(view_systems)[field_indices]
            labels = np.asarray(view_labels)[field_indices]
            output[name] = (patches, systems, labels)
    return output


def _write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(records[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def plot_summary(controlled: list[dict], real_summary: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    branch_counts = sorted({record["truth_branches"] for record in controlled})
    for truth in branch_counts:
        subset = [record for record in controlled if record["truth_branches"] == truth]
        axes[0].plot(
            [record["model_branches"] for record in subset],
            [record["mean_complex_rel_l2"] for record in subset],
            marker="o",
            label=f"truth M={truth}",
        )
    axes[0].set_xlabel("Fitted branches K")
    axes[0].set_ylabel("Held-out complex RelL2")
    axes[0].set_title("Controlled coherent mixtures")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    labels = [label for label in ("iid", "open", "partial", "closed") if label in real_summary]
    x = np.arange(len(labels))
    widths = 0.24
    colors = ("#64748b", "#0ea5e9", "#f97316")
    for position, branches in enumerate((1, 2, 4)):
        axes[1].bar(
            x + (position - 1) * widths,
            [real_summary[label][f"mean_error_k{branches}"] for label in labels],
            width=widths,
            color=colors[position],
            label=f"K={branches}",
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Held-out complex RelL2")
    axes[1].set_title("Real simulated wave patches")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    controlled_records, controlled_summary = controlled_audit(
        config["controlled"], config["dictionary"], device=device
    )
    _write_csv(output / "controlled.csv", controlled_records)

    collections = load_real_collections(config)
    all_patch_records = []
    all_system_records = []
    real_summary = {}
    real_mechanisms = {}
    for name, (patches, systems, labels) in collections.items():
        patch_size = int(config["real_fields"][f"{name}_patch_size"])
        patch_records, system_records, summary = evaluate_patch_collection(
            patches,
            systems,
            labels,
            config["dictionary"],
            patch_size=patch_size,
            batch_size=int(config["dictionary"]["batch_size"]),
            device=device,
        )
        for record in patch_records:
            record["dataset"] = name
        for record in system_records:
            record["dataset"] = name
        all_patch_records.extend(patch_records)
        all_system_records.extend(system_records)
        real_mechanisms[name] = summary.pop("mechanism")
        real_summary.update(summary)
    _write_csv(output / "patch_records.csv", all_patch_records)
    _write_csv(output / "system_records.csv", all_system_records)

    primary_labels = [label for label in ("partial", "closed") if label in real_summary]
    passed_labels = []
    for label in primary_labels:
        entry = real_summary[label]
        if (
            entry["paired_relative_improvement_k4_vs_k1"] >= 0.15
            and entry["paired_improvement_95ci"][0] > 0.0
        ):
            passed_labels.append(label)
    summary = {
        "config": config,
        "device": str(device),
        "controlled": controlled_summary,
        "real": real_summary,
        "real_mechanisms": real_mechanisms,
        "oracle_gate": {
            "passed": bool(passed_labels),
            "passing_labels": passed_labels,
            "required": "at least one of partial/closed has >=15% paired K4-vs-K1 gain and positive 95% CI",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_summary(controlled_records, real_summary, output / "mps0_oracle")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
