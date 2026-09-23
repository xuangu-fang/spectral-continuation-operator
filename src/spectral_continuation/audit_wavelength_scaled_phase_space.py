"""Wavelength-scaled follow-up to the MPS-0 multi-branch oracle audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .audit_multibranch_phase_space import (
    _bootstrap_interval,
    evaluate_patch_collection,
    extract_random_patches,
)


def wavelength_window_pixels(
    frequency: float,
    spacing: float,
    *,
    reference_speed: float,
    wavelengths: float,
    minimum: int,
    maximum: int,
) -> int:
    """Choose an odd window spanning a fixed number of reference wavelengths."""

    raw = wavelengths * reference_speed / (frequency * spacing)
    clipped = int(np.clip(round(raw), minimum, maximum))
    if clipped % 2 == 0:
        clipped += 1 if clipped < maximum else -1
    return clipped


def _write_csv(path: Path, records: list[dict]) -> None:
    keys = list(records[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _load_selected_views(
    path: str,
    *,
    dataset: str,
    max_systems: int,
    source_indices: list[int],
    frequency_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    with np.load(path, allow_pickle=False) as data:
        fields = data["field"]
        frequencies = data["frequencies"]
        spacing = float(data["spacing"]) if "spacing" in data.files else 1.0 / 127.0
        if dataset == "iid":
            available = np.flatnonzero(data["split"] == "test")[:max_systems]
            labels = np.asarray(["iid"] * len(available))
        else:
            available = np.arange(min(max_systems, len(fields)))
            labels = data["boundary"][available].astype(str)
        views = []
        view_systems = []
        view_labels = []
        for local_index, system_index in enumerate(available):
            for source_index in source_indices:
                views.append(fields[system_index, source_index, frequency_index])
                view_systems.append(int(system_index))
                view_labels.append(str(labels[local_index]))
    return (
        np.stack(views),
        np.asarray(view_systems),
        np.asarray(view_labels),
        float(frequencies[frequency_index]),
        spacing,
    )


def _aggregate_system_records(
    records: list[dict], branch_counts: list[int]
) -> tuple[list[dict], dict]:
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record["dataset"], record["label"], record["seed"], record["system_id"])
        grouped.setdefault(key, []).append(record)
    aggregate = []
    for (dataset, label, seed, system_id), values in grouped.items():
        row = {
            "dataset": dataset,
            "label": label,
            "seed": seed,
            "system_id": system_id,
            "frequency_groups": len(values),
            "multimodality": float(np.mean([value["multimodality"] for value in values])),
        }
        for branches in branch_counts:
            row[f"error_k{branches}"] = float(
                np.mean([value[f"error_k{branches}"] for value in values])
            )
        aggregate.append(row)

    first, last = min(branch_counts), max(branch_counts)
    summary = {}
    for dataset, label in sorted({(row["dataset"], row["label"]) for row in aggregate}):
        subset = [
            row for row in aggregate if row["dataset"] == dataset and row["label"] == label
        ]
        # Average the repeated patch-mask seeds for each physical system before
        # bootstrapping.  The statistical unit remains the system, not a patch.
        system_ids = sorted({row["system_id"] for row in subset})
        per_system = []
        for system_id in system_ids:
            repeated = [row for row in subset if row["system_id"] == system_id]
            entry = {"system_id": system_id}
            for branches in branch_counts:
                entry[f"error_k{branches}"] = float(
                    np.mean([row[f"error_k{branches}"] for row in repeated])
                )
            entry["multimodality"] = float(
                np.mean([row["multimodality"] for row in repeated])
            )
            per_system.append(entry)
        baseline = np.asarray([row[f"error_k{first}"] for row in per_system])
        improved = np.asarray([row[f"error_k{last}"] for row in per_system])
        paired = (baseline - improved) / np.maximum(baseline, 1e-12)
        summary[label] = {
            "dataset": dataset,
            "systems": len(per_system),
            "patch_seeds": len({row["seed"] for row in subset}),
            **{
                f"mean_error_k{branches}": float(
                    np.mean([row[f"error_k{branches}"] for row in per_system])
                )
                for branches in branch_counts
            },
            f"paired_relative_improvement_k{last}_vs_k{first}": float(paired.mean()),
            "paired_improvement_95ci": list(
                _bootstrap_interval(paired, seed=20260923 + len(label))
            ),
            "mean_multimodality": float(
                np.mean([row["multimodality"] for row in per_system])
            ),
        }
    return aggregate, summary


def run_adaptive(config: dict, device: torch.device) -> tuple[list[dict], dict, list[dict]]:
    sampling = config["sampling"]
    window = config["window"]
    dictionary = config["dictionary"]
    all_system_records = []
    frequency_records = []
    for dataset in ("iid", "boundary"):
        maximum = int(sampling[f"max_{dataset}_systems"])
        for frequency_index in sampling["frequency_indices"]:
            views, view_systems, view_labels, frequency, spacing = _load_selected_views(
                config["datasets"][dataset],
                dataset=dataset,
                max_systems=maximum,
                source_indices=sampling["source_indices"],
                frequency_index=int(frequency_index),
            )
            patch_size = wavelength_window_pixels(
                frequency,
                spacing,
                reference_speed=float(window["reference_speed"]),
                wavelengths=float(window["wavelengths"]),
                minimum=int(window["minimum_pixels"]),
                maximum=int(window["maximum_pixels"]),
            )
            for seed in sampling["patch_seeds"]:
                patches, field_indices = extract_random_patches(
                    views,
                    patch_size=patch_size,
                    patches_per_field=int(sampling["patches_per_view"]),
                    seed=int(seed) + 101 * int(frequency_index) + patch_size,
                )
                _, system_records, summary = evaluate_patch_collection(
                    patches,
                    view_systems[field_indices],
                    view_labels[field_indices],
                    dictionary,
                    patch_size=patch_size,
                    batch_size=int(dictionary["batch_size"]),
                    device=device,
                )
                for record in system_records:
                    record.update(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "frequency_index": int(frequency_index),
                            "frequency": frequency,
                            "patch_size": patch_size,
                        }
                    )
                all_system_records.extend(system_records)
                for label, values in summary.items():
                    if label == "mechanism":
                        continue
                    frequency_records.append(
                        {
                            "dataset": dataset,
                            "label": label,
                            "seed": int(seed),
                            "frequency_index": int(frequency_index),
                            "frequency": frequency,
                            "patch_size": patch_size,
                            **values,
                        }
                    )
    aggregate, summary = _aggregate_system_records(
        all_system_records, [int(value) for value in dictionary["branch_counts"]]
    )
    return aggregate, summary, frequency_records


def run_fixed_window_sensitivity(config: dict, device: torch.device) -> list[dict]:
    sampling = config["sampling"]
    dictionary = config["dictionary"]
    records = []
    loaded = []
    for frequency_index in sampling["frequency_indices"]:
        loaded.append(
            _load_selected_views(
                config["datasets"]["boundary"],
                dataset="boundary",
                max_systems=int(sampling["max_boundary_systems"]),
                source_indices=sampling["source_indices"],
                frequency_index=int(frequency_index),
            )
        )
    views = np.concatenate([value[0] for value in loaded], axis=0)
    systems = np.concatenate([value[1] for value in loaded])
    labels = np.concatenate([value[2] for value in loaded])
    for patch_size in config["window"]["fixed_boundary_sensitivity_pixels"]:
        patches, field_indices = extract_random_patches(
            views,
            patch_size=int(patch_size),
            patches_per_field=max(2, int(sampling["patches_per_view"]) // 2),
            seed=20261000 + int(patch_size),
        )
        _, _, summary = evaluate_patch_collection(
            patches,
            systems[field_indices],
            labels[field_indices],
            dictionary,
            patch_size=int(patch_size),
            batch_size=int(dictionary["batch_size"]),
            device=device,
        )
        for label, values in summary.items():
            if label == "mechanism":
                continue
            records.append({"patch_size": int(patch_size), "label": label, **values})
    return records


def plot_results(summary: dict, sensitivity: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    labels = [label for label in ("iid", "open", "partial", "closed") if label in summary]
    x = np.arange(len(labels))
    width = 0.24
    for offset, (branches, color) in enumerate(zip((1, 2, 4), ("#64748b", "#0ea5e9", "#f97316"))):
        axes[0].bar(
            x + (offset - 1) * width,
            [summary[label][f"mean_error_k{branches}"] for label in labels],
            width=width,
            color=color,
            label=f"K={branches}",
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Held-out complex RelL2")
    axes[0].set_title("Wavelength-scaled windows")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for label, color in zip(("open", "partial", "closed"), ("#16a34a", "#f59e0b", "#dc2626")):
        subset = sorted(
            [record for record in sensitivity if record["label"] == label],
            key=lambda value: value["patch_size"],
        )
        axes[1].plot(
            [record["patch_size"] for record in subset],
            [record["paired_relative_improvement_k4_vs_k1"] * 100.0 for record in subset],
            marker="o",
            color=color,
            label=label,
        )
    axes[1].axhline(15.0, color="black", linestyle="--", linewidth=1, label="15% gate")
    axes[1].set_xlabel("Fixed window size (pixels)")
    axes[1].set_ylabel("K=4 improvement over K=1 (%)")
    axes[1].set_title("Locality is scale dependent")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
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

    system_records, summary, frequency_records = run_adaptive(config, device)
    sensitivity = run_fixed_window_sensitivity(config, device)
    _write_csv(output / "system_records.csv", system_records)
    _write_csv(output / "frequency_records.csv", frequency_records)
    _write_csv(output / "fixed_window_sensitivity.csv", sensitivity)

    passed = []
    for label in ("partial", "closed"):
        result = summary[label]
        if (
            result["paired_relative_improvement_k4_vs_k1"] >= 0.15
            and result["paired_improvement_95ci"][0] > 0.0
        ):
            passed.append(label)
    output_summary = {
        "config": config,
        "adaptive": summary,
        "promotion_gate": {
            "passed": bool(passed),
            "passing_labels": passed,
            "required": ">=15% K4-vs-K1 system-paired gain on partial or closed with positive 95% CI",
        },
        "claim_boundary": (
            "Oracle headroom for wavelength-scaled local multi-component atoms; "
            "not evidence that learned or path-semantic tokens are solved."
        ),
    }
    (output / "summary.json").write_text(json.dumps(output_summary, indent=2))
    plot_results(summary, sensitivity, output / "mps0b_wavelength_scaled")
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
