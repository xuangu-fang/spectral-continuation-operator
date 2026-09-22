"""Summarize geometry, noise, resolution, and boundary shifts for M0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPRESENTATIONS = ("raw", "amplitude", "amplitude_soft_phase_gradient")
LABELS = {
    "raw": "Raw complex",
    "amplitude": "Amplitude only",
    "amplitude_soft_phase_gradient": "Amplitude + phase advance",
}
COLORS = {
    "raw": "#5B6770",
    "amplitude": "#E69F00",
    "amplitude_soft_phase_gradient": "#0072B2",
}


def _load(root: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]


def _condition(record: dict) -> str:
    if record["label"] == "iid":
        return f"iid_snr{record['snr_db']:g}"
    return str(record["label"])


def _aggregate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    rows = []
    boundary_rows = []
    for representation in REPRESENTATIONS:
        own = [record for record in records if record["representation"] == representation]
        for condition in ("ellipse", "iid_snr20", "iid_snr10", "legacy128"):
            values = np.asarray(
                [record["contrast_rel_l2_mean"] for record in own if _condition(record) == condition]
            )
            if len(values) != 3:
                raise RuntimeError(f"expected three {representation}/{condition} evaluations")
            rows.append(
                {
                    "representation": representation,
                    "condition": condition,
                    "rel_l2_mean": float(values.mean()),
                    "rel_l2_seed_std": float(values.std()),
                }
            )
        legacy = [record for record in own if _condition(record) == "legacy128"]
        for boundary in ("open", "partial", "closed"):
            values = np.asarray(
                [record["groups"][boundary]["contrast_rel_l2_mean"] for record in legacy]
            )
            boundary_rows.append(
                {
                    "representation": representation,
                    "boundary": boundary,
                    "rel_l2_mean": float(values.mean()),
                    "rel_l2_seed_std": float(values.std()),
                }
            )
    return rows, boundary_rows


def _plot_summary(rows: list[dict], boundary_rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    conditions = ("ellipse", "iid_snr20", "iid_snr10")
    x = np.arange(len(conditions))
    width = 0.24
    for index, representation in enumerate(REPRESENTATIONS):
        own = {
            row["condition"]: row for row in rows if row["representation"] == representation
        }
        axes[0].bar(
            x + (index - 1) * width,
            [own[condition]["rel_l2_mean"] for condition in conditions],
            width,
            yerr=[own[condition]["rel_l2_seed_std"] for condition in conditions],
            color=COLORS[representation],
            capsize=3,
            label=LABELS[representation],
        )
    axes[0].set_xticks(x, ("Ellipse OOD", "20 dB", "10 dB"))
    axes[0].set_ylabel("Contrast RelL2")
    axes[0].set_title("Geometry and noise shifts", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    boundaries = ("open", "partial", "closed")
    x = np.arange(len(boundaries))
    for index, representation in enumerate(REPRESENTATIONS):
        own = {
            row["boundary"]: row
            for row in boundary_rows
            if row["representation"] == representation
        }
        axes[1].bar(
            x + (index - 1) * width,
            [own[boundary]["rel_l2_mean"] for boundary in boundaries],
            width,
            yerr=[own[boundary]["rel_l2_seed_std"] for boundary in boundaries],
            color=COLORS[representation],
            capsize=3,
        )
    axes[1].set_xticks(x, ("Open", "Partial", "Closed"))
    axes[1].set_ylabel("Contrast RelL2")
    axes[1].set_title("Zero-shot 64→128 + boundary", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _plot_noise_example(root: Path, output: Path) -> None:
    archives = {}
    for representation in REPRESENTATIONS:
        path = next(root.glob(f"iid_snr20_{representation}_seed20260922_predictions.npz"))
        archives[representation] = np.load(path)
    errors = archives["amplitude_soft_phase_gradient"]["error"]
    position = int(np.argsort(errors)[len(errors) // 2])
    target = archives["raw"]["target"][position]
    images = [target] + [archives[name]["prediction"][position] for name in REPRESENTATIONS]
    titles = ["Target"] + [LABELS[name] for name in REPRESENTATIONS]
    figure, axes = plt.subplots(1, 4, figsize=(11.2, 2.8), constrained_layout=True)
    for axis, image, title in zip(axes, images, titles, strict=True):
        artist = axis.imshow(image, cmap="magma", vmin=0.0, vmax=0.7)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(artist, ax=axes, fraction=0.025, label="Medium contrast")
    figure.suptitle("20 dB test: clean-trained operators", fontsize=11)
    figure.savefig(output, dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = _load(args.root)
    rows, boundary_rows = _aggregate(records)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "shift_rows": rows, "boundary_rows": boundary_rows}
    (args.output / "wave_set_robustness_summary.json").write_text(json.dumps(summary, indent=2))
    for name, values in (("shift", rows), ("boundary", boundary_rows)):
        with (args.output / f"wave_set_{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=values[0].keys(), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(values)
    _plot_summary(rows, boundary_rows, args.output / "wave_set_robustness.png")
    _plot_noise_example(args.root, args.output / "wave_set_20db_examples.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
