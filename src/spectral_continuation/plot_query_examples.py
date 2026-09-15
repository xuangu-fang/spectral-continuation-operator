"""Plot a deterministic, median-contrast spectral-continuation example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SpectralQueryDataset, load_ocean_npz
from .metrics import metric_vectors
from .model import QueryUNet, reconstruct


@torch.no_grad()
def predict(run_dir: Path, frequencies: tuple[float, ...], device: torch.device):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    bundle = load_ocean_npz(Path(summary["args"]["dataset"]))
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    normalization = checkpoint["normalization"]
    validation = np.flatnonzero(bundle.split == "val")
    model = QueryUNet(
        SpectralQueryDataset(
            bundle,
            validation,
            summary["visible_frequencies_hz"],
            frequencies[0],
            normalization,
            int(summary["args"]["max_slots"]),
            bool(summary["field_only"]),
        ).input_channels,
        int(summary["args"]["width"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions = {}
    for frequency in frequencies:
        dataset = SpectralQueryDataset(
            bundle,
            validation,
            summary["visible_frequencies_hz"],
            frequency,
            normalization,
            int(summary["args"]["max_slots"]),
            bool(summary["field_only"]),
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        current = []
        for inputs, carrier, _ in loader:
            output = reconstruct(
                model(inputs.to(device)),
                carrier.to(device),
                normalization.target_scale,
                summary["parameterization"],
            )
            current.append(output.cpu())
        predictions[frequency] = torch.cat(current)
    targets = {frequency: bundle.fields[frequency][validation] for frequency in frequencies}
    return predictions, targets, bundle.ids[validation]


def run(args):
    frequencies = tuple(map(float, args.frequencies.split(",")))
    device = torch.device(args.device)
    baseline, targets, ids = predict(args.baseline_run, frequencies, device)
    candidate, candidate_targets, candidate_ids = predict(args.candidate_run, frequencies, device)
    if not np.array_equal(ids, candidate_ids):
        raise RuntimeError("baseline and candidate validation IDs differ")
    for frequency in frequencies:
        if not torch.equal(targets[frequency], candidate_targets[frequency]):
            raise RuntimeError("baseline and candidate targets differ")
    rank_frequency = args.rank_frequency or frequencies[0]
    baseline_error = metric_vectors(baseline[rank_frequency], targets[rank_frequency])[
        "complex_rel_l2"
    ]
    candidate_error = metric_vectors(candidate[rank_frequency], targets[rank_frequency])[
        "complex_rel_l2"
    ]
    improvement = baseline_error - candidate_error
    order = np.argsort(improvement)
    selected = int(order[len(order) // 2])

    figure, axes = plt.subplots(len(frequencies), 5, figsize=(14.2, 2.6 * len(frequencies)))
    axes = np.atleast_2d(axes)
    column_titles = (
        "Target: real field",
        "Single/direct: real field",
        "Fine-band/carrier: real field",
        "Single/direct: |error|",
        "Fine-band/carrier: |error|",
    )
    for axis, title in zip(axes[0], column_titles, strict=True):
        axis.set_title(title, fontsize=10)
    for row, frequency in enumerate(frequencies):
        target = torch.complex(targets[frequency][selected, 0], targets[frequency][selected, 1])
        base = torch.complex(baseline[frequency][selected, 0], baseline[frequency][selected, 1])
        method = torch.complex(candidate[frequency][selected, 0], candidate[frequency][selected, 1])
        real_scale = float(torch.quantile(target.real.abs().flatten(), 0.995).clamp_min(1e-8))
        error_scale = float(
            torch.quantile(
                torch.cat(((base - target).abs().flatten(), (method - target).abs().flatten())),
                0.995,
            ).clamp_min(1e-8)
        )
        images = (
            target.real,
            base.real,
            method.real,
            (base - target).abs(),
            (method - target).abs(),
        )
        for column, (axis, image) in enumerate(zip(axes[row], images, strict=True)):
            if column < 3:
                handle = axis.imshow(
                    image, cmap="RdBu_r", vmin=-real_scale, vmax=real_scale, aspect="auto"
                )
            else:
                handle = axis.imshow(image, cmap="magma", vmin=0, vmax=error_scale, aspect="auto")
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                base_rel = metric_vectors(
                    baseline[frequency][selected : selected + 1],
                    targets[frequency][selected : selected + 1],
                )["complex_rel_l2"][0]
                method_rel = metric_vectors(
                    candidate[frequency][selected : selected + 1],
                    targets[frequency][selected : selected + 1],
                )["complex_rel_l2"][0]
                axis.set_ylabel(f"{frequency:g} Hz\nRelL2 {base_rel:.2f} → {method_rel:.2f}")
            if column in (2, 4):
                figure.colorbar(handle, ax=axis, fraction=0.025, pad=0.02)
    figure.suptitle(
        f"Median-ranked validation contrast at {rank_frequency:g} Hz (environment {ids[selected]})",
        fontsize=12,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "selection_rule": f"median baseline-minus-candidate RelL2 at {rank_frequency:g} Hz",
        "selected_validation_id": str(ids[selected]),
        "selected_position": selected,
        "rank_frequency_hz": rank_frequency,
        "improvement_at_rank_frequency": float(improvement[selected]),
        "baseline_run": str(args.baseline_run),
        "candidate_run": str(args.candidate_run),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--frequencies", default="100,125,150")
    parser.add_argument("--rank-frequency", type=float, default=125.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
