"""Plot fixed WaveBench test examples for the raw and phase-coordinate models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .train_wavebench_identifier import WaveBenchBank, make_split, response_coordinates
from .wave_set_encoder import ScalarFieldDecoder, WaveSetEncoder


def _load_model(
    path: Path, channels: int, device: str
) -> tuple[WaveSetEncoder, ScalarFieldDecoder]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = WaveSetEncoder(channels, width=24, dimension=128, aggregation="slots", slots=4)
    decoder = ScalarFieldDecoder(24)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder


@torch.no_grad()
def _predict(
    model: tuple[WaveSetEncoder, ScalarFieldDecoder],
    bank: WaveBenchBank,
    indices: np.ndarray,
    subset: tuple[int, ...],
    representation: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    response, target = bank.load(indices, subset)
    value = response_coordinates(torch.from_numpy(response).to(device), representation)
    metadata = torch.from_numpy(bank.metadata(len(indices), subset)).to(device)
    encoder, decoder = model
    normalized = decoder(encoder.spatial(value, metadata))[:, 0].cpu().numpy()
    return normalized * bank.target_std + bank.target_mean, target[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = make_split(42, 8000, 500, 500)
    # Pre-declared positions rather than examples chosen by visual quality.
    positions = np.asarray((0, len(split.test) // 2, len(split.test) - 1))
    indices = split.test[positions]
    bank = WaveBenchBank(args.data_root, split.train)
    raw = _load_model(args.raw_checkpoint, 2, device)
    phase = _load_model(args.phase_checkpoint, 5, device)
    raw_prediction, truth = _predict(raw, bank, indices, (0, 1, 2, 3), "raw", device)
    phase_prediction, _ = _predict(
        phase, bank, indices, (0, 1, 2, 3), "amplitude_soft_phase_gradient", device
    )

    figure, axes = plt.subplots(3, 5, figsize=(10.5, 5.8), constrained_layout=True)
    for row, system in enumerate(indices):
        vmin, vmax = float(truth[row].min()), float(truth[row].max())
        error_max = float(
            max(
                np.abs(raw_prediction[row] - truth[row]).max(),
                np.abs(phase_prediction[row] - truth[row]).max(),
            )
        )
        panels = (
            (truth[row], "Truth", "viridis", vmin, vmax),
            (raw_prediction[row], "Raw complex", "viridis", vmin, vmax),
            (phase_prediction[row], "Phase coordinate", "viridis", vmin, vmax),
            (np.abs(raw_prediction[row] - truth[row]), "Raw |error|", "magma", 0.0, error_max),
            (np.abs(phase_prediction[row] - truth[row]), "Phase |error|", "magma", 0.0, error_max),
        )
        for column, (value, title, cmap, lower, upper) in enumerate(panels):
            axes[row, column].imshow(value, cmap=cmap, vmin=lower, vmax=upper)
            axes[row, column].set_xticks(())
            axes[row, column].set_yticks(())
            if row == 0:
                axes[row, column].set_title(title, fontsize=9)
        axes[row, 0].set_ylabel(f"system {int(system)}", fontsize=8)
    figure.suptitle("WaveBench system identification from all four frequencies", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=240)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
