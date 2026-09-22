"""Create paper-style tables and figures for the wave-representation POC."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("artifacts/wave_operator_representation")
SEEDS = (20260922, 20260923, 20260924)
COLORS = {
    "Raw complex": "#6b7280",
    "Homogeneous carrier": "#9ca3af",
    "Amplitude": "#4c78a8",
    "Amp. + phase gradient": "#e45756",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std())


def _bar(ax, labels, means, stds, title, ylabel):
    colors = [COLORS[label] for label in labels]
    ax.bar(np.arange(len(labels)), means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(np.arange(len(labels)), [label.replace(" ", "\n") for label in labels])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def summarize(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    contrastive_paths = {
        "Raw complex": lambda seed: ROOT / f"m2_stable_seed{seed}/contrastive_raw.json",
        "Homogeneous carrier": lambda seed: ROOT / f"m2_stable_seed{seed}/contrastive_homogeneous_carrier.json",
        "Amplitude": lambda seed: ROOT / f"m2_stable_seed{seed}/contrastive_amplitude.json",
        "Amp. + phase gradient": lambda seed: ROOT / f"m2_ablation_seed{seed}/contrastive_amplitude_phase_gradient.json",
    }
    retrieval = {
        label: _mean_std([_read(path(seed))["test_top1"] for seed in SEEDS])
        for label, path in contrastive_paths.items()
    }
    medium_paths = {
        "Raw complex": lambda seed: ROOT / f"m3_medium_seed{seed}/medium_probe_raw.json",
        "Amplitude": lambda seed: ROOT / f"m3_medium_seed{seed}/medium_probe_amplitude.json",
        "Amp. + phase gradient": lambda seed: ROOT / f"m3_ablation_seed{seed}/medium_probe_amplitude_phase_gradient.json",
    }
    medium = {
        label: {
            "error": _mean_std(
                [_read(path(seed))["test_contrast_rel_l2_mean"] for seed in SEEDS]
            ),
            "iou": _mean_std(
                [_read(path(seed))["test_inclusion_iou_mean"] for seed in SEEDS]
            ),
        }
        for label, path in medium_paths.items()
    }
    ood_names = {
        "Raw complex": "raw",
        "Amplitude": "amplitude",
        "Amp. + phase gradient": "amplitude_phase_gradient",
    }
    ood = {
        label: _mean_std(
            [
                _read(ROOT / f"m4_ood/ood_{name}_seed{seed}.json")["contrast_rel_l2_mean"]
                for seed in SEEDS
            ]
        )
        for label, name in ood_names.items()
    }
    budget = {"Raw complex": [], "Amp. + phase gradient": []}
    for count in (1, 2, 4, 6):
        if count == 6:
            raw_path = ROOT / "m3_medium_seed20260922/medium_probe_raw.json"
            phase_path = ROOT / "m3_ablation_seed20260922/medium_probe_amplitude_phase_gradient.json"
        else:
            raw_path = ROOT / f"m3_budget_s{count}_seed20260922/medium_probe_raw.json"
            phase_path = ROOT / f"m3_budget_s{count}_seed20260922/medium_probe_amplitude_phase_gradient.json"
        budget["Raw complex"].append(_read(raw_path)["test_contrast_rel_l2_mean"])
        budget["Amp. + phase gradient"].append(
            _read(phase_path)["test_contrast_rel_l2_mean"]
        )

    clean = {
        "Raw complex": _read(ROOT / "m3_medium_seed20260922/medium_probe_raw.json")["test_contrast_rel_l2_mean"],
        "Amplitude": _read(ROOT / "m3_medium_seed20260922/medium_probe_amplitude.json")["test_contrast_rel_l2_mean"],
        "Unit phase gradient": _read(ROOT / "m3_ablation_seed20260922/medium_probe_amplitude_phase_gradient.json")["test_contrast_rel_l2_mean"],
        "Soft phase gradient": _read(ROOT / "m3_soft_seed20260922/medium_probe_amplitude_soft_phase_gradient.json")["test_contrast_rel_l2_mean"],
        "Noise-trained soft": _read(ROOT / "m5_noise_eval/noise_trained_clean_amplitude_soft_phase_gradient_seed20260922.json")["contrast_rel_l2_mean"],
    }
    noise = {label: [value] for label, value in clean.items()}
    filename_names = {
        "Raw complex": ("m4_noise", "iid_noise", "raw"),
        "Amplitude": ("m4_noise", "iid_noise", "amplitude"),
        "Unit phase gradient": ("m4_noise", "iid_noise", "amplitude_phase_gradient"),
        "Soft phase gradient": ("m4_noise", "iid_noise", "amplitude_soft_phase_gradient"),
        "Noise-trained soft": ("m5_noise_eval", "noise_trained", "amplitude_soft_phase_gradient"),
    }
    for label, (folder, prefix, representation) in filename_names.items():
        for snr in (30, 20, 10, 0):
            record = _read(
                ROOT / folder / f"{prefix}_snr{snr}_{representation}_seed20260922.json"
            )
            noise[label].append(record["contrast_rel_l2_mean"])

    # convert_geometry_pool orders each of ten media as open/partial/closed.
    boundary = np.tile(np.asarray(["open", "partial", "closed"]), 10)
    cross_grid = {}
    cross_files = {
        "Raw complex": "legacy_128_boundary_clean_raw_seed20260922_predictions.npz",
        "Amplitude": "legacy_128_boundary_clean_amplitude_seed20260922_predictions.npz",
        "Amp. + phase gradient": "legacy_128_physical_stride_clean_amplitude_phase_gradient_seed20260922_predictions.npz",
    }
    for label, filename in cross_files.items():
        values = np.load(ROOT / "m6_cross_dataset" / filename)
        prediction, target = values["prediction"], values["target"]
        error = np.linalg.norm((prediction - target).reshape(len(target), -1), axis=1)
        error /= np.linalg.norm(target.reshape(len(target), -1), axis=1) + 1e-9
        cross_grid[label] = {
            name: float(error[boundary == name].mean())
            for name in ("open", "partial", "closed")
        }

    summary = {
        "retrieval_top1": retrieval,
        "medium_iid": medium,
        "medium_ood_ellipses": ood,
        "source_budget": budget,
        "noise_curve": noise,
        "cross_grid_boundary": cross_grid,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "headline_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["representation", "retrieval_top1_mean", "iid_rel_l2_mean", "iid_iou_mean", "ellipse_ood_rel_l2_mean"])
        for label in ("Raw complex", "Amplitude", "Amp. + phase gradient"):
            writer.writerow(
                [label, retrieval[label][0], medium[label]["error"][0], medium[label]["iou"][0], ood[label][0]]
            )

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.axis("off")
    ax.text(0.02, 0.92, "A  Response-conditioned system representation", fontweight="bold", fontsize=11)
    boxes = [
        (0.01, 0.64, "Low-f / source A\nresponses"),
        (0.37, 0.64, "Amplitude + local\nphase advance"),
        (0.73, 0.64, "System\nrepresentation"),
    ]
    for x, y, label in boxes:
        ax.text(
            x, y, label, fontsize=8, ha="left", va="center",
            bbox={"boxstyle": "round,pad=.35", "fc": "#f3f4f6", "ec": "#64748b"},
        )
    ax.annotate("", xy=(0.36, 0.64), xytext=(0.29, 0.64), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.72, 0.64), xytext=(0.66, 0.64), arrowprops={"arrowstyle": "->"})
    ax.text(
        0.17, 0.26, "Cross-view retrieval\n(high-f / source set B)", ha="center",
        bbox={"boxstyle": "round,pad=.35", "fc": "#eef2ff", "ec": "#6366f1"},
    )
    ax.text(
        0.68, 0.26, "Hidden coefficient recovery\n+ OOD / noise stress tests", ha="center",
        bbox={"boxstyle": "round,pad=.35", "fc": "#fff1f2", "ec": "#e11d48"},
    )

    labels = list(retrieval)
    _bar(axes[0, 1], labels, [retrieval[x][0] for x in labels], [retrieval[x][1] for x in labels], "B  Disjoint-view system retrieval", "Top-1 accuracy")
    axes[0, 1].axhline(1 / 24, color="black", linestyle="--", linewidth=1, label="chance")
    axes[0, 1].set_ylim(0, 1.08)

    labels3 = ["Raw complex", "Amplitude", "Amp. + phase gradient"]
    width = 0.35
    x = np.arange(3)
    axes[0, 2].bar(x - width / 2, [medium[k]["error"][0] for k in labels3], width, yerr=[medium[k]["error"][1] for k in labels3], label="IID circles", color=[COLORS[k] for k in labels3], alpha=.95, capsize=3)
    axes[0, 2].bar(x + width / 2, [ood[k][0] for k in labels3], width, yerr=[ood[k][1] for k in labels3], label="OOD ellipses", color=[COLORS[k] for k in labels3], alpha=.45, capsize=3)
    axes[0, 2].set_xticks(x, [k.replace(" ", "\n") for k in labels3])
    axes[0, 2].set_ylabel("Coefficient contrast RelL2 ↓")
    axes[0, 2].set_title("C  Spatial sufficiency", loc="left", fontweight="bold")
    axes[0, 2].legend(frameon=False)
    axes[0, 2].grid(axis="y", alpha=.25)

    for label, values in budget.items():
        axes[1, 0].plot((1, 2, 4, 6), values, marker="o", label=label, color=COLORS[label])
    axes[1, 0].set_title("D  Source-view budget", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Number of sources (4 frequencies each)")
    axes[1, 0].set_ylabel("Coefficient contrast RelL2 ↓")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=.25)

    noise_colors = ["#6b7280", "#4c78a8", "#f59e0b", "#e45756", "#7c3aed"]
    for (label, values), color in zip(noise.items(), noise_colors):
        axes[1, 1].plot(range(5), values, marker="o", label=label, color=color)
    axes[1, 1].set_xticks(range(5), ["clean", "30", "20", "10", "0"])
    axes[1, 1].set_xlabel("Test SNR (dB)")
    axes[1, 1].set_ylabel("Coefficient contrast RelL2 ↓")
    axes[1, 1].set_title("E  Noise reveals a reliability boundary", loc="left", fontweight="bold")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(0.04, 3.5)
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 1].grid(alpha=.25)

    boundaries = ("open", "partial", "closed")
    x = np.arange(3)
    width = 0.25
    for position, label in enumerate(labels3):
        axes[1, 2].bar(x + (position - 1) * width, [cross_grid[label][b] for b in boundaries], width, label=label, color=COLORS[label])
    axes[1, 2].set_xticks(x, boundaries)
    axes[1, 2].set_ylabel("Coefficient contrast RelL2 ↓")
    axes[1, 2].set_title("F  Zero-shot 64→128 + boundary shift", loc="left", fontweight="bold")
    axes[1, 2].legend(frameon=False, fontsize=7)
    axes[1, 2].grid(axis="y", alpha=.25)
    figure.savefig(output / "wave_representation_main.png", dpi=220)
    figure.savefig(output / "wave_representation_main.pdf")
    plt.close(figure)

    prediction_files = {
        "Raw complex": ROOT / "m3_medium_seed20260922/medium_probe_raw_predictions.npz",
        "Amplitude": ROOT / "m3_medium_seed20260922/medium_probe_amplitude_predictions.npz",
        "Amp. + phase gradient": ROOT / "m3_ablation_seed20260922/medium_probe_amplitude_phase_gradient_predictions.npz",
    }
    predictions = {label: np.load(path) for label, path in prediction_files.items()}
    example = 4
    target = predictions["Raw complex"]["target"][example]
    figure, axes = plt.subplots(2, 4, figsize=(12, 5.8), constrained_layout=True)
    images = [("Ground truth", target)] + [(label, predictions[label]["prediction"][example]) for label in labels3]
    for column, (label, image) in enumerate(images):
        handle = axes[0, column].imshow(image, cmap="magma", vmin=0.0, vmax=0.6)
        axes[0, column].set_title(label)
        axes[0, column].axis("off")
        if column == 3:
            figure.colorbar(handle, ax=axes[0, :], shrink=.7, label="wavespeed contrast")
        error_map = np.abs(image - target)
        axes[1, column].imshow(error_map, cmap="viridis", vmin=0.0, vmax=0.25)
        axes[1, column].set_title("absolute error")
        axes[1, column].axis("off")
    figure.savefig(output / "wave_representation_examples.png", dpi=220)
    figure.savefig(output / "wave_representation_examples.pdf")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    summarize(ROOT / "summary")
