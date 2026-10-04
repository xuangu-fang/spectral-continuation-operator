"""Scientifically labeled OOD information-condition chart for the same 16 systems."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(paths: dict[str, Path], output: Path) -> None:
    ood = json.loads(paths["ood"].read_text())
    matched = json.loads(paths["matched"].read_text())
    joint = json.loads(paths["joint"].read_text())
    exact = json.loads(paths["exact"].read_text())
    systems = sorted({r["system"] for r in ood})
    if len(systems) != 16:
        raise ValueError("expected the complete same 16-system OOD cohort")
    series = [
        ("fixed_taper5_lowband", ood, lambda r: r["method"] == "rank256_context"),
        ("matched_taper10_lowband", matched,
         lambda r: r["method"] == "matched_taper10_context"),
        ("exact_known_speed_family_lowband", exact,
         lambda r: r["method"] == "exact_known_speed_family"),
        ("joint_target_oracle", joint,
         lambda r: r["step"] == 5 and r["method"] == "joint_response_rank256_oracle"),
        ("true_medium32_oracle", ood,
         lambda r: r["method"] == "true_medium32_oracle"),
    ]
    colors = ["#252525", "#d95f02", "#1f78b4", "#7570b3", "#1b9e77"]
    markers = ["o", "s", "P", "^", "D"]
    display = [
        "Frozen prior, low-band only",
        "Target-family labels + low-band",
        "Known exact speed family + low-band",
        "Both target fields leaked (oracle)",
        "True medium readout (oracle)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True)
    records = []
    for axis, source in zip(axes, ["original", "narrow2m"], strict=True):
        for index, (name, dataset, predicate) in enumerate(series):
            lookup = {
                r["system"]: float(r["complex_rel_l2"])
                for r in dataset if predicate(r) and r["frequency"] == 800
                and r["source"] == source
            }
            if set(lookup) != set(systems):
                raise ValueError(f"incomplete {name} {source} data")
            values = np.array([lookup[system] for system in systems])
            records.append({"series": name, "source": source, "systems": systems,
                            "raw": values.tolist(), "mean": float(values.mean())})
            axis.plot(np.arange(len(systems)), values, color=colors[index],
                      marker=markers[index], markersize=4, linewidth=1.4,
                      label=display[index])
        axis.axhline(0.05, color="#666666", linestyle="--", linewidth=1.0)
        axis.set_title("Original broad source" if source == "original" else "Unseen narrow 2 m source")
        axis.set_xlabel("New taper-10 system index")
        axis.set_xticks([0, 3, 7, 11, 15])
        axis.set_xlim(-0.5, 15.5)
        axis.grid(True, which="both", alpha=0.18)
        axis.set_yscale("log")
        axis.set_ylim(0.001, 1.0)
    axes[0].set_ylabel("Raw complex relative L2 at 800 Hz (no phase alignment)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, fontsize=7.6, frameon=False)
    fig.suptitle("Medium correlation shift: accuracy depends on available information",
                 fontsize=13, y=0.995)
    fig.text(0.5, 0.025,
             "All curves use the same 16 systems. Orange has 512 extra target-family labels; "
             "blue knows the true synthetic family; purple/green read target field/medium. Not a fair ranking.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 0.845))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "Same 16 taper10 systems; information conditions NOT matched; oracle curves are nondeployable.",
        "inputs_sha256": {name: digest(path) for name, path in paths.items()},
        "series": records,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run({"ood": args.ood, "matched": args.matched, "joint": args.joint,
         "exact": args.exact}, args.output)


if __name__ == "__main__":
    main()
