"""Summarize continuous-query sampling limits across receiver grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_one(path: Path, domain: str | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if domain is not None:
        return payload[domain]
    return next(iter(payload.values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepwave64", type=Path, required=True)
    parser.add_argument("--deepwave128", type=Path, required=True)
    parser.add_argument("--deepwave176", type=Path, required=True)
    parser.add_argument("--ocean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deepwave = [load_one(args.deepwave64), load_one(args.deepwave128), load_one(args.deepwave176)]
    ocean = load_one(args.ocean, "Ocean")
    rows = []
    for value in deepwave:
        oracle = min(value["oracle_interpolation_rel_l2"].values())
        rows.append(
            {
                "observed_receivers": value["observed_receivers"],
                "oracle": oracle,
                "scalar": value["test"]["baseline_mean"],
                "spectrum": value["test"]["candidate_mean"],
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "deepwave_sampling_curve": rows,
        "ocean": {
            "observed_receivers": ocean["observed_receivers"],
            "query_receivers": ocean["query_receivers"],
            "oracle": min(ocean["oracle_interpolation_rel_l2"].values()),
            "scalar": ocean["test"]["baseline_mean"],
            "spectrum": ocean["test"]["candidate_mean"],
        },
        "conclusion": (
            "Deepwave joint source-coordinate query is sampling-limited on the discrete ring; "
            "Ocean passes the continuous-query gate."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    x = [value["observed_receivers"] for value in rows]
    axes[0].plot(x, [value["oracle"] for value in rows], "o-", label="sampling oracle")
    axes[0].plot(x, [value["scalar"] for value in rows], "^--", label="scalar operator")
    axes[0].plot(x, [value["spectrum"] for value in rows], "s-", label="spectrum operator")
    axes[0].set_title("Deepwave ring: sampling limited")
    axes[0].set_xlabel("Observed receiver grid size")
    axes[0].set_ylabel("Held-out receiver complex RelL2")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    ocean_values = summary["ocean"]
    axes[1].bar(
        ("oracle", "scalar", "spectrum"),
        (ocean_values["oracle"], ocean_values["scalar"], ocean_values["spectrum"]),
        color=("#999999", "#777777", "#2a6fbb"),
    )
    axes[1].set_title("Ocean depth array: identifiable")
    axes[1].set_ylabel("Held-out receiver complex RelL2")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output / "continuous_sampling.png", dpi=220)
    figure.savefig(args.output / "continuous_sampling.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
