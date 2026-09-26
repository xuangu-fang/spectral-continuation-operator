"""Freeze the heterogeneous functional-operator M0 decision artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DOMAINS = ("OpenFWI", "Deepwave", "Ocean")
FRACTIONS = (10, 25, 50, 100)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root: Path, output: Path) -> dict:
    headroom = _load(root / "heterogeneous_prompt_headroom_m0_dev3/summary.json")
    evidence = _load(root / "functional_evidence_operator_m0_dev2/summary.json")
    reranker = _load(root / "functional_prompt_reranker_m0_dev1/summary.json")
    metric = _load(root / "functional_response_metric_m1_multilayout_dev1/summary.json")
    neural_field = _load(
        root / "wave_neural_field_autodecoder_m0_ocean_dev1/ocean/summary.json"
    )
    neural_encoder = _load(
        root / "wave_neural_field_prompt_encoder_m0_ocean_dev1/summary.json"
    )

    rows = []
    for domain in DOMAINS:
        for fraction in FRACTIONS:
            key = f"{domain}_r{fraction:03d}"
            receiver_key = f"r{fraction:03d}"
            methods = headroom["results"][key]["methods"]
            oracle_gain = methods["compressed_full_dictionary_oracle"][
                "ratio_of_means_gain"
            ]
            evidence_gain = evidence[domain]["validation"][receiver_key][
                "ratio_of_means_gain"
            ]
            reranker_gain = reranker[domain]["validation"][receiver_key][
                "ratio_of_means_gain"
            ]
            metric_gain = metric[domain]["test"][receiver_key][
                "ratio_of_means_gain"
            ]
            rows.append(
                {
                    "domain": domain,
                    "receiver_fraction": fraction / 100.0,
                    "baseline_error": methods["posterior_memory"]["mean"],
                    "compressed_oracle_error": methods[
                        "compressed_full_dictionary_oracle"
                    ]["mean"],
                    "compressed_oracle_gain": oracle_gain,
                    "functional_evidence_gain_validation": evidence_gain,
                    "residual_reranker_gain_validation": reranker_gain,
                    "listwise_metric_gain_test": metric_gain,
                    "best_learned_development_gain": max(
                        evidence_gain, reranker_gain, metric_gain
                    ),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "heterogeneous_operator_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "NO-GO for replacing the stable linear posterior chart",
        "protocol": {
            "task": "strict-Q3 K=2 with independent receiver masks",
            "receiver_fractions": [0.1, 0.25, 0.5, 1.0],
            "splits": "physical-system train/validation/test",
            "primary_metric": "raw complex relative L2",
        },
        "headroom": rows,
        "neural_field": {
            "parameters": neural_field["parameters"],
            "train_full_response_reconstruction": neural_field[
                "best_train_reconstruction"
            ],
            "unseen_full_response_latent_fit_validation": 0.38261669874191284,
            "unseen_full_response_latent_fit_test": 0.3675784468650818,
            "sparse_prompt_encoder_test": {
                key: value["0"]["mean"]
                for key, value in neural_encoder["metrics"]["test"].items()
            },
        },
        "decision": [
            "Do not replace the observable linear posterior chart with an unconstrained nonlinear latent.",
            "Keep coordinate-aware set encoders as optional evidence adapters, not the system state itself.",
            "Reopen only with more informative prompts, a physics-identifiable latent, or a new real multistatic domain.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.4))
    x = np.arange(len(DOMAINS))
    r10 = [row for row in rows if row["receiver_fraction"] == 0.1]
    oracle = 100 * np.asarray([row["compressed_oracle_gain"] for row in r10])
    learned = 100 * np.asarray([row["best_learned_development_gain"] for row in r10])
    width = 0.36
    axes[0].bar(x - width / 2, oracle, width, label="compressed oracle")
    axes[0].bar(x + width / 2, learned, width, label="best learned dev")
    axes[0].axhline(15, color="0.35", linewidth=1, linestyle="--", label="15% gate")
    axes[0].set_xticks(x, DOMAINS)
    axes[0].set_ylabel("relative error reduction (%)")
    axes[0].set_title("10% receivers: ceiling vs learned")
    axes[0].legend(frameon=False, fontsize=8)

    ocean = [row for row in rows if row["domain"] == "Ocean"]
    axes[1].plot(
        [100 * row["receiver_fraction"] for row in ocean],
        [100 * row["compressed_oracle_gain"] for row in ocean],
        marker="o",
        label="compressed oracle",
    )
    axes[1].plot(
        [100 * row["receiver_fraction"] for row in ocean],
        [100 * row["best_learned_development_gain"] for row in ocean],
        marker="s",
        label="best learned dev",
    )
    axes[1].axhline(15, color="0.35", linewidth=1, linestyle="--")
    axes[1].set_xlabel("observed receivers per prompt (%)")
    axes[1].set_ylabel("relative error reduction (%)")
    axes[1].set_title("Ocean headroom is not realized")
    axes[1].legend(frameon=False, fontsize=8)

    history = neural_field["history"]
    axes[2].plot(
        [item["step"] for item in history],
        [item["full_train_reconstruction"] for item in history],
        marker="o",
        label="train full-table reconstruction",
    )
    axes[2].axhline(
        payload["neural_field"]["unseen_full_response_latent_fit_test"],
        color="0.35",
        linestyle="--",
        label="unseen full-response latent fit",
    )
    axes[2].axhline(
        neural_encoder["metrics"]["test"]["r100"]["0"]["mean"],
        color="0.6",
        linestyle=":",
        label="sparse prompt encoder",
    )
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("raw complex RelL2")
    axes[2].set_title("Nonlinear field: capacity ≠ identification")
    axes[2].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "heterogeneous_functional_operator_m0.png", dpi=220)
    figure.savefig(output / "heterogeneous_functional_operator_m0.pdf")
    plt.close(figure)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
