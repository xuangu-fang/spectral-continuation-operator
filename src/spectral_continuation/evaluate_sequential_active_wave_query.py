"""Evaluate greedy target-aware source--frequency acquisition for PBNO."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .cross_domain_response_data import (
    CrossDomainResponseArchive,
    load_cross_domain_archive,
)
from .probabilistic_response_operator import paired_summary
from .train_posterior_operator_composer import (
    DomainState,
    _adapter_gate,
    _build_state,
    _episode_branches,
)


def attach_evaluation_archive(
    state: DomainState,
    archive: CrossDomainResponseArchive,
    source_split: str,
    destination_split: str = "test",
) -> None:
    """Attach unseen systems without refitting any learned operator component."""

    reference = state.archive
    if archive.responses.shape[1:] != reference.responses.shape[1:]:
        raise ValueError("evaluation response geometry differs from the frozen archive")
    for name in ("sources", "receivers", "frequencies"):
        if not np.allclose(getattr(archive, name), getattr(reference, name)):
            raise ValueError(f"evaluation {name} differ from the frozen archive")
    if not np.array_equal(
        archive.source_receiver_index, reference.source_receiver_index
    ):
        raise ValueError("evaluation source-receiver alignment differs")
    selected = archive.split == source_split
    if not np.any(selected):
        raise ValueError(f"evaluation archive has no {source_split!r} systems")
    raw = torch.as_tensor(
        archive.responses[selected].reshape(int(selected.sum()), -1),
        dtype=torch.complex64,
        device=state.basis.device,
    )
    state.split_raw[destination_split] = raw
    state.split_encoded[destination_split] = (
        raw - state.center[None]
    ) / state.scale[None]
    state.neural_encoded[destination_split] = (
        raw - state.neural_center[None]
    ) / state.neural_scale[None]


def candidate_target_risks(
    state: DomainState,
    target_pair: np.ndarray,
    context: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Evaluate posterior target risk for candidate query extensions."""

    receiver = torch.arange(state.receiver_count, device=state.basis.device)
    target_query = int(target_pair[0] * state.frequency_count + target_pair[1])
    target = target_query * state.receiver_count + receiver
    target_basis = state.basis[target].to(torch.complex128)
    target_basis = target_basis * state.scale[target, None].double()
    target_gram = target_basis.mH @ target_basis
    context_query = torch.as_tensor(
        context[:, 0] * state.frequency_count + context[:, 1],
        dtype=torch.long,
        device=state.basis.device,
    )
    context_observed = (
        context_query[:, None] * state.receiver_count + receiver[None]
    ).reshape(-1)
    context_basis = state.basis[context_observed].to(torch.complex128)
    base_precision = context_basis.mH @ context_basis + torch.diag(
        state.penalty.double()
    ).to(torch.complex128)
    candidate_query = torch.as_tensor(
        candidates[:, 0] * state.frequency_count + candidates[:, 1],
        dtype=torch.long,
        device=state.basis.device,
    )
    candidate_observed = (
        candidate_query[:, None] * state.receiver_count + receiver[None]
    )
    candidate_basis = state.basis[candidate_observed].to(torch.complex128)
    precision = base_precision[None] + candidate_basis.mH @ candidate_basis
    solved = torch.linalg.solve(precision, target_gram[None].expand(len(candidates), -1, -1))
    risks = torch.diagonal(solved, dim1=-2, dim2=-1).sum(dim=-1).real
    return risks.cpu().numpy().astype(np.float64)


def greedy_variance_schedule(
    state: DomainState,
    target_pair: np.ndarray,
    ordered_candidates: np.ndarray,
    maximum_budget: int,
) -> np.ndarray:
    """Greedily minimize target posterior trace; the first probe stays fixed."""

    selected = [ordered_candidates[0]]
    remaining = [value for value in ordered_candidates[1:]]
    while len(selected) < maximum_budget:
        risks = candidate_target_risks(
            state, target_pair, np.stack(selected), np.stack(remaining)
        )
        best = int(np.argmin(risks))
        selected.append(remaining.pop(best))
    return np.stack(selected)


@torch.inference_mode()
def adapter_error(
    state: DomainState,
    split: str,
    context: np.ndarray,
    target_pair: np.ndarray,
) -> torch.Tensor:
    *_, observed, target, branches = _episode_branches(state, split, context, target_pair)
    gate, _ = _adapter_gate(state, split, context, observed, target, branches)
    prediction = branches[1] + gate[:, None] * (branches[2] - branches[1])
    truth = state.split_raw[split][:, target]
    return torch.linalg.vector_norm(prediction - truth, dim=-1) / torch.linalg.vector_norm(
        truth, dim=-1
    ).clamp_min(1e-12)


@torch.inference_mode()
def run(
    config_path: Path,
    output: Path,
    device_name: str,
    evaluation_source_split: str | None = None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    budgets = tuple(int(value) for value in config.get("active_budgets", (1, 2, 4, 8)))
    states: dict[tuple[str, int], DomainState] = {}
    for domain, setting in config["domains"].items():
        for budget in budgets:
            state = _build_state(
                domain,
                budget,
                setting,
                setting["budgets"][str(budget)],
                int(config["rank"]),
                int(config["value_rank"]),
                Path(config["adapter_checkpoint"]),
                device,
            )
            if "evaluation_dataset" in setting:
                attach_evaluation_archive(
                    state,
                    load_cross_domain_archive(Path(setting["evaluation_dataset"])),
                    str(
                        evaluation_source_split
                        or setting.get("evaluation_source_split", "test")
                    ),
                )
            states[(domain, budget)] = state
    arrays: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for prompt_seed in config["formal_test_seeds"]:
        for domain, setting in config["domains"].items():
            selection_state = states[(domain, int(config.get("selection_budget", 2)))]
            system_count = len(selection_state.split_raw["test"])
            totals = {
                budget: (
                    torch.zeros(system_count, device=device),
                    torch.zeros(system_count, device=device),
                )
                for budget in budgets
            }
            event_count = 0
            for target_source in range(selection_state.source_count):
                for target_frequency in range(selection_state.frequency_count):
                    target_pair = np.asarray(
                        [target_source, target_frequency], dtype=np.int64
                    )
                    candidates = np.asarray(
                        [
                            (source, frequency)
                            for source in range(selection_state.source_count)
                            for frequency in range(selection_state.frequency_count)
                            if source != target_source and frequency != target_frequency
                        ],
                        dtype=np.int64,
                    )
                    target_seed = (
                        int(prompt_seed)
                        + 1_000_003 * target_source
                        + 10_007 * target_frequency
                    )
                    random_schedule = candidates[
                        np.random.default_rng(target_seed).permutation(len(candidates))
                    ]
                    active_schedule = greedy_variance_schedule(
                        selection_state,
                        target_pair,
                        random_schedule,
                        max(budgets),
                    )
                    for budget in budgets:
                        state = states[(domain, budget)]
                        random_total, active_total = totals[budget]
                        random_total += adapter_error(
                            state, "test", random_schedule[:budget], target_pair
                        )
                        active_total += adapter_error(
                            state, "test", active_schedule[:budget], target_pair
                        )
                    event_count += 1
            for budget in budgets:
                random_total, active_total = totals[budget]
                arrays.setdefault((domain, budget), []).append(
                    (
                        (random_total / event_count).cpu().numpy().astype(np.float64),
                        (active_total / event_count).cpu().numpy().astype(np.float64),
                    )
                )
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {
        "method": "PBNO with greedy target-posterior acquisition",
        "evaluation_datasets": {
            domain: setting.get("evaluation_dataset", setting["dataset"])
            for domain, setting in config["domains"].items()
        },
        "evaluation_source_split": evaluation_source_split or "config/default",
        "domains": {},
    }
    for offset, ((domain, budget), values) in enumerate(sorted(arrays.items())):
        random_error = np.mean([value[0] for value in values], axis=0)
        active_error = np.mean([value[1] for value in values], axis=0)
        statistics = paired_summary(random_error, active_error, 20260927 + offset)
        row = {
            "domain": domain,
            "budget": budget,
            "random_mean": float(random_error.mean()),
            "active_mean": float(active_error.mean()),
            "relative_improvement": float(1.0 - active_error.mean() / random_error.mean()),
            **statistics,
        }
        rows.append(row)
        summary["domains"].setdefault(domain, {})[str(budget)] = row
        np.savez_compressed(
            output / f"{domain.lower()}_k{budget}_systems.npz",
            random=random_error,
            active=active_error,
        )
    with (output / "sequential_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, len(config["domains"]), figsize=(13.2, 3.6), sharey=False)
    for axis, domain in zip(axes, config["domains"], strict=True):
        domain_rows = [row for row in rows if row["domain"] == domain]
        domain_rows.sort(key=lambda value: int(value["budget"]))
        x = [int(value["budget"]) for value in domain_rows]
        axis.plot(x, [value["random_mean"] for value in domain_rows], "o--", label="random")
        axis.plot(x, [value["active_mean"] for value in domain_rows], "o-", label="target-aware")
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, labels=[str(value) for value in x])
        axis.set_title(domain)
        axis.set_xlabel("probe budget K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("raw complex relative L2")
    axes[-1].legend(frameon=False)
    figure.suptitle("Sequential source-frequency interrogation with the PBNO belief")
    figure.tight_layout()
    figure.savefig(output / "sequential_active_query.png", dpi=220)
    figure.savefig(output / "sequential_active_query.pdf")
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-source-split")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.output,
                args.device,
                args.evaluation_source_split,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
