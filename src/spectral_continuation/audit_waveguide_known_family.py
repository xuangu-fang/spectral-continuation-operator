"""Development-only inversion with extra knowledge of the exact 15-parameter medium family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .audit_waveguide_frequency_scaling import modal_raw, propagate_code
from .audit_waveguide_generator_identification import generator_geometry, true_projection
from .audit_waveguide_trajectory_newton import observation_objective, trajectory_jacobian
from .coherent_representation import complex_metrics
from .coherent_waveguide_data import depth_basis


def public_spec(spec):
    """Prevent the hidden Gaussian parameters from entering an inference call."""
    copy = dict(spec)
    copy["centers_m"] = [[1000.0, 50.0]] * 3
    copy["widths_m"] = [[250.0, 16.5]] * 3
    copy["contrasts_m_s"] = [0.0] * 3
    return copy


def pack(spec):
    arrays = np.column_stack(
        [
            np.asarray(spec["centers_m"])[:, 0],
            np.asarray(spec["centers_m"])[:, 1],
            np.asarray(spec["widths_m"])[:, 0],
            np.asarray(spec["widths_m"])[:, 1],
            np.asarray(spec["contrasts_m_s"]),
        ]
    )
    lower = np.array([300, 20, 150, 8, -35.0])
    upper = np.array([1700, 80, 350, 25, 35.0])
    return ((arrays - lower) / (upper - lower)).ravel()


def unpack(public, x):
    lower = np.array([300, 20, 150, 8, -35.0])
    upper = np.array([1700, 80, 350, 25, 35.0])
    values = np.asarray(x).reshape(3, 5) * (upper - lower) + lower
    return {
        **public,
        "centers_m": values[:, :2].tolist(),
        "widths_m": values[:, 2:4].tolist(),
        "contrasts_m_s": values[:, 4].tolist(),
    }


def projection(public, geometry, x):
    return true_projection(unpack(public, x), geometry)


def derivatives(public, geometry, x, step=1e-4):
    columns = []
    for index in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[index] = min(1, plus[index] + step)
        minus[index] = max(0, minus[index] - step)
        column = (projection(public, geometry, plus) - projection(public, geometry, minus)) / (
            plus[index] - minus[index]
        )
        columns.append(column)
    return np.stack(columns, axis=-1)


def context_error(x, public, geometry, initial, frequencies, observed, roi, indices):
    beta = projection(public, geometry, x)
    return observation_objective(
        beta, initial, geometry, frequencies, observed, roi, indices
    )


def refine(x, public, geometry, initial, frequencies, observed, roi, indices, steps):
    history = []
    options = [1.0, 0.5, 0.25, 0.125, 0.0625]
    for iteration in range(steps):
        beta = projection(public, geometry, x)
        factor = derivatives(public, geometry, x)
        residuals, jacobians = [], []
        for frequency, truth in zip(frequencies, observed, strict=True):
            states, jacobian = trajectory_jacobian(beta, factor, initial, frequency, geometry)
            norm = np.linalg.norm(truth[roi])
            residuals.append(((truth - states[indices])[roi] / norm).ravel())
            jacobians.append((jacobian[indices][roi] / norm).reshape(-1, len(x)))
        residual, jacobian = np.concatenate(residuals), np.concatenate(jacobians)
        gram = (jacobian.conj().T @ jacobian).real
        rhs = (jacobian.conj().T @ residual).real
        damping = 1e-8 * np.trace(gram) / len(x)
        delta = np.linalg.solve(gram + damping * np.eye(len(x)), rhs)
        delta *= min(1.0, 0.25 / max(np.max(np.abs(delta)), 1e-20))
        before = context_error(
            x, public, geometry, initial, frequencies, observed, roi, indices
        )
        chosen = x.copy()
        after = before
        alpha = 0.0
        for candidate_alpha in options:
            candidate = np.clip(x + candidate_alpha * delta, 0, 1)
            value = context_error(
                candidate, public, geometry, initial, frequencies, observed, roi, indices
            )
            if value < before:
                chosen, after, alpha = candidate, value, candidate_alpha
                break
        x = chosen
        history.append(
            {"step": iteration + 1, "context_before": before, "context_after": after, "alpha": alpha}
        )
    return x, history


def evaluate(system, public, x, config, truth_root, output):
    spec = unpack(public, x)
    rows = []
    for frequency in config["query_frequencies"]:
        path = truth_root / f"system{system}_f{frequency}.npz"
        with np.load(path) as data:
            truth = data["modal_coefficients"]
            ranges, depths = data["ranges_m"], data["depths_m"]
        prediction = propagate_code(
            spec,
            frequency,
            ranges,
            config["decoder_cells"],
            config["decoder_modes"],
            config["quadrature_nodes"],
            config["device"],
            config["batch"],
        )
        roi = ranges >= config["roi_min_range_m"]
        modal = modal_raw(prediction[roi], truth[roi])
        basis = depth_basis(depths, spec["depth_m"], truth.shape[1])
        target = (basis @ truth.transpose(1, 0, 2).reshape(truth.shape[1], -1)).reshape(
            len(depths), len(ranges), 2
        )
        readout = depth_basis(depths, spec["depth_m"], prediction.shape[1])
        field = (
            readout @ prediction.transpose(1, 0, 2).reshape(prediction.shape[1], -1)
        ).reshape(len(depths), len(ranges), 2)
        for source_index, source in enumerate(config["source_variants"]):
            rows.append(
                {
                    "system": system,
                    "frequency": frequency,
                    "source": source,
                    "modal_parseval_raw": float(modal[source_index]),
                    **complex_metrics(field[:, roi, source_index], target[:, roi, source_index]),
                }
            )
    (output / f"system{system}_query.json").write_text(json.dumps(rows, indent=2))
    return rows


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    with np.load(config["training_prior"]) as data:
        train_codes, train_ids = data["target"], data["systems"].tolist()
    training_root = Path(config["training_data"])
    training_specs = []
    for system in train_ids:
        with np.load(training_root / f"system{system}_modal.npz") as data:
            training_specs.append(json.loads(str(data["spec"])))
    dataset = Path(config["development_data"])
    truth_root = Path(config["query_truth"])
    all_rows, histories = [], []
    for system in config["systems"]:
        with np.load(dataset / f"system{system}_m32_r2049.npz") as data:
            actual = json.loads(str(data["spec"]))
            frequency_indices = [
                int(np.flatnonzero(data["frequencies_hz"] == f)[0])
                for f in config["context_frequencies"]
            ]
            take = np.linspace(0, len(data["ranges_m"]) - 1, 65, dtype=int)
            observed = data["modal_coefficients"][frequency_indices][:, take]
            ranges = data["ranges_m"][take]
        public = public_spec(actual)
        with np.load(Path(config["prior32_run"]) / f"system{system}_cells256_step5.npz") as data:
            inferred = data["beta"]
        geometry = generator_geometry(
            public, config["decoder_cells"], 32, modes=32, quadrature_nodes=160
        )
        coarse = inferred.reshape(64, 4, 32).mean(axis=1)
        distances = np.linalg.norm((train_codes - coarse).reshape(len(train_codes), -1), axis=1)
        nearest = np.argsort(distances)[: config["initializations"]]
        initial = observed[0, 0]
        roi = ranges >= config["roi_min_range_m"]
        indices = np.linspace(0, config["decoder_cells"], len(ranges), dtype=int)
        candidates = []
        for training_index in nearest:
            x0 = pack(training_specs[training_index])
            fitted = least_squares(
                lambda x, public=public, geometry=geometry, inferred=inferred: (
                    projection(public, geometry, x) - inferred
                ).ravel(),
                x0,
                bounds=(np.zeros(15), np.ones(15)),
                max_nfev=config["fit_max_evaluations"],
                ftol=1e-7,
            )
            for stage, x in [("nearest_training", x0), ("fit_to_code", fitted.x)]:
                candidates.append(
                    {
                        "stage": stage,
                        "training_system": train_ids[training_index],
                        "x": x,
                        "context_error": context_error(
                            x, public, geometry, initial, config["context_frequencies"],
                            observed, roi, indices
                        ),
                    }
                )
        best = min(candidates, key=lambda item: item["context_error"])
        x, history = refine(
            best["x"], public, geometry, initial, config["context_frequencies"],
            observed, roi, indices, config["newton_steps"]
        )
        np.savez_compressed(
            output / f"system{system}_parameters.npz",
            before=best["x"],
            after=x,
            context=observed.astype(np.complex64),
        )
        histories.append(
            {
                "system": system,
                "initialization": {
                    k: v for k, v in best.items() if k != "x"
                },
                "steps": history,
            }
        )
        (output / "history.json").write_text(json.dumps(histories, indent=2))
        for stage, code in [("initial", best["x"]), ("gn5", x)]:
            for row in evaluate(system, public, code, config, truth_root, output):
                all_rows.append({"method": stage, **row})
        (output / "results.json").write_text(json.dumps(all_rows, indent=2))
        print(json.dumps({"system": system, "context_after": history[-1]["context_after"]}), flush=True)
    summary = []
    for method in ["initial", "gn5"]:
        for frequency in config["query_frequencies"]:
            for source in config["source_variants"]:
                sample = [
                    row for row in all_rows
                    if row["method"] == method and row["frequency"] == frequency
                    and row["source"] == source
                ]
                summary.append(
                    {
                        "method": method,
                        "frequency": frequency,
                        "source": source,
                        "systems": len(sample),
                        "complex_rel_l2": float(np.mean([r["complex_rel_l2"] for r in sample])),
                        "energy_weighted_phase_rmse_rad": float(
                            np.mean([r["energy_weighted_phase_rmse_rad"] for r in sample])
                        ),
                    }
                )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
