"""Oracle local stability in the actual 15-parameter medium family.

Not a predictor, not a uniqueness theorem, and not a leaderboard metric.
Construct nearby physical environments whose context responses differ little,
then measure the actual nonlinear difference at new queries. No model changes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from .coarse_waveguide_operator import encode_operator, query_coefficients
from .coherent_waveguide_data import environment, quadrature

SCALE = np.array([35.0] * 3 + [700.0, 30.0] * 3 + [100.0, 8.5] * 3)
LOWER = np.array([-35.0] * 3 + [300.0, 20.0] * 3 + [150.0, 8.0] * 3) / SCALE
UPPER = np.array([35.0] * 3 + [1700.0, 80.0] * 3 + [350.0, 25.0] * 3) / SCALE


def pack(spec):
    return (
        np.concatenate(
            [spec["contrasts_m_s"], np.ravel(spec["centers_m"]), np.ravel(spec["widths_m"])]
        )
        / SCALE
    )


def unpack(parameters, original):
    values = np.asarray(parameters) * SCALE
    spec = copy.deepcopy(original)
    spec["contrasts_m_s"] = values[:3].tolist()
    spec["centers_m"] = values[3:9].reshape(3, 2).tolist()
    spec["widths_m"] = values[9:].reshape(3, 2).tolist()
    return spec


def responses(spec, config):
    modes = config["modes"]
    original = encode_operator(spec, config["cells"], modes, 256)
    z, w, basis = quadrature(spec, modes, 256)
    narrow = {
        **original,
        "initial": basis.T @ (w * np.exp(-0.5 * ((z - spec["source_depth_m"]) / 2.0) ** 2)),
    }
    observed_r = np.linspace(0, 2000, 65)
    query_r = np.linspace(0, 2000, 2049)
    output = {}
    for f in [60, 100, 140]:
        output[f"context_{f}"] = query_coefficients(
            original, f, observed_r[observed_r >= 50]
        ).ravel()
    for name, latent, f in [
        ("broad110", original, 110),
        ("broad400", original, 400),
        ("narrow400", narrow, 400),
    ]:
        output[name] = query_coefficients(latent, f, query_r[query_r >= 50]).ravel()
    return output


def paired_step(parameters, direction, desired, context_jacobian, endpoint_limit):
    sensitivity = np.linalg.norm(context_jacobian @ direction)
    requested = desired / max(sensitivity, 1e-30)
    nonzero = np.abs(direction) > 1e-14
    margin = np.minimum(parameters - LOWER, UPPER - parameters)
    feasible = float(np.min(2 * margin[nonzero] / np.abs(direction[nonzero])))
    bounded = 2 * endpoint_limit / max(float(np.max(np.abs(direction))), 1e-30)
    return float(min(requested, feasible * 0.99, bounded)), float(requested)


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for system in config["systems"]:
        original = environment(system)
        parameters = pack(original)
        base = responses(original, config)
        norms = {k: np.linalg.norm(v) for k, v in base.items()}
        jacobians = []
        for step in config["finite_difference_steps"]:
            columns = {k: [] for k in base}
            for dimension in range(len(parameters)):
                delta = np.zeros_like(parameters)
                delta[dimension] = step
                plus = responses(unpack(parameters + delta, original), config)
                minus = responses(unpack(parameters - delta, original), config)
                for key, values in columns.items():
                    values.append((plus[key] - minus[key]) / (2 * step * norms[key]))
            jacobians.append({k: np.stack(v, axis=-1) for k, v in columns.items()})
        consistency = {
            k: float(
                np.linalg.norm(jacobians[0][k] - jacobians[1][k])
                / max(np.linalg.norm(jacobians[1][k]), 1e-30)
            )
            for k in base
        }
        (output / f"system{system}_derivative_consistency.json").write_text(
            json.dumps(consistency, indent=2)
        )
        if max(consistency.values()) > config["derivative_consistency_gate"]:
            raise ValueError(f"finite-difference Jacobian not reliable for {system}: {consistency}")
        jac = jacobians[-1]
        context = np.concatenate([jac[f"context_{f}"] for f in [60, 100, 140]]) / np.sqrt(3)
        gram = (context.conj().T @ context).real
        np.savez_compressed(
            output / f"system{system}_sensitivity.npz",
            context_gram=gram,
            query_grams=np.array(
                [(jac[k].conj().T @ jac[k]).real for k in ["broad110", "broad400", "narrow400"]]
            ),
            query_names=["broad110", "broad400", "narrow400"],
            parameters=parameters,
            config_json=json.dumps(config, sort_keys=True),
        )
        floor = config["gram_floor_relative"] * np.trace(gram) / len(parameters)
        eigenvalues = np.linalg.eigvalsh(gram)
        for query in ["broad110", "broad400", "narrow400"]:
            metric = (jac[query].conj().T @ jac[query]).real
            _, vectors = eigh(metric, gram + floor * np.eye(len(parameters)))
            direction = vectors[:, -1]
            direction /= np.linalg.norm(direction)
            linear_ratio = float(
                np.linalg.norm(jac[query] @ direction) / np.linalg.norm(context @ direction)
            )
            for desired in config["context_perturbations"]:
                length, requested = paired_step(
                    parameters,
                    direction,
                    desired,
                    context,
                    config["max_endpoint_parameter_fraction"],
                )
                minus_spec = unpack(parameters - 0.5 * length * direction, original)
                plus_spec = unpack(parameters + 0.5 * length * direction, original)
                minus, plus = responses(minus_spec, config), responses(plus_spec, config)
                differences = {
                    k: float(np.linalg.norm(plus[k] - minus[k]) / norms[k]) for k in base
                }
                observed = float(
                    np.sqrt(np.mean([differences[f"context_{f}"] ** 2 for f in [60, 100, 140]]))
                )
                record = {
                    "system": system,
                    "query": query,
                    "desired_context_rms": desired,
                    "actual_context_rms": observed,
                    "query_normalized_difference": differences[query],
                    "actual_amplification": differences[query] / max(observed, 1e-30),
                    "linearized_amplification": linear_ratio,
                    "parameter_step": length,
                    "requested_parameter_step": requested,
                    "feasibility_limited": length < requested,
                    "context_gram_eigenvalues": eigenvalues.tolist(),
                    "all_query_differences": differences,
                    "minus_environment": minus_spec,
                    "plus_environment": plus_spec,
                    "scope": "oracle stability pair; norms centered at reference, contextRMS not leaderboard mean; same actual15parameter family; not exact non-identifiability",
                }
                rows.append(record)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in record.items()
                            if k
                            not in [
                                "minus_environment",
                                "plus_environment",
                                "context_gram_eigenvalues",
                            ]
                        }
                    ),
                    flush=True,
                )
            (output / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
