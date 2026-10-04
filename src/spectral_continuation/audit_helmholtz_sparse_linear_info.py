"""Local information audit for a sparse Helmholtz response, not a method score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import svd
from scipy.sparse.linalg import splu

from .audit_helmholtz_generator_identification import laplacian
from .audit_waveguide_spectral_context_robustness import digest
from .coherent_helmholtz_data import sample_environment


def randomized_speed_prior(labels: np.ndarray, rank: int, seed: int):
    flat = labels.reshape(len(labels), -1).astype(np.float64)
    mean = flat.mean(axis=0)
    centered = flat - mean
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal((centered.shape[1], rank + 16))
    subspace = centered @ probe
    for _ in range(2):
        subspace = centered @ (centered.T @ subspace)
    q, _ = np.linalg.qr(subspace, mode="reduced")
    _, singular, right = svd(q.T @ centered, full_matrices=False)
    directions = (right[:rank] * singular[:rank, None] / np.sqrt(len(labels) - 1))
    return mean.reshape(labels.shape[1:]), directions.reshape(rank, *labels.shape[1:]), singular


def tangent_response(speed: np.ndarray, directions: np.ndarray, loss: np.ndarray,
                     source: np.ndarray, frequency: float, lap: sparse.spmatrix):
    """Differentiate (L + kappa²/c²)u=-s w.r.t. real speed coordinates."""
    interior = np.s_[1:-1, 1:-1]
    kappa = 2 * np.pi * frequency * (1 + 1j * loss[interior])
    coefficient = (kappa / speed[interior]) ** 2
    matrix = lap + sparse.diags(coefficient.ravel(), format="csc")
    lu = splu(matrix)
    u = lu.solve(-source[interior].ravel().astype(np.complex128))
    factor = 2 * kappa**2 * u.reshape(coefficient.shape) / speed[interior] ** 3
    rhs = (factor[None] * directions[:, 1:-1, 1:-1]).reshape(len(directions), -1).T
    tangent = lu.solve(rhs)
    field = np.zeros_like(speed, dtype=np.complex128)
    field[interior] = u.reshape(coefficient.shape)
    jacobian = np.zeros((len(directions), *speed.shape), dtype=np.complex128)
    jacobian[:, 1:-1, 1:-1] = tangent.T.reshape(len(directions), *coefficient.shape)
    return field, jacobian


def receivers(values: np.ndarray, line_indices: np.ndarray) -> np.ndarray:
    """Interpolate fixed teacher-grid receivers onto the diagnostic grid129."""
    x = np.asarray([88, 128, 168, 208])[line_indices] / 2
    y = (32 + 3 * np.arange(65)) / 2
    lo = np.floor(y).astype(int)
    hi = np.ceil(y).astype(int)
    weight = y - lo
    return (1 - weight)[None, :, None] * values[..., lo[:, None], x[None, :].astype(int)] \
        + weight[None, :, None] * values[..., hi[:, None], x[None, :].astype(int)]


def query_uncertainty(field: np.ndarray, jacobian: np.ndarray, covariance: np.ndarray,
                      roi: np.ndarray) -> dict:
    target = field[np.ix_(roi, roi)].ravel()
    tangent = jacobian[:, roi][:, :, roi].reshape(len(jacobian), -1).T
    weighted = tangent @ covariance
    raw = np.sqrt(max(0, np.real(np.sum(weighted * tangent.conj())))) / np.linalg.norm(target)
    phase = np.imag(tangent * target.conj()[:, None] /
                    np.maximum(abs(target[:, None]) ** 2, 1e-20))
    phase_variance = np.sum((phase @ covariance) * phase, axis=1)
    phase_rmse = np.sqrt(np.sum(abs(target) ** 2 * phase_variance) /
                         np.sum(abs(target) ** 2))
    return {"linearized_expected_raw_rms": float(raw),
            "linearized_expected_weighted_phase_rms_rad": float(phase_rmse)}


def run(config: dict, output: Path) -> None:
    if (config["grid"] != 129 or config["prior_rank"] not in (64, 256)
        or config["context_frequencies_hz"] != [6.0, 10.0, 14.0]
        or config["query_frequencies_hz"] != [11.3, 17.0]
        or config["receiver_line_counts"] not in ([1, 2, 4], [2, 4])
        or config["relative_complex_noise_rms"] not in
            ([0.001, 0.01, 0.05], [0.01])):
        raise ValueError("frozen local information setting changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    with np.load(config["training_medium_labels"], allow_pickle=False) as data:
        labels = data["speed"]
    if labels.shape != (512, 65, 65):
        raise ValueError("training labels changed")
    with np.load(config["public_context"], allow_pickle=False) as data:
        spec = json.loads(str(data["public_spec"]))
        if set(data.files) != {"samples", "frequencies", "x", "y", "public_spec"}:
            raise ValueError("public context schema changed")
    mean, directions, singular = randomized_speed_prior(
        labels, config["prior_rank"], config["randomized_svd_seed"]
    )
    coarse = np.linspace(0, 1, 65)
    axis = np.linspace(0, 1, config["grid"])
    speed = RectBivariateSpline(coarse, coarse, mean)(axis, axis)
    direction_fine = np.stack([
        RectBivariateSpline(coarse, coarse, direction)(axis, axis)
        for direction in directions
    ])
    if speed.min() < 0.4:
        raise ValueError("unsafe prior mean speed")
    _, loss, source = sample_environment(
        {**spec, "centers": [], "widths": [], "contrasts": []}, axis
    )
    lap = laplacian(config["grid"])
    observations = {}
    for frequency in config["context_frequencies_hz"] + config["query_frequencies_hz"]:
        observations[frequency] = tangent_response(
            speed, direction_fine, loss, source, frequency, lap
        )
        print(json.dumps({"frequency": frequency, "status": "tangent_done"}), flush=True)
    roi = (axis >= config["roi"][0]) & (axis <= config["roi"][1])
    eye = np.eye(config["prior_rank"])
    rows = []
    for count in config["receiver_line_counts"]:
        line_indices = np.arange(count) if count == 4 else (
            np.array([1]) if count == 1 else np.array([1, 2])
        )
        for noise in config["relative_complex_noise_rms"]:
            gram = np.zeros_like(eye)
            for frequency in config["context_frequencies_hz"]:
                field, jacobian = observations[frequency]
                measurement = receivers(field[None], line_indices).ravel()
                derivative = receivers(jacobian, line_indices).reshape(len(jacobian), -1).T
                sigma = noise * np.sqrt(np.mean(abs(measurement) ** 2))
                gram += 2 * np.real(derivative.conj().T @ derivative) / sigma**2
            covariance = np.linalg.solve(eye + gram, eye)
            for frequency in config["query_frequencies_hz"]:
                field, jacobian = observations[frequency]
                rows.append({"receiver_lines": count, "relative_complex_noise_rms": noise,
                             "query_frequency_hz": frequency,
                             "posterior_trace_over_rank": float(np.trace(covariance) / len(eye)),
                             **query_uncertainty(field, jacobian, covariance, roi)})
    priors = []
    for frequency in config["query_frequencies_hz"]:
        field, jacobian = observations[frequency]
        priors.append({"query_frequency_hz": frequency,
                       **query_uncertainty(field, jacobian, eye, roi)})
    manifest = {"scope": config["scope"], "implementation_sha256": digest(__file__),
                "training_labels_sha256": digest(config["training_medium_labels"]),
                "public_context_sha256": digest(config["public_context"]),
                "label_singular_top": singular[:5].tolist(),
                "mean_speed_range": [float(speed.min()), float(speed.max())]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "results.json").write_text(json.dumps({"rows": rows, "prior": priors}, indent=2))
    print(json.dumps({"rows": len(rows), "prior": priors}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
