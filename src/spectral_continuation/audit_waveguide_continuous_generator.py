"""Continuous training-prior basis vs decoder discretization compensation.

All choices use the original development cohort. Extra-information medium
projection is evaluated only as an oracle, never in code inference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .audit_waveguide_generator_identification import (
    generator_geometry,
    latent_from_coefficients,
    true_projection,
)
from .audit_waveguide_generator_prior import fit_prior
from .audit_waveguide_spectral_transport import encode_field
from .audit_waveguide_trajectory_newton import newton_step
from .coarse_waveguide_operator import query_field
from .coherent_representation import complex_metrics


def resample_generator(values, cells):
    """Continuous range basis. Boundary half-cells use cubic extrapolation."""
    if cells < 4 or len(values) < 4:
        raise ValueError("at least four range cells required")
    old = (np.arange(len(values)) + 0.5) / len(values)
    new = (np.arange(cells) + 0.5) / cells
    return CubicSpline(old, values, axis=0)(new)


def load_truth(path):
    with np.load(path) as data:
        return {
            "spec": json.loads(str(data["spec"])),
            "fields": data["field"],
            "frequencies": data["frequencies_hz"],
            "ranges": data["ranges_m"],
            "depths": data["depths_m"],
        }


def query_rows(beta, data, frequencies, modes, roi_min=50):
    geometry = generator_geometry(data["spec"], len(beta), beta.shape[1], modes=modes)
    # Source/background metadata are public; hidden contrast is removed by geometry.
    spec = data["spec"]
    initial = geometry["basis"].T @ (
        geometry["weights"]
        * np.exp(-0.5 * ((geometry["z"] - spec["source_depth_m"]) / spec["source_width_m"]) ** 2)
    )
    latent, clipping = latent_from_coefficients(beta, initial, geometry)
    roi = data["ranges"] >= roi_min
    rows = []
    for frequency in frequencies:
        fi = int(np.flatnonzero(data["frequencies"] == frequency)[0])
        start = time.perf_counter()
        prediction = query_field(latent, frequency, data["ranges"], data["depths"])
        seconds = time.perf_counter() - start
        rows.append(
            {
                "frequency": float(frequency),
                "clipping_fraction": clipping,
                "query_only_seconds": seconds,
                **complex_metrics(prediction[:, roi], data["fields"][fi][:, roi]),
            }
        )
    return rows


def run(options, output):
    output.mkdir(parents=True, exist_ok=False)
    base = json.loads((Path(options["training_run"]) / "config.json").read_text())
    (output / "config.json").write_text(json.dumps({**options, "base": base}, indent=2))
    with np.load(Path(options["training_run"]) / "training_statistics.npz") as data:
        mean, factor = fit_prior(data["target"].astype(float), options["rank"])
    with np.load(options["initial_coefficients"]) as data:
        if data["systems"].tolist() != base["validation_systems"]:
            raise ValueError("initial code system order mismatch")
        initial_beta = data["coefficients"]
    coordinates = np.linalg.lstsq(
        factor.reshape(-1, options["rank"]),
        (initial_beta - mean).reshape(len(initial_beta), -1).T,
        rcond=None,
    )[0].T
    reconstructed = mean + np.einsum("rdk,bk->brd", factor, coordinates)
    np.testing.assert_allclose(initial_beta, reconstructed, atol=1e-10, rtol=1e-8)
    rows, history = [], []
    for si, system in enumerate(base["validation_systems"]):
        original = load_truth(Path(base["data"]) / f"system{system}_m32_r2049.npz")
        paths = [Path(root) / f"system{system}_narrow.npz" for root in options["narrow_data"]]
        paths = [path for path in paths if path.exists()]
        if len(paths) != 1:
            raise ValueError("exactly one narrow query truth required")
        narrow = load_truth(paths[0])
        take = np.linspace(0, len(original["ranges"]) - 1, base["context_range_samples"], dtype=int)
        observed = np.array(
            [
                encode_field(
                    original["fields"][int(np.flatnonzero(original["frequencies"] == f)[0])][
                        :, take
                    ],
                    original["depths"],
                    original["spec"]["depth_m"],
                    32,
                )
                for f in base["observed_frequencies"]
            ]
        )
        context_roi = original["ranges"][take] >= base["roi_min_range_m"]
        for cells in options["decoder_cells"]:
            fine_mean, fine_factor = (
                resample_generator(mean, cells),
                resample_generator(factor, cells),
            )
            beta = fine_mean + np.einsum("rdk,k->rd", fine_factor, coordinates[si])
            geometry = generator_geometry(original["spec"], cells, base["depth_terms"])
            indices = np.linspace(0, cells, base["context_range_samples"])
            if not np.array_equal(indices, indices.astype(int)):
                raise ValueError("fixed observation profiles must remain decoder edges")
            step_options = {**options, "observation_indices": indices.astype(int)}
            for step in range(max(options["checkpoints"]) + 1):
                if step in options["checkpoints"]:
                    for variant, data, frequencies, modes in [
                        ("original", original, base["query_frequencies"], 32),
                        ("narrow", narrow, narrow["frequencies"], 64),
                    ]:
                        rows.extend(
                            {
                                "system": system,
                                "cells": cells,
                                "step": step,
                                "method": "continuous_prior",
                                "depth_terms": base["depth_terms"],
                                "latent_real_dof": options["rank"],
                                "variant": variant,
                                **row,
                            }
                            for row in query_rows(beta, data, frequencies, modes)
                        )
                    np.savez_compressed(
                        output / f"system{system}_cells{cells}_step{step}.npz", beta=beta
                    )
                if step < max(options["checkpoints"]):
                    start = time.perf_counter()
                    beta, diagnostic = newton_step(
                        beta,
                        fine_factor,
                        observed[0, 0],
                        geometry,
                        base["observed_frequencies"],
                        observed,
                        context_roi,
                        step_options,
                    )
                    history.append(
                        {
                            "system": system,
                            "cells": cells,
                            "step": step + 1,
                            "seconds": time.perf_counter() - start,
                            **diagnostic,
                        }
                    )
            for terms in options["oracle_depth_terms"]:
                oracle_geometry = generator_geometry(original["spec"], cells, terms)
                oracle = true_projection(original["spec"], oracle_geometry)
                for variant, data, frequencies, modes in [
                    ("original", original, base["query_frequencies"], 32),
                    ("narrow", narrow, narrow["frequencies"], 64),
                ]:
                    rows.extend(
                        {
                            "system": system,
                            "cells": cells,
                            "step": -1,
                            "method": "true_medium_projection_oracle",
                            "depth_terms": terms,
                            "variant": variant,
                            **row,
                        }
                        for row in query_rows(oracle, data, frequencies, modes)
                    )
            (output / "results.json").write_text(json.dumps(rows, indent=2))
            (output / "history.json").write_text(json.dumps(history, indent=2))
            print(
                json.dumps({"system": system, "cells": cells, "complete_rows": len(rows)}),
                flush=True,
            )
    summary = []
    keys = sorted(
        {(r["method"], r["cells"], r["depth_terms"], r["step"], r["variant"]) for r in rows}
    )
    for method, cells, terms, step, variant in keys:
        selected = [
            r
            for r in rows
            if (r["method"], r["cells"], r["depth_terms"], r["step"], r["variant"])
            == (method, cells, terms, step, variant)
        ]
        summary.append(
            {
                "method": method,
                "cells": cells,
                "depth_terms": terms,
                "step": step,
                "variant": variant,
                "raw": float(np.mean([r["complex_rel_l2"] for r in selected])),
                "phase_rmse_rad": float(
                    np.mean([r["energy_weighted_phase_rmse_rad"] for r in selected])
                ),
                "narrow400_raw": float(
                    np.mean([r["complex_rel_l2"] for r in selected if r["frequency"] == 400])
                )
                if variant == "narrow"
                else None,
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
