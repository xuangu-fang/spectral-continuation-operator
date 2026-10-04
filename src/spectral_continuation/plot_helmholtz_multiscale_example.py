"""Physical-speed, amplitude, phase, and background-relative phase example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .audit_helmholtz_multiscale_fullfield_classic import public_spec
from .audit_waveguide_spectral_context_robustness import digest
from .coherent_helmholtz_data import solve


def run(truth_path: Path, output: Path) -> None:
    with np.load(truth_path) as data:
        spec = json.loads(str(data["spec"]))
        fields = data["field"]
        frequencies = data["frequencies"]
        speed = data["speed"]
    if spec["seed"] != 20269600:
        raise ValueError("preselected first M0 pilot system changed")
    index = int(np.flatnonzero(frequencies == 17.0)[0])
    field = fields[index].astype(np.complex128)
    background, _, _, _ = solve(public_spec(spec), 257, [17.0])
    amplitude = np.abs(field)
    keep = amplitude >= np.quantile(amplitude, 0.05)
    phase = np.ma.masked_where(~keep, np.angle(field))
    relative = np.ma.masked_where(
        ~keep, np.angle(field * np.conj(background[0]))
    )
    source_x, source_y = spec["source"]
    panels = [
        (speed, "Speed c(x, y)", "viridis", None, None),
        (np.log10(amplitude / np.max(amplitude) + 1e-8),
         "log10 amplitude / max at 17", "magma", -5, 0),
        (phase, "Complex phase at 17", "twilight", -np.pi, np.pi),
        (relative, "Phase shift vs homogeneous", "twilight_shifted", -np.pi, np.pi),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.3), constrained_layout=True)
    for axis, (values, title, cmap, low, high) in zip(axes, panels, strict=True):
        image = axis.imshow(values, origin="lower", extent=(0, 1, 0, 1),
                            cmap=cmap, vmin=low, vmax=high)
        axis.scatter([source_x], [source_y], marker="*", color="white",
                     edgecolor="black", s=80, linewidth=0.5)
        axis.set_title(title, fontsize=10.5)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle("2D multiscale Helmholtz M0: preselected development system 20269600",
                 fontsize=13)
    fig.text(0.5, 0.01,
             "Phase panels hide only the lowest 5% target amplitudes for display; all metrics use the full ROI without alignment.",
             ha="center", fontsize=8.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "DISPLAY ONLY, preselected first M0 DEV system, direct grid257 f17 truth and direct homogeneous reference; bottom5% amplitude mask only in phase display, not any metric.",
        "truth_sha256": digest(truth_path),
        "implementation_sha256": digest(__file__),
        "system": spec["seed"], "frequency": 17.0,
        "source": spec["source"],
        "speed_min": float(np.min(speed)), "speed_max": float(np.max(speed)),
        "display_mask_fraction": float(np.mean(~keep)),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.truth, args.output)


if __name__ == "__main__":
    main()
