"""Record reusable development data and post-run code/artifact checksums.

This explicitly captures a post-run working-tree snapshot; it must not be
misrepresented as proof that every earlier run used byte-identical source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(root, output):
    base = Path("/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets")
    datasets = {}
    dataset_specs = [
        ("coherent_helmholtz_development_20261003", 40),
        ("coherent_helmholtz_fine_queries_20261003", 4),
        ("coherent_ocean_fine_queries_20261003", 2),
        ("coherent_ocean_fractional_queries_20261003", 4),
        ("coherent_helmholtz_fractional_queries_20261003", 4),
        ("coherent_structural_narrow_source_20261003", 4),
        ("coherent_structural_high_contrast_20261003", 4),
        ("coherent_structural_narrow_high_contrast_20261003", 4),
        ("coherent_structural_reference_20261003", 2),
        ("coherent_waveguide_convergence_20261003", 12),
        ("coherent_waveguide_spectral_pilot_20261003", 2),
        ("coherent_waveguide_development_20261003", 40),
        ("coherent_waveguide_training_cache_20261003", 512),
        ("coherent_waveguide_generator_query_transfer_20261003", 16),
        ("coherent_waveguide_adaptive_modes_truth_20261003", 2),
        ("coherent_waveguide_narrow_source_modes_truth_20261003", 2),
        ("coherent_waveguide_narrow_source_modes_expansion_20261003", 6),
        ("coherent_waveguide_frozen_system_audit_20261003/original", 64),
        ("coherent_waveguide_frozen_system_audit_20261003/narrow", 64),
        ("coherent_waveguide_operator_action_20261004", 11),
        ("coherent_waveguide_frequency_scaling_20261004", 51),
        ("coherent_waveguide_new_systems_high_frequency_20261004", 64),
        ("coherent_waveguide_spectral_medium_20261004", 64),
        ("coherent_waveguide_spectral_confirm_20261004", 64),
        ("coherent_waveguide_spectral_amplitude_matched_dev_20261004", 16),
    ]
    frequency_curve_name = "coherent_waveguide_spectral_frequency_curve_20261004"
    if (base / frequency_curve_name).exists():
        dataset_specs.append((frequency_curve_name, 40))
    taper_shift_name = "coherent_waveguide_spectral_taper_shift_20261004"
    if (base / taper_shift_name).exists():
        dataset_specs.append((taper_shift_name, 32))
    speed_confirm_name = "coherent_waveguide_learned_speed_manifold_confirm_20261004"
    if (base / speed_confirm_name).exists():
        dataset_specs.append((speed_confirm_name, 32))
    speed_scaling_name = "coherent_waveguide_speed_resolution_scaling_20261004"
    if (base / speed_scaling_name).exists():
        dataset_specs.append((speed_scaling_name, 16))
    multiscale_helmholtz_name = "coherent_helmholtz_multiscale_pilot_20261004"
    if (base / multiscale_helmholtz_name).exists():
        dataset_specs.append((multiscale_helmholtz_name, 8))
    for name, expected in dataset_specs:
        paths = sorted((base / name).glob(
            "system_*_grid257.npz" if name == multiscale_helmholtz_name else "*.npz"
        ))
        if len(paths) != expected:
            raise ValueError(f"incomplete dataset {name}: {len(paths)} != {expected}")
        records = []
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                value_key = next(
                    key for key in ["field", "modal_coefficients", "transfer", "coarse", "coarse140"]
                    if key in z
                )
                field = z[value_key]
                frequency_key = next(
                    (key for key in ["frequencies", "frequencies_hz", "frequency_hz"] if key in z),
                    None,
                )
                if frequency_key:
                    frequencies = z[frequency_key].tolist()
                else:
                    metadata = json.loads((base / name / "config.json").read_text())
                    frequencies = metadata.get(
                        "audit_frequencies",
                        metadata.get("query_frequencies", [metadata.get("frequency")]),
                    )
                if not np.iscomplexobj(field) or not np.isfinite(field).all():
                    raise ValueError(f"invalid complex data: {path}")
                extra_arrays = {}
                if "fine" in z:
                    fine = z["fine"]
                    if not np.iscomplexobj(fine) or not np.isfinite(fine).all():
                        raise ValueError(f"invalid audit pair: {path}")
                    extra_arrays["fine"] = {"shape": list(fine.shape), "dtype": str(fine.dtype)}
                for key in sorted(k for k in z.files if k.startswith("fine") and k != "fine"):
                    fine = z[key]
                    if not np.iscomplexobj(fine) or not np.isfinite(fine).all():
                        raise ValueError(f"invalid audit pair: {path} {key}")
                    extra_arrays[key] = {"shape": list(fine.shape), "dtype": str(fine.dtype)}
                records.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                        "shape": list(field.shape),
                        "dtype": str(field.dtype),
                        "value_key": value_key,
                        "frequencies": frequencies,
                        "additional_arrays": extra_arrays,
                    }
                )
        independent = name.startswith("coherent_waveguide_frozen_system_audit_20261003/")
        new_independent = name == "coherent_waveguide_new_systems_high_frequency_20261004"
        spectral_medium = name == "coherent_waveguide_spectral_medium_20261004"
        spectral_confirm = name == "coherent_waveguide_spectral_confirm_20261004"
        amplitude_control = name == "coherent_waveguide_spectral_amplitude_matched_dev_20261004"
        frequency_curve = name == frequency_curve_name
        taper_shift = name == taper_shift_name
        speed_confirm = name == speed_confirm_name
        speed_scaling = name == speed_scaling_name
        metadata_paths = set((base / name).glob("*.json"))
        if name == multiscale_helmholtz_name:
            training_path = base / name / "training_medium_labels.npz"
            with np.load(training_path, allow_pickle=False) as z:
                if z["speed"].shape != (512, 65, 65) or not np.isfinite(z["speed"]).all():
                    raise ValueError("invalid multiscale Helmholtz training medium labels")
            metadata_paths.add(training_path)
        if independent:
            metadata_paths.update((base / name).parent.glob("*.json"))
        datasets[name] = {
            "evidence": (
                "independent64IID environments20262300–363; candidates frozen before generation; original/narrow are paired queries NOT128independent systems; no current-model tuning"
                if independent
                else "independent32IID systems20263400–431; prior16/32 candidates frozen before new data generation; distinct from prior64IID and not a test of later depth64 dev diagnostic"
                if new_independent
                else "new correlated-spectral medium statistical family; independent32 systems20264600–631; same one-way waveguide PDE; prior16/32 frozen before new truth; not a second physical domain or a deep-neural-method test"
                if spectral_medium
                else "independent new32 correlated-spectral systems20265000–031; rank256 method and gates frozen before truth generation; same PDE and medium family, NOT a second physical domain"
                if spectral_confirm
                else "posthoc first8 spectral development systems with Gaussian-family-matched RMS; tests contrast amplitude only, not independent method confirmation"
                if amplitude_control
                else "conditional fixed-code five-frequency follow-up on the first8 confirmation systems; NOT a second independent cohort; run only after confirmation gate"
                if frequency_curve
                else "fresh16 correlation-scale-shift systems20267000–015, RMS matched, taper5 training to taper10 test; post-main-result fixed-method OOD diagnostic within the SAME one-way PDE, not a second physics domain"
                if taper_shift
                else "fresh16 taper7.5 systems20268000–015, same one-way PDE and 16x16 cosine medium family; statistical INTERPOLATION between training taper5/10; learned speed variable and gates frozen before truth at git5fe88a5, NOT a second physics domain"
                if speed_confirm
                else "conditional first8 ALREADY-USED taper7.5 confirmation systems, fixed learned speed codes queried at1000/1200Hz with 4 physical decoder resolutions; 16 new target NPZ, NOT new independent environments or new PDE; prereg git a7363b7"
                if speed_scaling
                else "M0 multiscale TWO-WAY Helmholtz DATA PILOT ONLY: 8 direct grid257 development fields, 512 training speed labels stored separately as metadata, two grid257/513 numeric audit systems; NOT independent method test or completed cross-PDE learning gate"
                if name == multiscale_helmholtz_name
                else "development/training only; not independent test evidence"
            ),
            "files": records,
            "metadata": [
                {"path": str(path), "sha256": digest(path)} for path in sorted(metadata_paths)
            ],
        }
    sources = set()
    for pattern in [
        "*coherent*.py",
        "*learned_wave_chart.py",
        "*modal*.py",
        "*phase*rom*.py",
        "*phase_reduced*.py",
        "*fundamental_solution*.py",
        "*rom_grid*.py",
        "*continuous_phase*.py",
        "*contrast_source*.py",
        "*source_cache*.py",
        "*source_space*.py",
        "*waveguide*.py",
        "*frozen_generator*.py",
        "*generator_frozen*.py",
        "*generator_representation*.py",
        "*generator_frequency_scaling*.py",
        "*generator_new_systems*.py",
        "*helmholtz*.py",
        "*separable*.py",
        "*learned_speed*.py",
        "*speed_variable*.py",
        "*speed_resolution*.py",
        "*variable_projection*.py",
        "*q_capacity*.py",
        "*q768*.py",
    ]:
        sources.update((root / "src/spectral_continuation").glob(pattern))
    sources.update((root / "tests").glob("test_*coherent*.py"))
    sources.update((root / "tests").glob("test_*phase*rom*.py"))
    sources.update((root / "tests").glob("test_*fundamental_solution*.py"))
    for pattern in [
        "test_*contrast_source*.py",
        "test_*source_space*.py",
        "test_*waveguide*.py",
        "test_*frozen_generator*.py",
        "test_*helmholtz*.py",
        "test_*separable*.py",
        "test_*learned_speed*.py",
        "test_*q_capacity*.py",
    ]:
        sources.update((root / "tests").glob(pattern))
    artifacts = []
    dirs = sorted(
        set((root / "artifacts").glob("coherent_*_20261003"))
        | set((root / "artifacts").glob("coherent_*_20261004"))
        | set((root / "artifacts").glob("learned_wave_chart_*_20261003"))
    )
    for directory in dirs:
        if directory == output:
            continue
        files = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".pt", ".npz"):
                files.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                    }
                )
        artifacts.append({"directory": str(directory.relative_to(root)), "files": files})
    inventory = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "capture_type": "post-run working-tree snapshot; not an at-launch source attestation",
        "git_base": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "large_files": "NPZ and PT intentionally local-only; regenerate via stored config or resolve these local paths",
        "datasets": datasets,
        "source_snapshot_sha256": {
            str(path.relative_to(root)): digest(path) for path in sorted(sources)
        },
        "artifacts": artifacts,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    print(
        json.dumps(
            {
                "datasets": len(datasets),
                "dataset_files": sum(len(v["files"]) for v in datasets.values()),
                "artifact_directories": len(artifacts),
                "output": str(output),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/coherent_stage1_inventory_20261003")
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
