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
    for name, expected in [
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
    ]:
        paths = sorted((base / name).glob("*.npz"))
        if len(paths) != expected:
            raise ValueError(f"incomplete dataset {name}: {len(paths)} != {expected}")
        records = []
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                field = z["field"]
                frequency_key = "frequencies" if "frequencies" in z else "frequencies_hz"
                if not np.iscomplexobj(field) or not np.isfinite(field).all():
                    raise ValueError(f"invalid complex data: {path}")
                records.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                        "shape": list(field.shape),
                        "dtype": str(field.dtype),
                        "frequencies": z[frequency_key].tolist(),
                    }
                )
        datasets[name] = {
            "evidence": "development only; all system identities already used for development",
            "files": records,
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
        "*separable*.py",
    ]:
        sources.update((root / "src/spectral_continuation").glob(pattern))
    sources.update((root / "tests").glob("test_*coherent*.py"))
    sources.update((root / "tests").glob("test_*phase*rom*.py"))
    sources.update((root / "tests").glob("test_*fundamental_solution*.py"))
    for pattern in [
        "test_*contrast_source*.py",
        "test_*source_space*.py",
        "test_*waveguide*.py",
        "test_*separable*.py",
    ]:
        sources.update((root / "tests").glob(pattern))
    artifacts = []
    dirs = sorted(
        set((root / "artifacts").glob("coherent_*_20261003"))
        | set((root / "artifacts").glob("learned_wave_chart_*_20261003"))
    )
    for directory in dirs:
        if directory == output:
            continue
        files = []
        for path in sorted(directory.iterdir()):
            if path.suffix in (".json", ".pt"):
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
