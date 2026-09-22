"""Convert the legacy source-major geometry pool into system-major episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def convert(input_path: Path, output_path: Path) -> None:
    data = np.load(input_path, allow_pickle=True)
    field = data["field"].reshape(9, 3, 10, 8, 128, 128).transpose(2, 1, 0, 3, 4, 5)
    speed = data["speed"].reshape(9, 3, 10, 128, 128).transpose(2, 1, 0, 3, 4)
    # The coefficient map is identical across source replicas.  Keep one copy.
    speed = speed[:, :, 0].reshape(30, 128, 128)
    field = field.reshape(30, 9, 8, 128, 128)
    source = data["source"].reshape(9, 3, 10, 3)[:, 0, 0, 1:]
    boundary = np.broadcast_to(
        np.asarray(["open", "partial", "closed"])[None], (10, 3)
    ).reshape(30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        field=field,
        speed=speed,
        sources=source.astype(np.float32),
        frequencies=data["frequencies"],
        split=np.asarray(["test"] * len(field)),
        boundary=boundary,
    )
    print(f"saved {output_path}: {field.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
