import json
from pathlib import Path

import numpy as np
import pytest

from spectral_continuation.prepare_helmholtz_sparse_dev import extract_public_observations


def test_sparse_context_shape_and_private_metadata_removed():
    config = json.loads(Path(
        "configs/coherent_helmholtz_multiscale_sparse_dev_20261004.json"
    ).read_text())
    field = np.arange(5 * 257 * 257, dtype=np.float32).reshape(5, 257, 257).astype(complex)
    spec = {"seed": 20269600, "source": [0.2, 0.4], "source_width": 0.025,
            "sponge_width": 0.16, "sponge_strength": 2.0, "background_loss": 0.002,
            "centers": [[0.3, 0.4]], "contrasts": [0.2], "widths": [0.1],
            "spectral_texture": [[0.7]], "texture_scale": 0.12}
    data = {"field": field, "frequencies": np.array([6., 10., 14., 11.3, 17.]),
            "spec": json.dumps(spec), "speed": np.full((257, 257), 0.8)}
    public = extract_public_observations(data, config)
    assert set(public) == {"samples", "frequencies", "x", "y", "public_spec"}
    assert public["samples"].shape == (3, 65, 4)
    assert public["samples"][0, 0, 0] == field[0, 32, 88]
    assert public["samples"][2, 64, 3] == field[2, 224, 208]
    assert set(json.loads(public["public_spec"])) == {
        "seed", "source", "source_width", "sponge_width", "sponge_strength",
        "background_loss",
    }


def test_query_frequency_in_context_rejected():
    config = json.loads(Path(
        "configs/coherent_helmholtz_multiscale_sparse_dev_20261004.json"
    ).read_text())
    config["context_frequencies_hz"] = [6., 10., 17.]
    field = np.zeros((5, 257, 257), dtype=complex)
    spec = {"seed": 20269600, "source": [0.2, 0.4], "source_width": 0.025,
            "sponge_width": 0.16, "sponge_strength": 2.0, "background_loss": 0.002}
    with pytest.raises(ValueError, match="leak"):
        extract_public_observations({"field": field,
                                     "frequencies": np.array([6., 10., 14., 11.3, 17.]),
                                     "spec": json.dumps(spec)}, config)
