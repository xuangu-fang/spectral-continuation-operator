import numpy as np

from spectral_continuation.build_population_acquisition_schedule import _atomic_json


def test_population_schedule_json_is_atomic_and_round_trips(tmp_path):
    path = tmp_path / "schedule.json"
    payload = {"schedule": np.asarray([[1, 2], [3, 4]]).tolist()}
    _atomic_json(path, payload)
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not list(tmp_path.glob("*.tmp"))
