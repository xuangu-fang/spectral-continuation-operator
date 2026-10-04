import numpy as np

from spectral_continuation.audit_helmholtz_generator_identification import laplacian
from spectral_continuation.audit_helmholtz_sparse_linear_info import (
    receivers,
    tangent_response,
)
from spectral_continuation.coherent_helmholtz_data import sample_environment


def test_speed_tangent_matches_finite_difference():
    axis = np.linspace(0, 1, 17)
    spec = {"source": [0.22, 0.43], "source_width": 0.025,
            "sponge_width": 0.16, "sponge_strength": 2.0,
            "background_loss": 0.002, "centers": [], "widths": [], "contrasts": []}
    _, loss, source = sample_environment(spec, axis)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    speed = 1 + 0.05 * np.cos(2 * np.pi * xx) * np.cos(np.pi * yy)
    direction = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    lap = laplacian(17)
    field, tangent = tangent_response(speed, direction[None], loss, source, 3.0, lap)
    step = 1e-5
    perturbed, _ = tangent_response(
        speed + step * direction, direction[None], loss, source, 3.0, lap
    )
    finite = (perturbed - field) / step
    relative = np.linalg.norm(finite - tangent[0]) / np.linalg.norm(tangent[0])
    assert relative < 5e-4


def test_four_line_receiver_sampling_shape():
    values = np.ones((4, 129, 129), dtype=complex)
    assert receivers(values, np.arange(4)).shape == (4, 65, 4)
    np.testing.assert_array_equal(receivers(values, np.array([1])), 1)
