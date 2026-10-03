import numpy as np

from spectral_continuation.coarse_waveguide_operator import (
    encode_operator,
    query_coefficients,
    query_field,
)
from spectral_continuation.coherent_waveguide_data import environment, solve


def test_stationary_one_cell_is_exact_at_unordered_continuous_ranges():
    spec = environment(8)
    spec["contrasts_m_s"] = [0, 0, 0]
    reference, _, ranges, depths, _ = solve(spec, [60], 65, 65, 24, 160)
    latent = encode_operator(spec, 1, 24)
    prediction = query_field(latent, 60, ranges[::-1], depths)
    np.testing.assert_allclose(prediction, reference[0, :, ::-1], atol=1e-10)


def test_matrix_transport_preserves_source_linearity_and_passivity():
    spec = environment(8)
    latent = encode_operator(spec, 16, 24)
    ranges = np.linspace(0, 2000, 99)
    first = query_coefficients(latent, 100, ranges)
    second_latent = {**latent, "initial": latent["initial"][::-1]}
    second = query_coefficients(second_latent, 100, ranges)
    combined = query_coefficients(
        {**latent, "initial": latent["initial"] + 2j * second_latent["initial"]}, 100, ranges
    )
    np.testing.assert_allclose(combined, first + 2j * second, atol=1e-12)
    assert np.all(np.diff(np.linalg.norm(first, axis=1)) <= 1e-12)


def test_mass_is_positive_and_latent_has_no_frequency_or_output_grid():
    latent = encode_operator(environment(8), 16, 24)
    assert np.min(np.linalg.eigvalsh(latent["mass"])) > 0
    assert "frequency" not in latent and "field" not in latent and "ranges" not in latent
