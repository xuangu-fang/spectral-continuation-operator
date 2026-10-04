import numpy as np

from spectral_continuation.audit_waveguide_generator_identification import generator_geometry
from spectral_continuation.audit_waveguide_known_family import (
    derivatives,
    pack,
    projection,
    public_spec,
    unpack,
)
from spectral_continuation.coherent_waveguide_data import environment


def test_known_family_parameter_roundtrip_and_hidden_scrub():
    actual = environment(20261332)
    visible = public_spec(actual)
    assert visible["contrasts_m_s"] == [0.0] * 3
    assert visible["centers_m"] != actual["centers_m"]
    np.testing.assert_allclose(pack(unpack(visible, pack(actual))), pack(actual), atol=1e-14)


def test_known_family_projection_sensitivity_shape():
    actual = environment(20261333)
    visible = public_spec(actual)
    geometry = generator_geometry(visible, 8, 6, modes=6, quadrature_nodes=32)
    parameters = pack(actual)
    coefficients = projection(visible, geometry, parameters)
    tangent = derivatives(visible, geometry, parameters)
    assert coefficients.shape == (8, 6)
    assert tangent.shape == (8, 6, 15)
    assert np.isfinite(tangent).all()
    assert np.linalg.norm(tangent) > 0
