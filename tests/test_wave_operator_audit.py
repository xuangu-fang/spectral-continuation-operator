import numpy as np

from spectral_continuation.audit_wave_operator_representation import (
    _estimate_reciprocity_metric,
    _rbf_weights,
)


def test_rbf_weights_have_expected_shape():
    context = np.asarray([[0.0], [0.5], [1.0]])
    query = np.asarray([[0.25], [0.75]])
    assert _rbf_weights(context, query, 0.3, 1e-3).shape == (2, 3)


def test_reciprocity_metric_recovers_relative_weights():
    weights = np.asarray([0.4, 0.7, 1.2])
    pixels = np.asarray([[0, 0], [0, 1], [1, 0]])
    symmetric = np.asarray(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
        dtype=np.complex64,
    )
    field = np.zeros((3, 2, 2, 2), dtype=np.complex64)
    for source in range(3):
        for receiver in range(3):
            field[source, :, pixels[receiver, 0], pixels[receiver, 1]] = (
                symmetric[receiver, source] / weights[source]
            )
    estimate = _estimate_reciprocity_metric(field, pixels, [0, 1])
    estimate /= np.exp(np.mean(np.log(estimate)))
    expected = weights / np.exp(np.mean(np.log(weights)))
    np.testing.assert_allclose(estimate, expected, rtol=1e-5)
