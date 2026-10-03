import numpy as np

from spectral_continuation.audit_modal_query_components import compose, interpolated_states
from spectral_continuation.audit_modal_spectral_transport import modal_query


def test_component_state_reproduces_deployable_interpolator():
    rng = np.random.default_rng(18)
    f, fq = np.array([90.0, 100.0, 110.0, 120.0]), np.array([93.0, 104.0, 117.0])
    modes = []
    for frequency in f:
        count = 2 if frequency < 105 else 3
        k = np.linspace(0.3, 0.15, count) * frequency / 100
        a = rng.normal(size=(6, count)) + 1j * rng.normal(size=(6, count))
        modes.append((k, a))
    x = np.linspace(50, 750, 100)
    kq, aq = interpolated_states(f, modes, fq)
    expected = modal_query(f, modes, fq, x, 1500.0, False)
    actual = np.stack([compose(k, a, x) for k, a in zip(kq, aq, strict=True)])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
