import numpy as np

from spectral_continuation.audit_waveguide_operator_rank import rank_statistics


def test_best_response_rank_matches_known_flat_and_decaying_spectra():
    ranks = [1, 2, 4]
    flat = rank_statistics(np.eye(4)[None], ranks)
    for rank in ranks:
        assert (
            abs(flat["best_fixed_span_relative_frobenius"][str(rank)] - np.sqrt(1 - rank / 4))
            < 1e-12
        )
    diagonal = np.diag([4.0, 3.0, 2.0, 1.0])[None]
    result = rank_statistics(diagonal, ranks)
    np.testing.assert_allclose(result["best_fixed_span_relative_frobenius"]["2"], np.sqrt(5 / 30))
