import numpy as np
import pytest

from spectral_continuation.summarize_frozen_generator_audit import paired_summary


def test_system_paired_bootstrap_preserves_fixed_ratio():
    baseline = np.arange(1.0, 9.0)
    result = paired_summary(0.75 * baseline, baseline, repetitions=100)
    assert result["systems"] == 8
    assert result["relative_improvement"] == 0.25
    np.testing.assert_allclose(result["paired_bootstrap_95_percent"], [0.25, 0.25])
    assert result["system_win_rate"] == 1


def test_bootstrap_rejects_misaligned_or_degenerate_errors():
    with pytest.raises(ValueError):
        paired_summary([1, 2], [1])
    with pytest.raises(ValueError):
        paired_summary([1, 2], [1, 0])
