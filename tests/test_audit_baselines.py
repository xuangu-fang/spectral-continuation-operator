import numpy as np
import torch

from spectral_continuation.audit_baselines import (
    apply_global_complex_ridge,
    fit_global_complex_ridge,
)
from spectral_continuation.data import as_complex


def test_global_complex_ridge_recovers_linear_mapping():
    generator = torch.Generator().manual_seed(4)
    first = torch.randn(12, 2, 4, 5, generator=generator)
    second = torch.randn(12, 2, 4, 5, generator=generator)
    target_complex = (1.2 + 0.3j) * as_complex(first) + (-0.4 + 0.1j) * as_complex(second)
    target = torch.stack((target_complex.real, target_complex.imag), dim=1)
    fields = {10.0: first, 20.0: second, 40.0: target}
    coefficients = fit_global_complex_ridge(fields, (10.0, 20.0), 40.0, np.arange(10), ridge=1e-10)
    prediction = apply_global_complex_ridge(fields, (10.0, 20.0), np.arange(10, 12), coefficients)
    torch.testing.assert_close(prediction, target[10:], atol=1e-5, rtol=1e-5)
