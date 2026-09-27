import torch

from spectral_continuation.train_active_wave_query_policy import ActiveWaveQueryPolicy


def test_active_policy_initializes_to_analytic_ranking():
    model = ActiveWaveQueryPolicy(4, 7, 16, 3.0)
    belief = torch.randn(3, 8)
    action = torch.randn(5, 7)
    risk = torch.tensor([3.0, 1.0, 2.0, 4.0, 5.0])
    score = model(belief, action, risk)
    assert score.shape == (3, 5)
    assert torch.equal(score.argmax(dim=-1), torch.ones(3, dtype=torch.long))
