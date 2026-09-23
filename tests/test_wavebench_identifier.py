import torch

from spectral_continuation.train_wavebench_identifier import response_coordinates


def test_wavebench_response_coordinates_are_phase_invariant():
    torch.manual_seed(17)
    value = torch.randn(2, 3, 2, 12, 12)
    angle = torch.randn(2, 3, 1, 1)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotated = torch.stack(
        (
            value[:, :, 0] * cosine - value[:, :, 1] * sine,
            value[:, :, 0] * sine + value[:, :, 1] * cosine,
        ),
        dim=2,
    )
    for representation in (
        "amplitude",
        "amplitude_soft_phase_gradient",
        "amplitude_edge_connection",
        "amplitude_transport_connection",
    ):
        torch.testing.assert_close(
            response_coordinates(value, representation),
            response_coordinates(rotated, representation),
            atol=2e-5,
            rtol=2e-5,
        )


def test_edge_connection_keeps_five_channels_in_two_dimensions():
    value = torch.randn(2, 4, 2, 16, 16)
    coordinate = response_coordinates(value, "amplitude_edge_connection")
    assert coordinate.shape == (2, 4, 5, 16, 16)
    transport = response_coordinates(value, "amplitude_transport_connection")
    assert transport.shape == coordinate.shape
