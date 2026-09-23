import torch

from spectral_continuation.train_wavebench_identifier import response_coordinates


def test_windowed_coherence_shape_and_phase_invariance() -> None:
    generator = torch.Generator().manual_seed(17)
    value = torch.randn(3, 2, 2, 32, 32, generator=generator)
    first = response_coordinates(
        value, "amplitude_windowed_coherence", frequency_hz=(10, 40)
    )
    phase = 0.73
    rotated = torch.stack(
        (
            value[:, :, 0] * torch.cos(torch.tensor(phase))
            - value[:, :, 1] * torch.sin(torch.tensor(phase)),
            value[:, :, 0] * torch.sin(torch.tensor(phase))
            + value[:, :, 1] * torch.cos(torch.tensor(phase)),
        ),
        dim=2,
    )
    second = response_coordinates(
        rotated, "amplitude_windowed_coherence", frequency_hz=(10, 40)
    )
    assert first.shape == (3, 2, 17, 32, 32)
    torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)

    combined_first = response_coordinates(
        value, "amplitude_phase_windowed_coherence", frequency_hz=(10, 40)
    )
    combined_second = response_coordinates(
        rotated, "amplitude_phase_windowed_coherence", frequency_hz=(10, 40)
    )
    assert combined_first.shape == (3, 2, 21, 32, 32)
    torch.testing.assert_close(combined_first, combined_second, atol=2e-5, rtol=2e-5)

    packet_first = response_coordinates(
        value, "amplitude_phase_wave_packets", frequency_hz=(10, 40)
    )
    packet_second = response_coordinates(
        rotated, "amplitude_phase_wave_packets", frequency_hz=(10, 40)
    )
    assert packet_first.shape == (3, 2, 29, 32, 32)
    torch.testing.assert_close(packet_first, packet_second, atol=3e-5, rtol=3e-5)

    gram_first = response_coordinates(
        value, "amplitude_phase_wave_packet_gram", frequency_hz=(10, 40)
    )
    gram_second = response_coordinates(
        rotated, "amplitude_phase_wave_packet_gram", frequency_hz=(10, 40)
    )
    assert gram_first.shape == (3, 2, 29, 32, 32)
    torch.testing.assert_close(gram_first, gram_second, atol=3e-5, rtol=3e-5)

    multiscale_first = response_coordinates(
        value, "amplitude_phase_multiscale_packet_gram", frequency_hz=(10, 40)
    )
    multiscale_second = response_coordinates(
        rotated, "amplitude_phase_multiscale_packet_gram", frequency_hz=(10, 40)
    )
    assert multiscale_first.shape == (3, 2, 69, 32, 32)
    torch.testing.assert_close(multiscale_first, multiscale_second, atol=3e-5, rtol=3e-5)
