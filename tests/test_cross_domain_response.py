import numpy as np

from spectral_continuation.cross_domain_query_audit import (
    _fit_subspaces,
    _subspace_prediction,
    reciprocity_error,
    source_orders,
)
from spectral_continuation.cross_domain_response_data import (
    CrossDomainResponseArchive,
    validate_archive,
)


def _archive() -> CrossDomainResponseArchive:
    generator = np.random.default_rng(4)
    systems, sources, frequencies = 18, 4, 2
    coefficient = generator.normal(size=(systems, 2)) + 1j * generator.normal(size=(systems, 2))
    basis = generator.normal(size=(2, sources, sources)) + 1j * generator.normal(
        size=(2, sources, sources)
    )
    basis = 0.5 * (basis + basis.transpose(0, 2, 1))
    mean = np.eye(sources, dtype=np.complex64)[None]
    response = np.empty((systems, sources, frequencies, sources), dtype=np.complex64)
    for frequency in range(frequencies):
        response[:, :, frequency] = mean + np.einsum(
            "nk,kij->nij", coefficient * (frequency + 1), basis
        )
    split = np.asarray(["train"] * 12 + ["validation"] * 3 + ["test"] * 3)
    coordinate = np.stack((np.linspace(0, 1, sources), np.zeros(sources)), axis=1)
    return CrossDomainResponseArchive(
        responses=response,
        sources=coordinate.astype(np.float32),
        receivers=coordinate.astype(np.float32),
        frequencies=np.asarray([1.0, 2.0], dtype=np.float32),
        source_receiver_index=np.arange(sources),
        split=split,
        system_ids=np.asarray([f"system_{index}" for index in range(systems)]),
        valid_response_mask=np.ones((sources, sources), dtype=bool),
        provenance={"test": True},
    )


def test_cross_domain_archive_and_reciprocity() -> None:
    archive = _archive()
    validate_archive(archive)
    np.testing.assert_allclose(reciprocity_error(archive), 0.0, atol=1e-6)
    for target, orders in source_orders(4, 3, 9).items():
        assert all(target not in order for order in orders)


def test_response_subspace_completes_low_rank_table() -> None:
    archive = _archive()
    subspace = _fit_subspaces(archive, max_rank=2)[0]
    systems = np.flatnonzero(archive.split == "test")
    table = archive.responses[systems, :, 0]
    prediction = _subspace_prediction(
        subspace,
        table,
        context=np.asarray([0, 1, 2]),
        target=3,
        receiver_count=4,
        rank=2,
        ridge=1e-7,
    )
    np.testing.assert_allclose(prediction, table[:, 3], rtol=1e-4, atol=1e-4)
