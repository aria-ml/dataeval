"""`Embeddings._ingest` stores rows another loop computed, as the batch loop stores its own.

Fixtures: `SimpleDataset` of ten random float32 images of shape (1, 2, 2), and hand-made float32 rows of eight
values, so what is stored can be compared with what was given exactly.
"""

import numpy as np
import pytest

from dataeval import Embeddings
from tests.conftest import SimpleDataset

pytestmark = pytest.mark.required


@pytest.fixture
def dataset() -> SimpleDataset:
    return SimpleDataset(10, (1, 2, 2))


def _rows(n: int) -> np.ndarray:
    return np.arange(n * 8, dtype=np.float32).reshape(n, 8)


def test_ingest_fills_storage_as_a_batch_does(dataset):
    embeddings = Embeddings(dataset, batch_size=4)
    embeddings._ingest([0, 1, 2, 3], _rows(4))
    assert embeddings._cached_idx == {0, 1, 2, 3}
    assert np.array_equal(embeddings._embeddings[[0, 1, 2, 3]], _rows(4))


def test_ingest_honors_an_explicit_memmap_choice(dataset, tmp_path):
    on_disk = Embeddings(dataset, batch_size=4, path=tmp_path / "a", memory_threshold=1.0)
    on_disk._ingest([0, 1], _rows(2), use_memmap=True)
    assert isinstance(on_disk._embeddings, np.memmap)

    in_memory = Embeddings(dataset, batch_size=4, path=tmp_path / "b", memory_threshold=0.0)
    in_memory._ingest([0, 1], _rows(2), use_memmap=False)
    assert not isinstance(in_memory._embeddings, np.memmap)


def test_estimated_bytes(dataset):
    assert Embeddings(dataset, batch_size=4)._estimated_bytes((8,)) == 320
