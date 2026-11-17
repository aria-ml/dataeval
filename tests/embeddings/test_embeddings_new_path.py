"""An `Embeddings` made by `new()`, or by calling one on other data, stores its rows in a file of its own.

Fixtures: datasets of eight random float32 images of shape (1, 4, 4) from different seeds, so the embeddings of two of
them differ in every row; `memory_threshold=0.0`, which makes any store with a path a memmap; and the default
`FlattenExtractor`, so the rows are the images themselves.
"""

import numpy as np
import pytest

from dataeval import Embeddings

pytestmark = pytest.mark.required


class _Images:
    def __init__(self, seed: int, n: int = 8) -> None:
        self.images = np.random.default_rng(seed).random((n, 1, 4, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.images[i]


def test_new_writes_its_own_file_and_leaves_the_original_store_intact(tmp_path):
    train = Embeddings(_Images(0), batch_size=4, path=tmp_path, memory_threshold=0.0).compute()
    assert isinstance(train._embeddings, np.memmap)
    before = np.array(train[:])
    test = train.new(_Images(1)).compute()
    assert test._path is not None
    assert train._path is not None
    assert test._path != train._path
    assert test._path.parent == train._path.parent
    assert np.array_equal(train[:], before)
    assert not np.array_equal(test[:], before)


def test_calling_on_other_data_leaves_the_bound_rows_intact(tmp_path):
    train = Embeddings(_Images(0), batch_size=4, path=tmp_path, memory_threshold=0.0).compute()
    before = np.array(train[:])
    other = np.asarray(train(_Images(2)))
    assert np.array_equal(train[:], before)
    assert not np.array_equal(other, before)


def test_saving_a_new_instance_leaves_the_original_file_intact(tmp_path):
    first = Embeddings(_Images(0), batch_size=4, path=tmp_path).compute()
    first.save()
    assert first._path is not None
    saved = np.load(first._path)
    second = first.new(_Images(1)).compute()
    second.save()
    assert np.array_equal(np.load(first._path), saved)


def test_an_explicit_file_name_is_not_reused_by_new(tmp_path):
    train = Embeddings(_Images(0), batch_size=4, path=tmp_path / "train.npy", memory_threshold=0.0).compute()
    before = np.array(train[:])
    test = train.new(_Images(1)).compute()
    assert test._path is not None
    assert test._path.parent == tmp_path
    assert test._path.name != "train.npy"
    assert np.array_equal(train[:], before)


def test_new_without_a_path_has_none():
    assert Embeddings(_Images(0), batch_size=4).new(_Images(1))._path is None
