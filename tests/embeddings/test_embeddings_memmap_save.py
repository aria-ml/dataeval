"""An `Embeddings` stored in a memmap saves to any path it is given, and its file is a `.npy` that `np.load` reads.

Before the fix, `save(path)` on a memmap store only flushed the instance's own file and wrote nothing at `path`, and
the memmap file had no NumPy header, so `np.load` reported it as pickled data.

Fixtures: datasets of eight random float32 images of shape (1, 4, 4), from a seed each test picks, so two datasets of
different seeds differ in every row; `memory_threshold=0.0`, which makes any store with a path a memmap; and the
default `FlattenExtractor`, so the rows are the images themselves.
"""

import os

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


def _memmap(directory, seed: int = 0) -> Embeddings:
    emb = Embeddings(_Images(seed), batch_size=4, path=directory, memory_threshold=0.0).compute()
    assert isinstance(emb._embeddings, np.memmap)
    return emb


def test_the_memmap_file_loads_with_np_load(tmp_path):
    emb = _memmap(tmp_path)
    emb.save()
    assert emb.path is not None
    assert np.array_equal(np.load(emb.path), emb[:])
    assert np.array_equal(np.load(emb.path, mmap_mode="r"), emb[:])


def test_save_to_another_file_writes_the_rows_there(tmp_path):
    emb = _memmap(tmp_path / "store")
    target = tmp_path / "copy.npy"
    emb.save(target)
    assert np.array_equal(np.load(target), emb[:])


def test_save_to_another_directory_writes_a_file_there(tmp_path):
    emb = _memmap(tmp_path / "store")
    (tmp_path / "out").mkdir()
    emb.save(tmp_path / "out")
    saved = list((tmp_path / "out").glob("emb-*.npy"))
    assert len(saved) == 1
    assert np.array_equal(np.load(saved[0]), emb[:])


def test_save_to_another_file_keeps_the_instance_store(tmp_path):
    emb = _memmap(tmp_path / "store")
    own = emb.path
    emb.save(tmp_path / "copy.npy")
    assert emb.path == own
    assert isinstance(emb._embeddings, np.memmap)
    assert str(emb._embeddings.filename) == str(own)


def test_save_to_its_own_file_by_another_spelling_flushes_it(tmp_path):
    # The same file through a symlinked directory: a copy onto itself would raise or empty it.
    emb = _memmap(tmp_path / "store")
    assert emb.path is not None
    (tmp_path / "link").symlink_to(tmp_path / "store")
    rows = np.array(emb[:])
    emb.save(tmp_path / "link" / emb.path.name)
    assert np.array_equal(np.load(emb.path), rows)


def test_setting_a_new_path_writes_the_rows_there(tmp_path):
    emb = _memmap(tmp_path / "store")
    target = tmp_path / "moved.npy"
    emb.path = target
    assert emb.path == target
    assert np.array_equal(np.load(target), emb[:])


def test_setting_a_new_path_moves_the_store_onto_it(tmp_path):
    emb = _memmap(tmp_path / "store")
    old = emb.path
    assert old is not None
    rows = np.array(emb[:])
    target = tmp_path / "moved.npy"
    emb.path = target
    assert isinstance(emb._embeddings, np.memmap)
    assert str(emb._embeddings.filename) == str(target)
    old.unlink()
    emb.save()
    assert np.array_equal(emb[:], rows)
    assert np.array_equal(np.load(target), rows)


def test_save_onto_the_file_of_another_instance_is_refused_and_leaves_its_rows(tmp_path):
    first = _memmap(tmp_path / "first", seed=0)
    second = _memmap(tmp_path / "second", seed=1)
    rows = np.array(first[:])
    with pytest.raises(ValueError, match="another Embeddings"):
        second.save(first.path)
    assert np.array_equal(first[:], rows)


def test_setting_the_file_of_another_instance_as_path_is_refused_and_changes_nothing(tmp_path):
    first = _memmap(tmp_path / "first", seed=0)
    second = _memmap(tmp_path / "second", seed=1)
    own = second.path
    rows = np.array(first[:])
    with pytest.raises(ValueError, match="another Embeddings"):
        second.path = first.path
    assert second.path == own
    assert np.array_equal(first[:], rows)


def test_an_in_memory_save_onto_the_file_of_another_instance_is_refused(tmp_path):
    first = _memmap(tmp_path / "first", seed=0)
    rows = np.array(first[:])
    second = Embeddings(_Images(1), batch_size=4).compute()
    with pytest.raises(ValueError, match="another Embeddings"):
        second.save(first.path)
    assert np.array_equal(first[:], rows)


def test_save_to_a_hard_link_of_its_own_file_flushes_it(tmp_path):
    emb = _memmap(tmp_path / "store")
    assert emb.path is not None
    rows = np.array(emb[:])
    os.link(emb.path, tmp_path / "hard.npy")
    emb.save(tmp_path / "hard.npy")
    assert np.array_equal(np.load(emb.path), rows)


@pytest.mark.parametrize("threshold", [0.0, 1.0], ids=["memmap", "in-memory"])
def test_a_file_name_without_npy_is_kept_as_given(tmp_path, threshold):
    emb = Embeddings(_Images(0), batch_size=4, path=tmp_path / "store", memory_threshold=threshold).compute()
    emb.save(tmp_path / "rows.bin")
    assert np.array_equal(np.load(tmp_path / "rows.bin"), emb[:])
    assert not (tmp_path / "rows.bin.npy").exists()


def test_an_in_memory_store_saves_as_before(tmp_path):
    emb = Embeddings(_Images(0), batch_size=4, path=tmp_path).compute()
    assert not isinstance(emb._embeddings, np.memmap)
    target = tmp_path / "copy.npy"
    emb.save(target)
    assert np.array_equal(np.load(target), emb[:])
