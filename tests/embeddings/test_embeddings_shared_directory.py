"""Two live `Embeddings` that name their files in one directory each get a file of their own.

The file name comes from the dataset's id and the extractor, so two datasets that claim one id, two views of one
source and two crop views of one source all give one name. Before the fix the second instance opened that file with
`mode="w+"`, which changed the first instance's rows or, when the second store was smaller, crashed the process
with a bus error on the next read.

Fixtures: datasets of eight random images from different seeds, so their rows differ; one dataset id, `"same"`, for
every dataset; `memory_threshold=0.0`, which makes any store with a path a memmap; and the default `FlattenExtractor`,
so the rows are the images themselves.
"""

import gc
from pathlib import Path

import numpy as np
import pytest

from dataeval import Embeddings
from dataeval.data import DetectionCrops, Limit, View
from dataeval.extractors import FlattenExtractor
from dataeval.protocols import Array, DatasetMetadata, DatumMetadata, ObjectDetectionTarget

pytestmark = pytest.mark.required


class _Named:
    """Image dataset whose metadata gives the same id for every seed."""

    def __init__(self, seed: int, n: int = 8) -> None:
        self.images = np.random.default_rng(seed).random((n, 1, 4, 4), dtype=np.float32)
        self.metadata = {"id": "same"}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        return self.images[i], np.zeros(1, dtype=np.float32), {"id": i}


class _Target:
    def __init__(self) -> None:
        self.boxes = np.array([[4.0, 4.0, 20.0, 20.0]], dtype=np.float32)
        self.labels = np.array([0], dtype=np.intp)
        self.scores = np.ones(1, dtype=np.float32)


class _Detections:
    """Object detection dataset of 32 x 32 images with one box each."""

    def __init__(self, seed: int, n: int = 8) -> None:
        self.images = np.random.default_rng(seed).integers(0, 255, (n, 3, 32, 32)).astype(np.uint8)
        self.metadata = DatasetMetadata(id="same", index2label={0: "a"})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[Array, ObjectDetectionTarget, DatumMetadata]:
        return self.images[i], _Target(), {"id": i}


def _memmap(dataset, directory) -> Embeddings:
    return Embeddings(dataset, batch_size=4, path=directory, memory_threshold=0.0)


def test_two_datasets_with_one_id_keep_their_own_rows(tmp_path):
    first = _memmap(_Named(0), tmp_path)
    second = _memmap(_Named(1), tmp_path)
    assert hash(first) == hash(second)
    assert first.path != second.path
    before = np.array(first.compute()[:])
    second.compute()
    assert isinstance(first._embeddings, np.memmap)
    assert np.array_equal(first[:], before)
    assert not np.array_equal(second[:], before)


def test_a_smaller_view_of_the_same_source_gets_its_own_file(tmp_path):
    # Paths are checked before any read: on a shared file the first read after the second compute crashed.
    source = _Named(0)
    whole = _memmap(source, tmp_path)
    part = _memmap(View(source, Limit(4)), tmp_path)
    assert whole.path != part.path
    before = np.array(whole.compute()[:])
    part.compute()
    assert np.array_equal(whole[:], before)
    assert np.array_equal(part[:], before[:4])


def test_crops_with_different_settings_keep_their_own_rows(tmp_path):
    source = _Detections(0)
    tight = _memmap(DetectionCrops(source, padding=0.0, square="off"), tmp_path)
    padded = _memmap(DetectionCrops(source, padding=0.5, square="off"), tmp_path)
    assert tight.path != padded.path
    before = np.array(tight.compute()[:])
    padded.compute()
    assert np.array_equal(tight[:], before)
    assert padded[:].shape != before.shape


def test_new_on_a_dataset_with_the_same_id_writes_its_own_file(tmp_path):
    train = _memmap(_Named(0), tmp_path).compute()
    before = np.array(train[:])
    test = train.new(_Named(1)).compute()
    assert test.path != train.path
    assert np.array_equal(train[:], before)


def test_new_names_a_file_in_a_directory_not_yet_made_whose_name_has_a_dot(tmp_path):
    # In memory, so the directory is not made, and "v1.2" alone would read as a file name.
    train = Embeddings(_Named(0), batch_size=4, path=tmp_path / "v1.2" / "train.npy")
    first, second = train.new(_Named(1)), train.new(_Named(2))
    assert first.path is not None
    assert second.path is not None
    assert first.path.parent == tmp_path / "v1.2"
    assert first.path != second.path


def test_instances_bound_later_get_their_own_files(tmp_path):
    first = Embeddings(batch_size=4, path=tmp_path, memory_threshold=0.0)
    second = Embeddings(batch_size=4, path=tmp_path, memory_threshold=0.0)
    assert first.path != second.path
    before = np.array(first.bind(_Named(0)).compute()[:])
    second.bind(_Named(1)).compute()
    assert np.array_equal(first[:], before)


def test_a_relative_path_and_a_symlink_to_one_directory_are_the_same_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "store").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "store")
    first = _memmap(_Named(0), Path("store"))
    second = _memmap(_Named(1), tmp_path / "link")
    assert first.path is not None
    assert second.path is not None
    assert first.path.resolve() != second.path.resolve()


def test_compute_multi_on_two_datasets_with_one_id_keeps_their_rows(tmp_path):
    flat = {"flat": FlattenExtractor()}
    first = Embeddings.compute_multi(_Named(0), flat, batch_size=4, path=tmp_path, memory_threshold=0.0)["flat"]
    before = np.array(first[:])
    second = Embeddings.compute_multi(_Named(1), flat, batch_size=4, path=tmp_path, memory_threshold=0.0)["flat"]
    assert isinstance(first._embeddings, np.memmap)
    assert first.path != second.path
    assert np.array_equal(first[:], before)


def test_an_array_returned_by_a_call_keeps_its_rows_after_the_next_call(tmp_path):
    # The instance a call makes is gone once the call returns, and its array still reads the file.
    emb = Embeddings(batch_size=4, path=tmp_path, memory_threshold=0.0)
    first = emb(_Named(0))
    before = np.array(first)
    emb(_Named(1))
    assert np.array_equal(np.asarray(first), before)


def test_an_array_of_a_freed_instance_keeps_its_rows(tmp_path):
    rows = np.asarray(_memmap(_Named(0), tmp_path))
    before = np.array(rows)
    gc.collect()
    _memmap(_Named(1), tmp_path).compute()
    assert np.array_equal(rows, before)


def test_a_lone_instance_keeps_the_name_of_its_hash(tmp_path):
    lone = _memmap(_Named(0), tmp_path)
    assert lone.path == tmp_path / f"emb-{hash(lone)}.npy"


def test_a_name_is_free_again_once_its_instance_is_gone(tmp_path):
    # A script run twice in one process, or rerun, reuses its file rather than leaving a new one behind each time.
    first = _memmap(_Named(0), tmp_path).compute()
    name = first.path
    del first
    gc.collect()
    assert _memmap(_Named(1), tmp_path).path == name


def test_setting_the_same_directory_again_keeps_the_instance_file(tmp_path):
    first = _memmap(_Named(0), tmp_path)
    second = _memmap(_Named(1), tmp_path).compute()
    name = second.path
    second.path = tmp_path
    assert first.path != name
    assert second.path == name


def test_setting_the_same_directory_again_keeps_the_file_after_a_lower_name_is_freed(tmp_path):
    first = _memmap(_Named(0), tmp_path)
    second = _memmap(_Named(1), tmp_path).compute()
    name = second.path
    del first
    gc.collect()
    second.path = tmp_path
    assert second.path == name
    assert isinstance(second._embeddings, np.memmap)
    assert str(second._embeddings.filename) == str(name)
