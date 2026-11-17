"""`Embeddings.compute_multi` computes several extractors' embeddings in one pass, each item read once.

Fixtures: a dataset of ten `(image, 0, {})` items whose float32 images of shape (1, 4, 4) come from a seeded generator
and are made afresh on each read, which every read counts; two tiny CPU torch models with fixed seeds, so a solo run of
each gives a fixed array to compare against; and small callables that misbehave in one way each.
"""

from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from dataeval import Embeddings
from dataeval.config import use_load_threads
from dataeval.exceptions import ShapeMismatchError
from dataeval.extractors import FlattenExtractor, TorchExtractor

pytestmark = [pytest.mark.required, pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")]

N = 10


class _Dataset:
    """Ten items; `make` turns an index into the image. Every read is counted."""

    def __init__(self, size: int = N, make: Callable[[int], Any] | None = None) -> None:
        self.size = size
        self.make = make or (lambda i: np.random.default_rng(i).random((1, 4, 4), dtype=np.float32))
        self.reads: Counter[int] = Counter()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[Any, int, dict[str, Any]]:
        self.reads[index] += 1
        return self.make(index), 0, {}

    @property
    def total(self) -> int:
        return sum(self.reads.values())


def _torch(width: int, **kwargs: Any) -> TorchExtractor:
    torch.manual_seed(width)
    return TorchExtractor(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, width)), device="cpu", **kwargs)


def _ways() -> dict[str, Any]:
    return {"a": _torch(3), "b": _torch(5), "flat": FlattenExtractor()}


def _solo(extractor: Any, dataset: Any = None, batch_size: int = 4) -> np.ndarray:
    dataset = _Dataset() if dataset is None else dataset
    return np.asarray(Embeddings(dataset, extractor=extractor, batch_size=batch_size)[:])


def _notes(error: BaseException) -> str:
    return " ".join(getattr(error, "__notes__", []))


def test_each_child_equals_a_solo_run():
    ways = _ways()
    children = Embeddings.compute_multi(_Dataset(), ways, batch_size=4)
    for key, way in ways.items():
        assert np.array_equal(children[key][:], _solo(way))


def test_children_are_plain_embeddings_in_the_given_order():
    ways = _ways()
    children = Embeddings.compute_multi(_Dataset(), ways, batch_size=4)
    assert list(children) == list(ways)
    for key, child in children.items():
        assert type(child) is Embeddings
        assert child._extractor is ways[key]
        assert child.batch_size == 4


def test_each_item_is_read_once_plus_one_probe_per_child():
    dataset = _Dataset()
    Embeddings.compute_multi(dataset, _ways(), batch_size=4)
    assert dataset.total == N + 3
    assert all(dataset.reads[i] == 1 for i in range(1, N))


def test_an_in_place_transform_in_one_way_does_not_reach_the_next():
    ways = {"zeroes": _torch(3, transforms=[lambda t: t.mul_(0)]), "flat": FlattenExtractor()}
    children = Embeddings.compute_multi(_Dataset(), ways, batch_size=4)
    assert np.array_equal(children["flat"][:], _solo(FlattenExtractor()))


def test_tensor_items_are_cloned_for_each_way():
    def tensors(i: int) -> torch.Tensor:
        return torch.as_tensor(np.random.default_rng(i).random((1, 4, 4), dtype=np.float32))

    def zeroes(batch: list[torch.Tensor]) -> np.ndarray:
        return torch.stack([t.mul_(0) for t in batch]).flatten(1).numpy()

    children = Embeddings.compute_multi(
        _Dataset(make=tensors), {"zeroes": zeroes, "flat": FlattenExtractor()}, batch_size=4
    )
    assert np.array_equal(children["flat"][:], _solo(FlattenExtractor(), _Dataset(make=tensors)))


def test_an_extractor_that_writes_into_its_numpy_input_equals_its_solo_run():
    def inplace(batch: list[np.ndarray]) -> np.ndarray:
        return np.stack([np.multiply(x, 2, out=x) for x in batch]).reshape(len(batch), -1)

    children = Embeddings.compute_multi(_Dataset(), {"inplace": inplace, "flat": FlattenExtractor()}, batch_size=4)
    assert np.array_equal(children["inplace"][:], _solo(inplace))
    assert np.array_equal(children["flat"][:], _solo(FlattenExtractor()))


def test_a_write_through_torch_does_not_reach_the_next_way():
    # torch.from_numpy ignores NumPy's write flag, so only a copy keeps such a write to the way that made it.
    def through_torch(batch: list[np.ndarray]) -> np.ndarray:
        return np.stack([torch.from_numpy(x).mul_(0).numpy().ravel() for x in batch])

    children = Embeddings.compute_multi(_Dataset(), {"zeroes": through_torch, "flat": FlattenExtractor()}, batch_size=4)
    assert np.array_equal(children["flat"][:], _solo(FlattenExtractor()))


def test_an_in_place_transform_on_a_grayscale_video_does_not_reach_the_next_way():
    from dataeval.extractors import VideoTorchExtractor

    class _Flatten(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flatten(1)

    def gray(i: int) -> np.ndarray:  # (T, H, W): sliced into clips rather than walked, with no copy on the way
        return np.random.default_rng(i).random((4, 3, 3), dtype=np.float32)

    zeroes = VideoTorchExtractor(_Flatten(), transforms=[lambda t: t.mul_(0)], num_frames=2, device="cpu")
    keep = VideoTorchExtractor(_Flatten(), num_frames=2, device="cpu")
    children = Embeddings.compute_multi(_Dataset(make=gray), {"zeroes": zeroes, "keep": keep}, batch_size=2)
    assert np.array_equal(children["keep"][:], _solo(keep, _Dataset(make=gray), batch_size=2))


def test_a_way_that_raises_is_named():
    calls = Counter()

    def fails_on_its_second_batch(batch: list[np.ndarray]) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("out of memory")
        return np.stack(batch).reshape(len(batch), -1)

    with pytest.raises(RuntimeError, match="out of memory") as raised:
        Embeddings.compute_multi(
            _Dataset(), {"flat": FlattenExtractor(), "flaky": fails_on_its_second_batch}, batch_size=4
        )
    assert "'flaky'" in _notes(raised.value)


@pytest.mark.parametrize(
    ("output", "error"),
    [(lambda batch: {"x": np.zeros((len(batch), 2))}, TypeError), (lambda batch: np.zeros((3, 2)), ShapeMismatchError)],
)
def test_a_refused_output_names_its_way(output, error):
    with pytest.raises(error) as raised:
        Embeddings.compute_multi(_Dataset(), {"flat": FlattenExtractor(), "bad": output}, batch_size=4)
    assert "'bad'" in _notes(raised.value)


def test_batch_sizes_that_differ_with_none_passed_raise():
    ways = {"four": _torch(3, batch_size=4), "eight": _torch(5, batch_size=8)}
    with pytest.raises(ValueError, match="batch_size"):
        Embeddings.compute_multi(_Dataset(), ways)
    children = Embeddings.compute_multi(_Dataset(), ways, batch_size=2)
    assert [child.batch_size for child in children.values()] == [2, 2]


def test_a_way_with_its_own_batch_size_keeps_it_when_none_is_passed():
    children = Embeddings.compute_multi(_Dataset(), {"four": _torch(3, batch_size=4)})
    assert children["four"].batch_size == 4


def test_load_threads_give_identical_children():
    serial = Embeddings.compute_multi(_Dataset(), _ways(), batch_size=4)
    with use_load_threads(4):
        threaded = Embeddings.compute_multi(_Dataset(), _ways(), batch_size=4)
    for key in serial:
        assert np.array_equal(serial[key][:], threaded[key][:])


def test_progress_fires_once_per_batch():
    calls: list[tuple[int, int | None]] = []

    def record(
        step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict[str, Any] | None = None
    ) -> None:
        calls.append((step, total))

    Embeddings.compute_multi(_Dataset(), _ways(), batch_size=4, progress_callback=record)
    assert calls == [(4, N), (8, N), (10, N)]


@pytest.mark.parametrize(
    ("ways", "error", "match"),
    [
        ({}, ValueError, "at least one extractor"),
        ({"emb": Embeddings(extractor=FlattenExtractor())}, TypeError, "its extractor"),
        ({"thing": object()}, TypeError, "not callable"),
    ],
)
def test_refusals_read_nothing(ways, error, match):
    dataset = _Dataset()
    with pytest.raises(error, match=match):
        Embeddings.compute_multi(dataset, ways, batch_size=4)
    assert dataset.total == 0


def test_no_dataset_is_refused():
    with pytest.raises(ValueError, match="dataset"):
        Embeddings.compute_multi(None, _ways(), batch_size=4)  # type: ignore[arg-type]


def _frames(i: int) -> list[np.ndarray]:
    return [np.full((1, 2, 2), i + k, dtype=np.float32) for k in range(3)]


def _walk(batch: list[Any]) -> np.ndarray:
    return np.stack([np.stack(list(item)).ravel() for item in batch])


def test_a_one_shot_stream_with_several_ways_raises():
    with pytest.raises(ValueError, match="one-shot"):
        Embeddings.compute_multi(_Dataset(make=lambda i: iter(_frames(i))), {"a": _walk, "b": _walk}, batch_size=4)
    alone = Embeddings.compute_multi(_Dataset(make=lambda i: iter(_frames(i))), {"a": _walk}, batch_size=4)
    assert alone["a"][:].shape == (N, 12)


class _Replayable:
    def __init__(self, i: int) -> None:
        self.i = i

    def __iter__(self):
        return iter(_frames(self.i))


def test_a_reiterable_item_reaches_every_way():
    children = Embeddings.compute_multi(_Dataset(make=_Replayable), {"a": _walk, "b": _walk}, batch_size=4)
    expected = _solo(_walk, _Dataset(make=_Replayable))
    assert np.array_equal(children["a"][:], expected)
    assert np.array_equal(children["b"][:], expected)


def test_an_empty_dataset_gives_empty_children_equal_to_solo():
    children = Embeddings.compute_multi(_Dataset(size=0), _ways(), batch_size=4)
    for key, way in _ways().items():
        assert np.array_equal(children[key][:], _solo(way, _Dataset(size=0)))


def test_it_is_experimental():
    # The decorator warns once per process from a flag no test can reset, so the mark is checked, as elsewhere.
    assert "Embeddings.compute_multi" in getattr(Embeddings.compute_multi, "__experimental__", "")


def _other(i: int) -> np.ndarray:
    return np.random.default_rng(100 + i).random((1, 4, 4), dtype=np.float32)


def test_each_child_gets_its_own_directory(tmp_path):
    children = Embeddings.compute_multi(_Dataset(), _ways(), batch_size=4, path=tmp_path)
    for key, child in children.items():
        assert child._path is not None
        assert child._path.parent == tmp_path / key
    assert len({child._path for child in children.values()}) == len(children)


def test_a_second_dataset_over_one_path_leaves_the_first_intact(tmp_path):
    ways = {"a": _torch(3), "flat": FlattenExtractor()}
    train = Embeddings.compute_multi(_Dataset(), ways, batch_size=4, path=tmp_path, memory_threshold=0.0)
    before = {key: np.array(child[:]) for key, child in train.items()}
    assert all(isinstance(child._embeddings, np.memmap) for child in train.values())
    test = Embeddings.compute_multi(_Dataset(make=_other), ways, batch_size=4, path=tmp_path, memory_threshold=0.0)
    for key in ways:
        assert train[key]._path != test[key]._path
        assert np.array_equal(train[key][:], before[key])


def test_two_models_with_equal_repr_get_different_files(tmp_path):
    ways = {"small": _torch(3), "large": _torch(5)}
    assert repr(ways["small"]) == repr(ways["large"])
    children = Embeddings.compute_multi(_Dataset(), ways, batch_size=4, path=tmp_path)
    assert children["small"]._path != children["large"]._path


def test_keys_that_name_one_directory_raise(tmp_path):
    dataset = _Dataset()
    with pytest.raises(ValueError, match="same directory"):
        Embeddings.compute_multi(
            dataset, {"a/b": FlattenExtractor(), "a_b": FlattenExtractor()}, batch_size=4, path=tmp_path
        )
    assert dataset.total == 0


@pytest.mark.parametrize("key", ["", ".", ".."])
def test_keys_that_name_no_directory_raise(tmp_path, key):
    dataset = _Dataset()
    with pytest.raises(ValueError, match="names no directory"):
        Embeddings.compute_multi(dataset, {key: FlattenExtractor()}, batch_size=4, path=tmp_path)
    assert dataset.total == 0


def test_a_path_that_is_a_file_raises(tmp_path):
    taken = tmp_path / "taken.npy"
    taken.write_bytes(b"")
    with pytest.raises(ValueError, match="not a directory"):
        Embeddings.compute_multi(_Dataset(), _ways(), batch_size=4, path=taken)


def test_memmap_is_chosen_on_the_sum(tmp_path, monkeypatch):
    # Ten items of 3, 5 and 16 float32 values take 120, 200 and 640 bytes, 960 together. With 1000 bytes available,
    # 80% is 800: each alone fits in memory, and all three together do not.
    import dataeval._embeddings as module

    monkeypatch.setattr(module.psutil, "virtual_memory", lambda: SimpleNamespace(available=1000))
    children = Embeddings.compute_multi(_Dataset(), _ways(), batch_size=4, path=tmp_path / "multi")
    assert all(isinstance(child._embeddings, np.memmap) for child in children.values())
    solo = Embeddings(_Dataset(), extractor=FlattenExtractor(), batch_size=4, path=tmp_path / "solo").compute()
    assert not isinstance(solo._embeddings, np.memmap)


def test_without_a_path_nothing_is_a_memmap():
    children = Embeddings.compute_multi(_Dataset(), _ways(), batch_size=4, memory_threshold=0.0)
    assert not any(isinstance(child._embeddings, np.memmap) for child in children.values())


def test_with_plenty_of_memory_a_path_holds_no_file_until_save(tmp_path):
    children = Embeddings.compute_multi(_Dataset(), {"flat": FlattenExtractor()}, batch_size=4, path=tmp_path)
    child = children["flat"]
    assert not isinstance(child._embeddings, np.memmap)
    assert list((tmp_path / "flat").iterdir()) == []
    child.save()
    assert child._path is not None
    assert child._path.exists()
