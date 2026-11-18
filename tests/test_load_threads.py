"""`load_threads` loads several dataset items at once in the whole-dataset passes, `Embeddings`, `Metadata`, `Filter`
and `Relabel`, and leaves every result and the order of reads exactly as they were when it is not set."""

import pickle
import threading
import time
import warnings
from typing import Any, cast

import numpy as np
import pytest
from polars.testing import assert_frame_equal

from dataeval import Embeddings, Metadata
from dataeval import _experimental as experimental_module
from dataeval.bias import Balance
from dataeval.config import get_load_threads, set_load_threads, use_load_threads
from dataeval.data import DetectionCrops, Filter, Relabel, SequenceFrames, View, merge_datasets
from dataeval.exceptions import ExperimentalWarning
from dataeval.extractors import FlattenExtractor
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.quality import Outliers
from tests.data import test_frames as frame_fixtures
from tests.data.conftest import OD_FACTORS, OD_LABELS, VIDEO_FACTORS, VIDEOS, FactorODDataset, FactorVideos

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

N = 32
DELAY = 0.02


class _SlowImages:
    """A classification dataset whose reads take `delay` seconds and record which thread made them."""

    metadata = DatasetMetadata(id="slow", index2label={0: "a", 1: "b", 2: "c"})

    def __init__(self, size: int = N, delay: float = DELAY) -> None:
        self.size, self.delay = size, delay
        self.order: list[int] = []
        self.idents: dict[int, int] = {}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        self.order.append(index)
        self.idents[index] = threading.get_ident()
        time.sleep(self.delay)
        image = np.random.default_rng(index).random((3, 8, 8), dtype=np.float32)
        metadata = cast(DatumMetadata, {"id": index, "group": ("x", "y")[index % 2]})
        return image, np.eye(3, dtype=np.float32)[index % 3], metadata

    @property
    def off_main_thread(self) -> int:
        return sum(1 for ident in self.idents.values() if ident != threading.get_ident())


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(experimental_module, "_WARNED", set())
    set_load_threads(None)
    yield
    set_load_threads(None)


def _same(left: Any, right: Any) -> None:
    """Assert two evaluator outputs, dictionaries of frames and other values, are equal."""
    assert left.keys() == right.keys()
    for key in left:
        if hasattr(left[key], "equals"):
            assert_frame_equal(left[key], right[key])
        else:
            assert repr(left[key]) == repr(right[key])


def _embed(dataset: Any) -> Any:
    return np.asarray(Embeddings(dataset, extractor=FlattenExtractor(), batch_size=8))


class TestTheSetting:
    def test_it_is_unset_by_default(self):
        assert get_load_threads() is None

    def test_set_get_and_use_restore(self):
        set_load_threads(4)
        assert get_load_threads() == 4
        with use_load_threads(2):
            assert get_load_threads() == 2
        assert get_load_threads() == 4

    @pytest.mark.parametrize("value", [0, -1])
    def test_a_value_below_one_is_refused(self, value):
        with pytest.raises(ValueError, match="load_threads"):
            set_load_threads(value)

    def test_asking_for_threads_warns_once_and_not_asking_raises_nothing(self):
        with pytest.warns(ExperimentalWarning, match="load_threads"):
            set_load_threads(4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            set_load_threads(4)
            with use_load_threads(2):
                pass
            set_load_threads(None)
            set_load_threads(1)


class TestDefaultIsUnchanged:
    def test_embeddings_read_in_order_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0)
        _embed(dataset)
        assert dataset.order[1:] == list(range(N))  # the first read is the constructor's probe of item 0
        assert dataset.off_main_thread == 0

    def test_the_metadata_walk_reads_in_order_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0)
        _ = Metadata(dataset).factor_names
        assert dataset.off_main_thread == 0
        assert dataset.order[-(N - 1) :] == list(range(1, N))


class _RecordingList(list):
    """Images already in memory, in a list that records which thread read each one."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.idents: list[int] = []

    def __getitem__(self, index: Any) -> Any:
        self.idents.append(threading.get_ident())
        return super().__getitem__(index)


class TestEmbeddings:
    def test_threads_give_the_same_embeddings_faster(self):
        serial_dataset = _SlowImages()
        start = time.perf_counter()
        serial = _embed(serial_dataset)
        serial_time = time.perf_counter() - start

        threaded_dataset = _SlowImages()
        with use_load_threads(4):
            start = time.perf_counter()
            threaded = _embed(threaded_dataset)
            threaded_time = time.perf_counter() - start

        np.testing.assert_array_equal(threaded, serial)
        assert threaded_dataset.off_main_thread > 0
        assert threaded_time < 0.6 * serial_time

    def test_a_single_index_builds_no_pool(self):
        dataset = _SlowImages(delay=0)
        with use_load_threads(4):
            Embeddings(dataset, extractor=FlattenExtractor(), batch_size=8)[5]
        assert dataset.off_main_thread == 0

    def test_a_slow_extractor_still_sees_batches_in_order(self):
        dataset = _SlowImages(delay=0.002)
        with use_load_threads(4):
            threaded = _embed(dataset)
        np.testing.assert_array_equal(threaded, _embed(_SlowImages(delay=0)))

    def test_only_the_first_item_of_a_pass_is_read_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0.002)
        with use_load_threads(4):
            _embed(dataset)
        assert [index for index, ident in dataset.idents.items() if ident == threading.get_ident()] == [0]

    def test_reads_run_at_most_four_times_the_threads_ahead_of_the_extractor(self):
        dataset = _SlowImages(delay=0)
        ahead: list[int] = []

        def extractor(batch: list[Any]) -> Any:
            time.sleep(0.01)  # time for the reads to run as far ahead as they may
            ahead.append(len(dataset.order) - 1 - (len(ahead) + 1) * 8)  # less the constructor's probe of item 0
            return np.zeros((len(batch), 1))

        with use_load_threads(2):
            Embeddings(dataset, extractor=extractor, batch_size=8)[:]
        assert 0 < max(ahead) <= 4 * 2

    @pytest.mark.parametrize("multi", [False, True], ids=["embeddings", "compute_multi"])
    def test_an_extractor_that_raises_leaves_no_read_thread_running(self, multi):
        def extractor(batch: list[Any]) -> Any:
            if calls:
                raise RuntimeError("extractor failed")
            calls.append(len(batch))
            return np.zeros((len(batch), 1))

        def run() -> None:
            if multi:
                Embeddings.compute_multi(_SlowImages(delay=0.002), {"one": extractor}, batch_size=8)
            else:
                Embeddings(_SlowImages(delay=0.002), extractor=extractor, batch_size=8)[:]

        calls: list[int] = []
        before = set(threading.enumerate())
        with use_load_threads(4), pytest.raises(RuntimeError, match="extractor failed") as raised:
            run()
        # The traceback is still held, as a notebook holds the last one, and with it every frame the error passed.
        assert raised.traceback
        assert set(threading.enumerate()) <= before

    def test_a_list_is_read_on_the_calling_thread(self):
        images = _RecordingList(np.full((3, 4, 4), index, dtype=np.float32) for index in range(12))
        with use_load_threads(4):
            embedded = np.asarray(Embeddings(images, extractor=FlattenExtractor(), batch_size=4))
        np.testing.assert_array_equal(embedded[:, 0], np.arange(12))
        assert set(images.idents) == {threading.get_ident()}

    def test_a_tracking_dataset_is_read_on_the_calling_thread(self):
        videos = _RecordingVideos(VIDEOS, VIDEO_FACTORS)
        with use_load_threads(4):
            Embeddings(videos, extractor=lambda batch: np.zeros((len(batch), 1)), batch_size=8)[:]
        assert set(videos.idents) == {threading.get_ident()}


class TestMetadataWalk:
    def test_threads_give_the_same_metadata_faster(self):
        start = time.perf_counter()
        serial = Metadata(_SlowImages()).dataframe
        serial_time = time.perf_counter() - start

        dataset = _SlowImages()
        with use_load_threads(4):
            start = time.perf_counter()
            threaded = Metadata(dataset).dataframe
            threaded_time = time.perf_counter() - start

        assert_frame_equal(threaded, serial)
        assert dataset.off_main_thread > 0
        assert threaded_time < 0.7 * serial_time

    def test_item_zero_is_read_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0.002)
        with use_load_threads(4):
            _ = Metadata(dataset).factor_names
        assert dataset.idents[0] == threading.get_ident()

    def test_a_tracking_dataset_is_walked_serially_and_gives_the_same_metadata(self):
        serial = Metadata(FactorVideos(VIDEOS, VIDEO_FACTORS)).dataframe
        with use_load_threads(4):
            threaded = Metadata(FactorVideos(VIDEOS, VIDEO_FACTORS)).dataframe
        assert_frame_equal(threaded, serial)

    def test_a_view_record_is_read_serially(self):
        dataset = _SlowImages(delay=0)
        # The predicate structures every item, so the view keeps its record.
        view = View(dataset, Filter("\"group\" = 'x'"))
        assert view._reads is not None
        before = len(dataset.order)
        with use_load_threads(4):
            frame = Metadata(view).dataframe
        assert dataset.off_main_thread == 0
        assert len(dataset.order) - before <= 1  # the probe of item 0, and nothing is read again
        assert_frame_equal(frame, Metadata(View(_SlowImages(delay=0), Filter("\"group\" = 'x'"))).dataframe)


class _RecordingVideos(FactorVideos):
    """Multi-object tracking videos that record which thread read each one."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.idents: list[int] = []

    def __getitem__(self, index: int) -> Any:
        self.idents.append(threading.get_ident())
        return super().__getitem__(index)


class TestFilter:
    """`Filter(where=...)` reads every item once to structure the view, and loads those reads in threads too."""

    WHERE = "\"group\" = 'x'"

    def test_the_default_reads_in_order_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0)
        View(dataset, Filter(self.WHERE))
        assert dataset.off_main_thread == 0
        assert dataset.order[-N:] == list(range(N))

    def test_threads_give_the_same_view_and_metadata_faster(self):
        start = time.perf_counter()
        serial = View(_SlowImages(), Filter(self.WHERE))
        serial_time = time.perf_counter() - start

        dataset = _SlowImages()
        with use_load_threads(4):
            start = time.perf_counter()
            threaded = View(dataset, Filter(self.WHERE))
            threaded_time = time.perf_counter() - start

        assert threaded.selection == serial.selection
        assert_frame_equal(Metadata(threaded).dataframe, Metadata(serial).dataframe)
        assert dataset.off_main_thread > 0
        assert threaded_time < 0.6 * serial_time

    def test_item_zero_is_read_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0.002)
        with use_load_threads(4):
            View(dataset, Filter(self.WHERE))
        assert dataset.idents[0] == threading.get_ident()

    def test_the_view_keeps_the_same_record_and_metadata_then_reads_nothing_again(self):
        dataset = _SlowImages(delay=0.002)
        with use_load_threads(4):
            view = View(dataset, Filter(self.WHERE))
        assert dataset.off_main_thread > 0
        assert view._reads is not None
        assert set(view._reads) == set(range(0, N, 2))
        before = len(dataset.order)
        frame = Metadata(view).dataframe
        assert len(dataset.order) - before <= 1  # the probe of item 0, and nothing is read again
        assert_frame_equal(frame, Metadata(View(_SlowImages(delay=0), Filter(self.WHERE))).dataframe)

    def test_an_item_that_cannot_be_read_raises_the_same_error(self):
        class Broken(_SlowImages):
            def __getitem__(self, index: int) -> Any:
                if index == 9:
                    raise RuntimeError("item 9 is unreadable")
                return super().__getitem__(index)

        with pytest.raises(RuntimeError, match="item 9 is unreadable"):
            View(Broken(delay=0), Filter(self.WHERE))
        with use_load_threads(4), pytest.raises(RuntimeError, match="item 9 is unreadable"):
            View(Broken(delay=0), Filter(self.WHERE))

    def test_a_tracking_dataset_is_filtered_serially_and_gives_the_same_view(self):
        serial = View(FactorVideos(VIDEOS, VIDEO_FACTORS), Filter("weather = 'rain'"))
        dataset = _RecordingVideos(VIDEOS, VIDEO_FACTORS)
        with use_load_threads(4):
            threaded = View(dataset, Filter("weather = 'rain'"))
        assert threaded.selection == serial.selection == [0, 2]
        assert set(dataset.idents) == {threading.get_ident()}

    def test_a_transform_the_view_applies_runs_in_the_threads(self):
        seen: set[int] = set()

        def record(datum: Any) -> Any:
            seen.add(threading.get_ident())
            return datum

        view = View(_SlowImages(delay=0.002))
        view.map(record)
        with use_load_threads(4):
            Filter(self.WHERE).apply(view)
        assert seen - {threading.get_ident()}


class _RecordingOD(FactorODDataset):
    """Object detection images that record which thread read each one."""

    def __init__(self) -> None:
        super().__init__(OD_LABELS, OD_FACTORS)
        self.idents: list[int] = []

    def __getitem__(self, index: int) -> Any:
        self.idents.append(threading.get_ident())
        return super().__getitem__(index)


class _Broken(_SlowImages):
    """A classification dataset whose item 9 cannot be read."""

    def __getitem__(self, index: int) -> Any:
        if index == 9:
            raise RuntimeError("item 9 is unreadable")
        return super().__getitem__(index)


class TestRelabel:
    """`Relabel` reads every datum once to decide what to drop, and loads those reads in threads too."""

    MAPPING = {"a": "x", "b": "x"}  # class "c" has no target class, so its items are dropped

    def test_the_default_reads_in_order_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0)
        View(dataset, Relabel(self.MAPPING))
        assert dataset.off_main_thread == 0
        assert dataset.order[-N:] == list(range(N))

    def test_threads_give_the_same_view_and_the_same_drops_faster(self):
        serial_op, threaded_op = Relabel(self.MAPPING), Relabel(self.MAPPING)
        start = time.perf_counter()
        serial = View(_SlowImages(), serial_op)
        serial_time = time.perf_counter() - start

        dataset = _SlowImages()
        with use_load_threads(4):
            start = time.perf_counter()
            threaded = View(dataset, threaded_op)
            threaded_time = time.perf_counter() - start

        assert threaded.selection == serial.selection
        assert threaded_op.dropped_indices == serial_op.dropped_indices
        assert serial_op.dropped_indices  # the class with no target class was dropped
        assert dataset.off_main_thread > 0
        assert threaded_time < 0.6 * serial_time

    def test_item_zero_is_read_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0.002)
        with use_load_threads(4):
            View(dataset, Relabel(self.MAPPING))
        assert dataset.idents[0] == threading.get_ident()

    def test_an_item_that_cannot_be_read_raises_the_same_error(self):
        with pytest.raises(RuntimeError, match="item 9 is unreadable"):
            View(_Broken(delay=0), Relabel(self.MAPPING))
        with use_load_threads(4), pytest.raises(RuntimeError, match="item 9 is unreadable"):
            View(_Broken(delay=0), Relabel(self.MAPPING))


class TestFilterByClasses:
    """`Filter(classes=...)` without `where` reads each target, and loads those reads in threads too."""

    def test_the_default_reads_in_order_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0)
        View(dataset, Filter(classes=[0]))
        assert dataset.off_main_thread == 0
        assert dataset.order[-N:] == list(range(N))

    def test_threads_give_the_same_view_faster(self):
        start = time.perf_counter()
        serial = View(_SlowImages(), Filter(classes=[0]))
        serial_time = time.perf_counter() - start

        dataset = _SlowImages()
        with use_load_threads(4):
            start = time.perf_counter()
            threaded = View(dataset, Filter(classes=[0]))
            threaded_time = time.perf_counter() - start

        assert threaded.selection == serial.selection == list(range(0, N, 3))
        assert dataset.off_main_thread > 0
        assert threaded_time < 0.6 * serial_time

    def test_it_still_starts_no_read_record(self):
        with use_load_threads(4):
            view = View(_SlowImages(delay=0.002), Filter(classes=[0]))
        assert view._reads is None

    def test_item_zero_is_read_on_the_calling_thread(self):
        dataset = _SlowImages(delay=0.002)
        with use_load_threads(4):
            View(dataset, Filter(classes=[0]))
        assert dataset.idents[0] == threading.get_ident()

    def test_an_item_that_cannot_be_read_raises_the_same_error(self):
        with pytest.raises(RuntimeError, match="item 9 is unreadable"):
            View(_Broken(delay=0), Filter(classes=[0]))
        with use_load_threads(4), pytest.raises(RuntimeError, match="item 9 is unreadable"):
            View(_Broken(delay=0), Filter(classes=[0]))

    def test_detections_are_removed_the_same_way_when_loaded_in_threads(self):
        serial = View(FactorODDataset(OD_LABELS, OD_FACTORS), Filter(classes=[0, 1]))
        dataset = _RecordingOD()
        with use_load_threads(4):
            threaded = View(dataset, Filter(classes=[0, 1]))
        assert threaded.selection == serial.selection
        assert_frame_equal(Metadata(threaded).dataframe, Metadata(serial).dataframe)
        assert set(dataset.idents) - {threading.get_ident()}

    def test_a_tracking_dataset_is_filtered_serially_and_gives_the_same_view(self):
        serial = View(FactorVideos(VIDEOS, VIDEO_FACTORS), Filter(classes=[0]))
        dataset = _RecordingVideos(VIDEOS, VIDEO_FACTORS)
        with use_load_threads(4):
            threaded = View(dataset, Filter(classes=[0]))
        assert threaded.selection == serial.selection
        assert set(dataset.idents) == {threading.get_ident()}


class TestWithTheReadRecord:
    def test_threaded_embeddings_fill_a_views_record_and_metadata_then_reads_nothing_again(self):
        dataset = _SlowImages(delay=0.002)
        view = View(dataset, record_reads=True)
        with use_load_threads(4):
            _embed(view)
        assert dataset.off_main_thread > 0
        assert view._reads is not None
        assert set(view._reads) == set(range(N))
        before = len(dataset.order)
        frame = Metadata(view).dataframe
        assert len(dataset.order) - before <= 1  # the probe of item 0, and nothing is read again
        assert_frame_equal(frame, Metadata(_SlowImages(delay=0)).dataframe)


class _Opaque:
    """A wrapper that is not DataEval's and names no parent, so nothing can see what it wraps."""

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset
        self.metadata = dataset.metadata

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        return self._dataset[index]


class TestSequenceFrames:
    """One video's frames come off one decoder in order, so a `SequenceFrames` is read serially however it is wrapped,
    and a wrapper that hides it still reads the right frames."""

    @pytest.fixture(autouse=True)
    def slow_frames(self, monkeypatch):
        """Each frame takes 2 ms to come off the stream and frees the GIL meanwhile, as a decoder does."""
        walk = frame_fixtures._CountingStream.__iter__

        def slow(stream: Any) -> Any:
            for frame in walk(stream):
                time.sleep(0.002)
                yield frame

        monkeypatch.setattr(frame_fixtures._CountingStream, "__iter__", slow)

    @staticmethod
    def _frames() -> tuple[SequenceFrames, list[dict[str, int]]]:
        dataset, counters = frame_fixtures.make_dataset(frame_counts=(12, 12, 12), shape=(1, 12, 14))
        return SequenceFrames(dataset), counters

    @pytest.mark.parametrize(
        "wrap",
        [lambda frames: frames, View, DetectionCrops, lambda frames: merge_datasets(frames, View(frames))],
        ids=["frames", "view", "crops", "merge"],
    )
    def test_embeddings_and_metadata_give_the_serial_result_and_decode_as_many_frames(self, wrap):
        results = []
        for threads in (1, 8):
            frames, counters = self._frames()
            with use_load_threads(threads):
                embedded = _embed(wrap(frames))
                frame = Metadata(wrap(frames)).dataframe
            results.append((embedded, frame, sum(counter["frames"] for counter in counters)))
        (serial, serial_frame, serial_decoded), (threaded, threaded_frame, threaded_decoded) = results
        np.testing.assert_array_equal(threaded, serial)
        assert_frame_equal(threaded_frame, serial_frame)
        assert threaded_decoded == serial_decoded

    def test_behind_a_wrapper_that_hides_it_the_frames_are_still_right(self):
        serial = _embed(self._frames()[0])
        with use_load_threads(8):
            threaded = _embed(_Opaque(self._frames()[0]))
        np.testing.assert_array_equal(threaded, serial)

    def test_a_resting_frame_view_and_a_view_over_it_pickle_and_read(self):
        expected = self._frames()[0][1][0]
        frames = self._frames()[0]
        for dataset in (frames, View(frames)):
            np.testing.assert_array_equal(pickle.loads(pickle.dumps(dataset))[1][0], expected)


class TestEvaluatorsThatBuildTheirOwn:
    def test_outliers_loads_its_embeddings_in_threads(self):
        dataset = _SlowImages(delay=0.005)
        with use_load_threads(4):
            threaded = Outliers(extractor=FlattenExtractor()).evaluate(dataset)
        assert dataset.off_main_thread > 0
        serial = Outliers(extractor=FlattenExtractor()).evaluate(_SlowImages(delay=0))
        assert_frame_equal(threaded.data(), serial.data())

    def test_balance_loads_its_metadata_in_threads(self):
        dataset = _SlowImages(delay=0.005)
        with use_load_threads(4):
            threaded = Balance().evaluate(dataset)
        assert dataset.off_main_thread > 0
        serial = Balance().evaluate(_SlowImages(delay=0))
        _same(threaded.data(), serial.data())
