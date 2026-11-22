"""`View(record_reads=True)` records each datum's pixel-free structure as the view reads it, so `Metadata` built after a
pass that reads the pixels does not read the dataset again."""

import inspect
import pickle
import warnings
from collections import Counter
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dataeval import Embeddings, Metadata
from dataeval import _experimental as experimental_module
from dataeval.bias import Balance
from dataeval.data import Filter, Limit, Operation, View
from dataeval.exceptions import ExperimentalWarning, MaiteShapeError
from dataeval.extractors import FlattenExtractor
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.quality import Outliers

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

N = 40


class _Images:
    """A classification dataset of random images that counts how often each item is read."""

    metadata = DatasetMetadata(id="images", index2label={0: "a", 1: "b", 2: "c"})

    def __init__(self, size: int = N) -> None:
        self.size = size
        self.reads: Counter[int] = Counter()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        self.reads[index] += 1
        image = np.random.default_rng(index).random((3, 8, 8), dtype=np.float32)
        metadata = cast(DatumMetadata, {"id": index, "group": ("x", "y")[index % 2]})
        return image, np.eye(3, dtype=np.float32)[index % 3], metadata

    @property
    def total(self) -> int:
        return sum(self.reads.values())


def _read_all(view: Any) -> None:
    for index in range(len(view)):
        view[index]


def _frame(dataset: Any) -> Any:
    return Metadata(dataset).dataframe


@pytest.fixture(autouse=True)
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(experimental_module, "_WARNED", set())


@pytest.mark.required
class TestRecordReads:
    def test_it_is_off_by_default(self):
        view = View(_Images())
        _read_all(view)
        assert view._reads is None

    def test_reading_an_item_records_its_skeleton(self):
        view = View(_Images(), record_reads=True)
        view[0]
        view[2]
        assert view._reads is not None
        assert set(view._reads) == {0, 2}
        assert all(skeleton[0] is None for skeleton in view._reads.values())

    def test_metadata_after_a_full_pass_does_not_read_the_dataset_again(self):
        dataset = _Images()
        view = View(dataset, record_reads=True)
        _read_all(view)
        before = dataset.total
        _ = Metadata(view).factor_names
        assert dataset.total - before <= 1  # the probe of item 0, which every Metadata makes

    def test_metadata_from_the_record_equals_a_direct_walk(self):
        view = View(_Images(), record_reads=True)
        _read_all(view)
        assert_frame_equal(_frame(view), _frame(View(_Images())))

    def test_a_partly_filled_record_is_completed_by_reading_only_what_is_missing(self):
        dataset = _Images()
        view = View(dataset, record_reads=True)
        for index in range(N // 2):
            view[index]
        before = Counter(dataset.reads)
        with pytest.warns(UserWarning, match=f"held {N // 2} of the {N} dataset items"):
            frame = _frame(view)
        assert all(dataset.reads[i] - before[i] <= (1 if i == 0 else 0) for i in range(N // 2))
        assert all(dataset.reads[i] - before[i] == 1 for i in range(N // 2, N))
        assert_frame_equal(frame, _frame(View(_Images())))

    def test_a_second_pass_does_not_compute_the_skeletons_again(self, monkeypatch):
        import dataeval.data._view as view_module

        calls = []
        original = view_module.skeleton
        monkeypatch.setattr(view_module, "skeleton", lambda datum: calls.append(1) or original(datum))
        view = View(_Images(), record_reads=True)
        _read_all(view)
        _read_all(view)
        assert len(calls) == N

    def test_map_clears_the_record_and_recording_resumes(self):
        view = View(_Images(), record_reads=True)
        _read_all(view)
        view.map(lambda datum: datum)
        assert view._reads is None
        view[3]
        assert view._reads is not None
        assert set(view._reads) == {3}

    def test_outliers_then_metadata_reads_each_item_about_once(self):
        dataset = _Images()
        view = View(dataset, record_reads=True)
        Outliers().evaluate(view)
        _ = Metadata(view).factor_names
        assert dataset.total < 1.5 * N  # without the record Outliers alone reads each item twice

    def test_a_view_over_a_recording_view_gives_the_same_metadata(self):
        inner = View(_Images(), record_reads=True)
        outer = View(inner)
        _read_all(outer)
        assert_frame_equal(_frame(outer), _frame(View(_Images())))

    def test_a_class_filter_after_recording_leaves_the_record_inside_the_selection(self):
        view = View(_Images(), record_reads=True)
        _read_all(view)
        Filter(classes=[0, 1]).apply(view)
        assert view._reads is not None
        assert set(view._reads) <= set(view.selection)

    def test_pickling_drops_the_record_and_the_flag(self):
        view = View(_Images(), record_reads=True)
        _read_all(view)
        copy = pickle.loads(pickle.dumps(view))
        assert copy._reads is None
        copy[0]
        assert copy._reads is None


@pytest.mark.required
class TestDatumsItDoesNotRecord:
    def test_a_two_tuple_dataset_is_refused_as_a_plain_view_refuses_it(self):
        class Pairs:
            def __len__(self) -> int:
                return 3

            def __getitem__(self, index: int) -> tuple[Any, int]:
                return np.zeros((3, 4, 4), dtype=np.float32), 1

        with pytest.raises(MaiteShapeError):
            Embeddings(View(Pairs()), extractor=FlattenExtractor())
        with pytest.raises(MaiteShapeError):
            Embeddings(View(Pairs(), record_reads=True), extractor=FlattenExtractor())

    def test_a_bare_image_dataset_records_nothing(self):
        class Bare:
            def __len__(self) -> int:
                return 3

            def __getitem__(self, index: int) -> Any:
                return np.zeros((3, 4, 4), dtype=np.float32)

        view = View(Bare(), record_reads=True)
        _read_all(view)
        assert view._reads is None

    def test_a_tracking_datum_is_not_recorded(self, factor_videos):
        view = View(factor_videos, record_reads=True)
        view[0]
        assert view._reads is None


@pytest.mark.required
class TestExperimentalWarning:
    def test_passing_it_warns_once(self):
        with pytest.warns(ExperimentalWarning, match="record_reads"):
            View(_Images(), record_reads=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            View(_Images(), record_reads=True)

    def test_the_default_raises_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _read_all(View(_Images()))


def _re_read_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning) and "held" in str(w.message)]


def _frame_warnings(view: Any) -> tuple[Any, list[str]]:
    """Build the metadata of `view` and return its dataframe with the re-read warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = _frame(view)
    return frame, _re_read_warnings(caught)


@pytest.mark.required
class TestReReadWarning:
    """`Metadata` from a recording view that holds only part of the selection warns, with the counts and the fix.

    The record holds item 0 before any pass runs: building a consumer, such as `Embeddings` or `Metadata`, reads it
    to check the dataset, and the view records that read like any other. Hence "held 1" when no pass ran.
    """

    def test_metadata_without_a_pixel_pass_warns(self):
        view = View(_Images(), record_reads=True)
        frame, messages = _frame_warnings(view)
        expected = (
            f"View(record_reads=True) held 1 of the {N} dataset items it selects, so Metadata reads the other {N - 1} "
            "from the dataset. Run the pass that reads the pixels on this view, such as "
            "Embeddings(view, ...).compute(), before building Metadata from it."
        )
        assert messages == [expected]
        assert_frame_equal(frame, _frame(View(_Images())))

    def test_an_embeddings_never_computed_warns(self):
        view = View(_Images(), record_reads=True)
        Embeddings(view, extractor=FlattenExtractor(), batch_size=8)
        _, messages = _frame_warnings(view)
        assert len(messages) == 1
        assert f"held 1 of the {N} dataset items" in messages[0]

    def test_a_computed_pass_does_not_warn(self):
        view = View(_Images(), record_reads=True)
        Embeddings(view, extractor=FlattenExtractor(), batch_size=8).compute()
        _, messages = _frame_warnings(view)
        assert messages == []

    def test_a_pass_over_part_of_the_view_warns_with_the_counts(self):
        view = View(_Images(), record_reads=True)
        Embeddings(View(view, Limit(10)), extractor=FlattenExtractor(), batch_size=8).compute()
        frame, messages = _frame_warnings(view)
        assert len(messages) == 1
        assert f"held 10 of the {N} dataset items it selects, so Metadata reads the other {N - 10}" in messages[0]
        assert_frame_equal(frame, _frame(View(_Images())))

    def test_a_filter_that_structured_every_item_does_not_warn(self):
        view = View(_Images(), Filter(pl.col("group") == "x"), record_reads=True)
        _, messages = _frame_warnings(view)
        assert messages == []

    def test_a_second_metadata_on_the_same_view_does_not_warn_again(self):
        view = View(_Images(), record_reads=True)
        _, first = _frame_warnings(view)
        _, second = _frame_warnings(view)
        assert len(first) == 1
        assert second == []

    def test_a_filter_completing_the_record_while_the_view_is_built_does_not_warn(self):
        class _Peek(Operation):
            def apply(self, view: View[Any]) -> None:
                view[0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            View(_Images(), [_Peek(), Filter(pl.col("group") == "x")], record_reads=True)
        assert _re_read_warnings(caught) == []

    @pytest.mark.parametrize("path", ["dataframe", "factor_names", "evaluator"])
    def test_the_warning_points_at_the_line_that_builds_the_metadata(self, path):
        # The depth from the warning to this line differs by path, so a fixed stack level would miss some of them.
        view = View(_Images(), record_reads=True)
        frame = inspect.currentframe()
        assert frame is not None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if path == "dataframe":
                line = frame.f_lineno + 1
                _ = Metadata(view).dataframe
            elif path == "factor_names":
                line = frame.f_lineno + 1
                _ = Metadata(view).factor_names
            else:
                line = frame.f_lineno + 1
                Balance().evaluate(view)
        held = [w for w in caught if "held" in str(w.message)]
        assert len(held) == 1
        assert (held[0].filename, held[0].lineno) == (__file__, line)
