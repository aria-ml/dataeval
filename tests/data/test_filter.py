import pickle
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval.data import ClassBalance, ClassFilter, Filter, Indices, View

# The legacy shim is the compatibility oracle for the new Filter interface.
pytestmark = [
    pytest.mark.filterwarnings("ignore:'ClassFilter' is deprecated:dataeval.exceptions.DeprecatedWarning"),
]

VOCAB = {0: "car", 1: "person", 2: "boat"}


def _snapshot(view: View[Any]) -> tuple[list[int], list[tuple[Any, dict[str, Any]]]]:
    """A view's selection, and each datum's labels and metadata, in a form two views can be compared by."""
    rows = []
    for _, target, metadata in view:
        labels = int(np.argmax(target)) if isinstance(target, np.ndarray) else np.asarray(target.labels).tolist()
        rows.append((labels, {key: np.asarray(value).tolist() for key, value in metadata.items()}))
    return list(view.selection), rows


def _labels(view: View[Any]) -> list[list[int]]:
    return [np.asarray(view[i][1].labels).tolist() for i in range(len(view))]


@dataclass
class _SegTarget:
    mask: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class _Segmentations:
    """Segmentation with one mask plane per detection, or one (H, W) label map per image."""

    def __init__(self, labels: list[list[int]], label_map: bool = False) -> None:
        self._labels = labels
        self._label_map = label_map
        self.metadata = {"id": "segmentations", "index2label": dict(VOCAB)}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Any, _SegTarget, dict[str, Any]]:
        labels = np.asarray(self._labels[index], dtype=np.intp)
        mask = np.zeros((4, 4)) if self._label_map else np.zeros((len(labels), 4, 4))
        target = _SegTarget(mask, labels, np.ones(len(labels), dtype=np.float32))
        return np.zeros((3, 4, 4)), target, {"id": index, "per_mask": list(range(len(labels)))}


class _OddMetadata:
    """Detection data Metadata cannot structure — an integer metadata key, ids of mixed types — which ClassFilter
    has always filtered, so Filter(classes=...) must too."""

    def __init__(self) -> None:
        self._labels = [[0, 1], [1], [2, 0]]
        self.metadata = {"id": "odd", "index2label": dict(VOCAB)}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[Any, Any]]:
        from tests.data.conftest import _ODTarget

        return np.zeros((3, 4, 4)), _ODTarget(self._labels[index]), {"id": "a" if index == 0 else index, 0: "x"}


def _datasets():
    from tests.data.conftest import MockICDataset, MockODDataset

    return {
        "ic": lambda: MockICDataset([0, 1, 2, 1, 0], VOCAB),
        "od": lambda: MockODDataset([[0, 1], [1], [2, 2], [], [0, 2]], VOCAB),
        "od-per-class": lambda: MockODDataset([[0, 1], [1], [2, 2], [], [0, 2]], VOCAB, per_class=True),
        "seg-planes": lambda: _Segmentations([[0, 1, 2], [0, 0], [1, 2], [2, 2]]),
        "seg-label-map": lambda: _Segmentations([[0, 1, 2], [0, 0], [1, 2], [2, 2]], label_map=True),
        "odd-metadata": _OddMetadata,
    }


@pytest.mark.required
class TestConstruction:
    def test_needs_where_or_classes(self):
        with pytest.raises(ValueError, match="needs `where`, `classes`, or both"):
            Filter()

    def test_refuses_empty_classes(self):
        with pytest.raises(ValueError, match="classes=None"):
            Filter(classes=[])

    def test_refuses_non_integer_classes(self):
        with pytest.raises(TypeError, match="integer class labels"):
            Filter(classes=[0, True])
        with pytest.raises(TypeError, match="integer class labels"):
            Filter(classes=["car"])  # type: ignore[list-item]

    def test_accepts_numpy_classes(self):
        assert Filter(classes=np.array([0, 2])).classes is not None
        assert Filter(classes=[np.int64(0)]).classes is not None

    def test_refuses_an_unknown_drop(self):
        with pytest.raises(ValueError, match="`drop` is one of"):
            Filter(classes=[0], drop="all")  # type: ignore[arg-type]

    def test_a_sql_syntax_error_is_a_value_error(self):
        with pytest.raises(ValueError, match="as a SQL predicate"):
            Filter("tod = = 'night'")

    def test_where_must_be_sql_or_an_expression(self):
        with pytest.raises(TypeError, match="SQL string or a polars expression"):
            Filter(42)  # type: ignore[arg-type]

    def test_a_string_and_an_expression_build_the_same_predicate(self):
        assert (
            Filter("tod = 'night'")._expr.meta.root_names() == Filter(pl.col("tod") == "night")._expr.meta.root_names()
        )

    def test_repr_shows_what_was_written(self):
        assert repr(Filter("tod = 'x'", classes=(0, 1))) == "Filter(where=\"tod = 'x'\", classes=(0, 1), drop='both')"

    def test_repr_of_an_expression_has_no_memory_address(self):
        text = repr(Filter(pl.col("tod") == "night"))
        assert " at 0x" not in text
        assert 'col("tod")' in text


@pytest.mark.required
class TestClassPathMatchesClassFilter:
    @pytest.mark.parametrize("name", list(_datasets()))
    @pytest.mark.parametrize("classes", [[0], [0, 2], (1,)])
    @pytest.mark.parametrize("filter_detections", [True, False])
    def test_same_items_targets_and_metadata(self, name, classes, filter_detections):
        dataset = _datasets()[name]()
        expected = _snapshot(View(dataset, ClassFilter(classes, filter_detections)))
        drop = "both" if filter_detections else "items"
        assert _snapshot(View(dataset, Filter(classes=classes, drop=drop))) == expected

    def test_a_0d_class_target_is_class_0(self):
        items = [(None, np.array(1.0), {"id": 0}), (None, np.array([0.0, 1.0]), {"id": 1})]
        assert View(items, Filter(classes=[0])).selection == [0]

    def test_five_column_boxes_are_masked_by_row(self):
        from tests.data.conftest import _ODTarget

        target = _ODTarget([0, 1])
        target.boxes = np.arange(10, dtype=np.float32).reshape(2, 5)
        view = View([(None, target, {"id": 0}), (None, _ODTarget([1]), {"id": 1})], Filter(classes=[0]))
        assert view.selection == [0]
        assert view[0][1].labels.tolist() == [0]
        assert view[0][1].boxes.tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0]]

    def test_the_unsupported_target_message_names_the_operation(self):
        from tests.data.conftest import _ODTarget

        items = [(None, _ODTarget([0]), {"id": 0}), (None, {"not": "a target"}, {"id": 1})]

        class _Mixed:
            metadata = {"id": "mixed", "index2label": dict(VOCAB)}

            def __len__(self) -> int:
                return len(items)

            def __getitem__(self, index: int) -> Any:
                return items[index]

        with pytest.raises(TypeError, match="Filter does not support targets of type"):
            View(_Mixed(), Filter(classes=[0]))


@pytest.mark.required
class TestClassPathDrop:
    def test_detections_keeps_every_image_and_masks(self, od_dataset):
        view = View(od_dataset([[0, 1], [1], [2]], VOCAB), Filter(classes=[0], drop="detections"))
        assert view.selection == [0, 1, 2]
        assert _labels(view) == [[0], [], []]

    def test_detections_on_segmentation_keeps_a_label_map_whole(self):
        view = View(_Segmentations([[0, 1], [2]], label_map=True), Filter(classes=[0], drop="detections"))
        assert view.selection == [0, 1]
        assert _labels(view) == [[0], []]
        assert view[0][1].mask.shape == (4, 4)

    def test_detections_on_classification_raises(self, ic_dataset):
        with pytest.raises(ValueError, match="classification items hold none"):
            View(ic_dataset([0, 1], VOCAB), Filter(classes=[0], drop="detections"))

    def test_numpy_classes_filter_like_a_list(self, od_dataset):
        dataset = od_dataset([[0, 1], [1], [2, 2], [], [0, 2]], VOCAB)
        assert (
            View(dataset, Filter(classes=np.array([0, 2]))).selection == View(dataset, Filter(classes=[0, 2])).selection
        )

    def test_duplicated_items_stay_duplicated_and_masked(self, factor_od):
        view = View(factor_od, [Indices([5, 0, 0, 2]), Filter(classes=[0])])
        assert view.selection == [0, 0]
        assert _labels(view) == [[0], [0]]

    def test_after_class_balance(self, od_dataset):
        dataset = od_dataset([[0, 1], [1], [2, 2], [0]], VOCAB)
        view = View(dataset, [ClassBalance("interclass", num_samples=12), Filter(classes=[0])])
        assert view.selection
        assert all(0 in labels for labels in _labels(view))
        assert all(set(labels) == {0} for labels in _labels(view))

    def test_a_view_that_masks_pickles(self, od_dataset):
        view = View(od_dataset([[0, 1], [1]], VOCAB), Filter(classes=[0]))
        assert _labels(pickle.loads(pickle.dumps(view))) == _labels(view)


@pytest.mark.required
class TestResults:
    def test_an_empty_result_warns(self, od_dataset):
        with pytest.warns(UserWarning, match="kept none of the 2 items"):
            assert len(View(od_dataset([[0], [1]], VOCAB), Filter(classes=[2]))) == 0

    @pytest.mark.parametrize("make", [lambda: Filter(classes=[2]), lambda: ClassFilter([2])], ids=["Filter", "shim"])
    def test_an_empty_result_warns_at_the_view_call(self, od_dataset, make):
        with pytest.warns(UserWarning, match="kept none") as record:
            View(od_dataset([[0], [1]], VOCAB), make())
        assert [w.filename for w in record if "kept none" in str(w.message)] == [__file__]

    def test_a_kept_result_logs_at_info(self, od_dataset, caplog):
        with caplog.at_level("INFO", logger="dataeval.data"):
            View(od_dataset([[0, 1], [1]], VOCAB), Filter(classes=[0]))
        assert "kept 1 of 2 items" in caplog.text

    def test_an_empty_view_is_left_alone(self, od_dataset):
        assert len(View(od_dataset([], VOCAB), Filter(classes=[0]))) == 0

    def test_the_vocabulary_is_unchanged(self, od_dataset):
        view = View(od_dataset([[0, 1], [2]], VOCAB), Filter(classes=[2]))
        assert view.metadata.get("index2label") == VOCAB


def _tracks(view: View[Any]) -> list[list[list[int]]]:
    return [[frame.track_ids.tolist() for frame in view[i][1].frame_tracks] for i in range(len(view))]


@pytest.mark.required
class TestClassPathOnTracking:
    def test_keeps_videos_with_a_kept_label_and_masks_the_rest(self, factor_videos):
        view = View(factor_videos, Filter(classes=[0]))
        assert view.selection == [0, 1, 2]
        assert _tracks(view) == [[[1], [1], [-1]], [[7], [7]], [[3], [], [3]]]

    def test_items_leaves_frames_whole(self, factor_videos):
        view = View(factor_videos, Filter(classes=[1], drop="items"))
        assert view.selection == [0, 2]
        assert _tracks(view) == [[[1, 2], [1], [2, -1]], [[3], [3], [3]]]

    def test_detections_keeps_every_video(self, factor_videos):
        view = View(factor_videos, Filter(classes=[1], drop="detections"))
        assert view.selection == [0, 1, 2]
        assert _tracks(view) == [[[2], [], [2]], [[], []], [[], [3], []]]

    def test_every_frame_is_kept(self, factor_videos):
        view = View(factor_videos, Filter(classes=[1]))
        assert [len(view[i][1].frame_tracks) for i in range(len(view))] == [3, 3]

    def test_a_view_that_masks_tracks_pickles(self, factor_videos):
        view = View(factor_videos, Filter(classes=[0]))
        assert _tracks(pickle.loads(pickle.dumps(view))) == _tracks(view)


def _occl(view: View[Any]) -> list[list[float]]:
    return [list(view[i][2].get("occl", [])) for i in range(len(view))]


@pytest.mark.required
class TestFactorsOnImages:
    def test_an_image_factor_keeps_images_without_detections_too(self, factor_od):
        assert View(factor_od, Filter("tod = 'night'")).selection == [0, 2, 3, 5]

    def test_a_string_and_an_expression_agree(self, factor_od):
        assert View(factor_od, Filter(pl.col("tod") == "night")).selection == [0, 2, 3, 5]

    def test_a_missing_factor_never_matches(self, factor_od):
        assert View(factor_od, Filter("alt > 150")).selection == [1, 2, 3, 5]
        assert View(factor_od, Filter("alt > 150 OR alt IS NULL")).selection == [1, 2, 3, 4, 5]

    def test_nan_is_missing(self):
        from tests.data.conftest import FactorODDataset

        dataset = FactorODDataset([[0], [0], [0]], [{"alt": 100.0}, {"alt": float("nan")}, {"alt": 300.0}])
        assert View(dataset, Filter("alt > 150")).selection == [2]
        assert View(dataset, Filter("alt IS NULL")).selection == [1]

    def test_booleans_compare_to_true(self):
        from tests.data.conftest import FactorODDataset

        dataset = FactorODDataset([[0], [0]], [{"flag": True}, {"flag": False}])
        assert View(dataset, Filter("flag = true")).selection == [0]
        with pytest.raises(ValueError, match="must answer a boolean"):
            View(dataset, Filter("flag"))

    def test_a_quoted_column_name(self):
        from tests.data.conftest import FactorODDataset

        dataset = FactorODDataset([[0], [0]], [{"time of day": "dusk"}, {"time of day": "noon"}])
        assert View(dataset, Filter("\"time of day\" = 'dusk'")).selection == [0]

    def test_detections_with_an_image_predicate_raises(self, factor_od):
        with pytest.raises(ValueError, match="about whole items"):
            View(factor_od, Filter("tod = 'night'", drop="detections"))

    def test_classification(self, factor_ic):
        assert View(factor_ic, Filter("tod = 'night'")).selection == [0, 2, 3]
        assert View(factor_ic, Filter("tod = 'night'", classes=[1])).selection == [3]
        with pytest.raises(ValueError, match="classification items hold none"):
            View(factor_ic, Filter("tod = 'night'", classes=[1], drop="detections"))


@pytest.mark.required
class TestFactorsOnDetections:
    def test_both(self, factor_od):
        view = View(factor_od, Filter("occl < 0.5"))
        assert view.selection == [0, 2, 5]
        assert _labels(view) == [[0], [2, 2], [1]]
        assert _occl(view) == [[0.1], [0.2, 0.3], [0.4]]

    def test_items(self, factor_od):
        view = View(factor_od, Filter("occl < 0.5", drop="items"))
        assert view.selection == [0, 2, 5]
        assert _labels(view) == [[0, 1], [2, 2], [1, 2]]

    def test_detections(self, factor_od):
        view = View(factor_od, Filter("occl < 0.5", drop="detections"))
        assert view.selection == [0, 1, 2, 3, 4, 5]
        assert _labels(view) == [[0], [], [2, 2], [], [], [1]]
        assert _occl(view) == [[0.1], [], [0.2, 0.3], [], [], [0.4]]

    def test_classes_and_where_combine(self, factor_od):
        view = View(factor_od, Filter("occl < 0.5", classes=[0, 2]))
        assert view.selection == [0, 2]
        assert _labels(view) == [[0], [2, 2]]

    def test_an_or_across_levels_is_about_detections(self, factor_od):
        # Image 3 is a night image with no detections, so it has no row to match.
        assert View(factor_od, Filter("tod = 'night' OR class_label = 0")).selection == [0, 2, 4, 5]

    def test_per_class_scores_stay_aligned(self):
        from tests.data.conftest import OD_FACTORS, OD_LABELS, FactorODDataset

        view = View(FactorODDataset(OD_LABELS, OD_FACTORS, per_class=True), Filter("occl < 0.5"))
        for i in range(len(view)):
            target = view[i][1]
            assert target.scores.shape == (len(target.labels), 3)
            assert target.scores.argmax(axis=1).tolist() == target.labels.tolist()

    def test_duplicated_items_stay_duplicated_and_masked(self, factor_od):
        view = View(factor_od, [Indices([5, 0, 0, 2]), Filter("occl < 0.5")])
        assert view.selection == [5, 0, 0, 2]
        assert _labels(view) == [[1], [0], [0], [2, 2]]

    def test_a_view_that_drops_detections_pickles(self, factor_od):
        view = View(factor_od, Filter("occl < 0.5"))
        assert _labels(pickle.loads(pickle.dumps(view))) == _labels(view)


@pytest.mark.required
class TestLevelResolution:
    def test_a_misspelled_column_says_so(self, factor_od):
        with pytest.raises(ValueError, match="no item in the view at this point declares 'tdo'"):
            View(factor_od, Filter("tdo = 'night'"))

    def test_a_double_quoted_value_says_it_names_a_column(self, factor_od):
        with pytest.raises(ValueError, match="declares 'night'.*double quotes name a column in SQL"):
            View(factor_od, Filter('tod = "night"'))

    def test_a_per_detection_factor_with_no_detections_left(self, factor_od):
        with pytest.raises(ValueError, match="declares 'occl'"):
            View(factor_od, [Indices([3]), Filter("occl < 0.5")])

    def test_a_bin_code_column_is_not_a_column(self, factor_od):
        with pytest.raises(ValueError, match="declares 'alt#'"):
            View(factor_od, Filter(pl.col("alt#") > 0))

    def test_a_mixed_type_column_points_at_unusable(self):
        from tests.data.conftest import FactorODDataset

        dataset = FactorODDataset([[0], [0]], [{"x": 1}, {"x": "one"}])
        with pytest.raises(ValueError, match="Metadata.unusable"):
            View(dataset, Filter("x = 1"))

    @pytest.mark.parametrize("where", ["tod > 5", "alt = 'high'"])
    def test_a_type_mismatch_is_a_value_error_naming_the_predicate(self, factor_od, where):
        with pytest.raises(ValueError, match=re.escape(f"could not evaluate where={where!r}")):
            View(factor_od, Filter(where))

    def test_a_factor_only_removed_items_carried(self):
        from tests.data.conftest import FactorODDataset

        dataset = FactorODDataset([[0], [0], [0]], [{"rare": 1}, {}, {}])
        with pytest.raises(ValueError, match="no item in the view at this point declares 'rare'"):
            View(dataset, [Indices([1, 2]), Filter("rare = 1")])

    def test_columns_of_levels_that_share_no_rows(self, factor_videos):
        with pytest.raises(ValueError, match="Filter found no level whose rows answer every column"):
            View(factor_videos, Filter("sequence_index = 0 AND unit_index = 0"))

    def test_segmentation_refuses_where(self):
        with pytest.raises(ValueError, match="needs a dataset dataeval.Metadata can structure"):
            View(_Segmentations([[0, 1]]), Filter("per_mask = 0"))


@pytest.mark.required
class TestComposition:
    def test_relabel_first_filters_the_new_vocabulary(self):
        from dataeval.data import Relabel
        from tests.data.conftest import FactorODDataset

        dataset = FactorODDataset([[0, 1], [2], [1]], [{}, {}, {}], index2label={0: "car", 1: "truck", 2: "boat"})
        relabel = Relabel({"car": "vehicle", "truck": "vehicle", "boat": "boat"}, ["vehicle", "boat"])
        assert View(dataset, [relabel, Filter("class_label = 0")]).selection == [0, 2]
        assert View(dataset, [Filter("class_label = 0"), relabel]).selection == [0]

    def test_chained_detection_filters_equal_one_and(self, factor_od):
        chained = View(factor_od, [Filter("occl < 0.5"), Filter(classes=[0, 2])])
        combined = View(factor_od, Filter("occl < 0.5", classes=[0, 2]))
        assert chained.selection == combined.selection
        assert _labels(chained) == _labels(combined)

    def test_chained_detection_cleaning_equals_one_and(self, factor_od):
        chained = View(
            factor_od,
            [Filter("occl < 0.5", drop="detections"), Filter("class_label IN (0, 2)", drop="detections")],
        )
        combined = View(factor_od, Filter("occl < 0.5 AND class_label IN (0, 2)", drop="detections"))
        assert chained.selection == combined.selection == [0, 1, 2, 3, 4, 5]
        assert _labels(chained) == _labels(combined) == [[0], [], [2, 2], [], [], []]

    def test_chained_image_filters_equal_one_and(self, factor_od):
        chained = View(factor_od, [Filter("tod = 'night'"), Filter("alt > 150")])
        assert chained.selection == View(factor_od, Filter("tod = 'night' AND alt > 150")).selection == [2, 3, 5]

    def test_an_empty_result_warns(self, factor_od):
        with pytest.warns(UserWarning, match="kept none of the 6 items"):
            View(factor_od, Filter("tod = 'fog'"))


@pytest.mark.required
class TestFactorsOnTracking:
    def test_a_sequence_factor_keeps_videos(self, factor_videos):
        assert View(factor_videos, Filter("weather = 'rain'")).selection == [0, 2]

    @pytest.mark.parametrize(
        ("drop", "selection", "tracks"),
        [
            ("both", [2], [[[3], [3], [3]]]),
            ("items", [2], [[[3], [3], [3]]]),
            ("detections", [0, 1, 2], [[[], [], [-1]], [[], []], [[3], [3], [3]]]),
        ],
    )
    def test_a_track_factor_removes_whole_tracks_and_spares_untracked(self, factor_videos, drop, selection, tracks):
        view = View(factor_videos, Filter("track_length > 2", drop=drop))
        assert view.selection == selection
        assert _tracks(view) == tracks

    def test_reading_a_detection_column_removes_untracked_too(self, factor_videos):
        view = View(factor_videos, Filter("track_length > 2 AND class_label IN (0, 1)", drop="detections"))
        assert _tracks(view) == [[[], [], []], [[], []], [[3], [3], [3]]]

    def test_track_id_alone_is_about_tracks(self, factor_videos):
        view = View(factor_videos, Filter("track_id >= 0", drop="detections"))
        assert _tracks(view)[0] == [[1, 2], [1], [2, -1]]

    def test_the_documented_spelling_removes_exactly_the_untracked(self, factor_videos):
        view = View(factor_videos, Filter("track_id >= 0 AND target_index >= 0", drop="detections"))
        assert _tracks(view) == [[[1, 2], [1], [2]], [[7], [7]], [[3], [3], [3]]]

    def test_a_detection_predicate_masks_exactly_and_a_flickering_track_survives_in_part(self, factor_videos):
        structured = View(factor_videos, Filter("class_label = 0"))
        by_target = View(factor_videos, Filter(classes=[0]))
        assert structured.selection == by_target.selection == [0, 1, 2]
        assert _tracks(structured) == _tracks(by_target) == [[[1], [1], [-1]], [[7], [7]], [[3], [], [3]]]

    def test_a_frame_predicate_keeps_videos_with_items(self, factor_videos):
        assert View(factor_videos, Filter("tod = 'night'", drop="items")).selection == [0]

    @pytest.mark.parametrize("drop", ["both", "detections"])
    def test_a_frame_predicate_cannot_remove_frames(self, factor_videos, drop):
        with pytest.raises(ValueError, match="SequenceFrames"):
            View(factor_videos, Filter("tod = 'night'", drop=drop))

    def test_a_track_and_a_frame_factor_resolve_to_detections(self, factor_videos):
        view = View(factor_videos, Filter("track_length > 2 AND tod = 'day'", drop="detections"))
        assert _tracks(view) == [[[], [], []], [[], []], [[3], [3], [3]]]

    def test_an_undeclared_frame_time_is_missing(self, timed_videos):
        assert View(timed_videos, Filter("time_s > 0.4 AND unit_index = 1", drop="items")).selection == [1, 2]
        assert View(timed_videos, Filter("time_s IS NULL", drop="items")).selection == [0]
