"""Tests for feasibility units, labels, replicas and splits."""

from typing import Any

import numpy as np
import pytest

from dataeval.data import DetectionCrops, Operation, SequenceFrames, VideoSegments, VideoStitch, View, Window
from dataeval.performance._feasibility_units import (
    _dataset_id,
    resolve_detection_units,
    resolve_replicas,
    resolve_units,
    split_units,
)
from tests.data.conftest import MockODDataset
from tests.data.test_crops import _simple_dataset
from tests.data.test_frames import make_dataset
from tests.data.test_stitch import make_mot


class _IC:
    """Classification dataset; a ``None`` label is an unlabeled item (empty target vector)."""

    def __init__(
        self, labels, *, declare: bool | dict = True, factors=None, ids: Any = "positional", name: str | None = "toy"
    ):
        self._labels = list(labels)
        self._factors = factors or {}
        self._ids = list(range(len(self._labels))) if ids == "positional" else ids
        self.reads = 0
        self.metadata: dict[str, Any] = {} if name is None else {"id": name}
        if declare:
            self.metadata["index2label"] = {0: "a", 1: "b", 2: "c"} if declare is True else declare

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, i: int):
        self.reads += 1
        label = self._labels[i]
        target = np.zeros(0) if label is None else np.eye(3, dtype=np.float32)[label]
        meta = ({} if self._ids is None else {"id": self._ids[i]}) | {k: v[i] for k, v in self._factors.items()}
        return np.zeros((3, 2, 2), dtype=np.float32), target, meta


class _Repeat(Operation):
    """Select source items 0, 0, 1, 2, 2: a view whose items repeat."""

    def apply(self, view: View) -> None:
        view.selection = [0, 0, 1, 2, 2]


@pytest.mark.required
class TestUnits:
    def test_unlabeled_items_are_excluded_and_population_is_conditioned(self):
        units = resolve_units(_IC([0, None, 1, 2, None]), groups=None)
        np.testing.assert_array_equal(units.labels, [0, 1, 2])
        np.testing.assert_array_equal(units.item_indices, [0, 2, 3])
        np.testing.assert_array_equal(units.groups, [0, 1, 2])
        assert units.n_excluded == 2
        assert units.population == "labeled_items"
        assert not units.grouped
        assert resolve_units(_IC([0, 1, 2]), groups=None).population == "items"

    @pytest.mark.parametrize("kind", ["view", "segments", "frames", "crops", "stitch"])
    def test_derived_views_require_groups(self, kind):
        views = {
            "view": lambda: View(View(_IC([0, 1, 2]))),  # a view over a view is derived twice
            "segments": lambda: VideoSegments(make_dataset()[0], Window(2)),
            "frames": lambda: SequenceFrames(make_dataset()[0]),
            "crops": lambda: DetectionCrops(_simple_dataset()),
            "stitch": lambda: VideoStitch(make_mot([[[1], [1]], [[2], [2]]])[0], group_by=None),
        }
        view = views[kind]()
        with pytest.raises(ValueError, match=rf"^{type(view).__name__} is a derived view.*groups="):
            resolve_units(view, groups=None)

    def test_groups_array_and_factor_name(self):
        ds = _IC([0, None, 1, 2], factors={"scene": ["x", "x", "y", "y"]})
        by_array = resolve_units(ds, groups=[7, 7, 3, 3])
        np.testing.assert_array_equal(by_array.groups, [1, 0, 0])
        assert by_array.grouped
        by_name = resolve_units(ds, groups="scene")
        np.testing.assert_array_equal(by_name.groups, [0, 1, 1])
        with pytest.raises(ValueError, match="names a metadata factor"):
            resolve_units(ds, groups="nope")
        with pytest.raises(ValueError, match="group ids"):
            resolve_units(ds, groups=[0, 1])

    def test_groups_factor_name_reads_raw_values(self):
        # Continuous factor: raw values group by equality, not by any binning.
        ds = _IC([0, 1, 2, 0], factors={"t": [0.1, 0.1000001, 0.1, 5.0]})
        np.testing.assert_array_equal(resolve_units(ds, groups="t").groups, [0, 1, 0, 2])

    def test_derived_view_with_groups_succeeds(self):
        crops = DetectionCrops(_simple_dataset())
        units = resolve_units(crops, groups=[0, 0, 1, 1, 2])
        assert units.grouped
        np.testing.assert_array_equal(units.groups, [0, 0, 1, 1, 2])
        view = View(_IC([0, 1, 2, 0]))
        np.testing.assert_array_equal(resolve_units(view, groups=[5, 5, 6, 6]).groups, [0, 0, 1, 1])

    def test_a_view_over_a_plain_dataset_groups_by_source_index(self):
        subset = resolve_units(View(_IC([0, 1, 2, 0])), groups=None)
        np.testing.assert_array_equal(subset.groups, [0, 1, 2, 3])
        assert not subset.grouped  # every group holds one item
        repeated = resolve_units(View(_IC([0, 1, 2]), _Repeat()), groups=None)
        np.testing.assert_array_equal(repeated.groups, [0, 0, 1, 2, 2])
        assert repeated.grouped

    def test_grouped_means_some_group_holds_several_items(self):
        assert not resolve_units(_IC([0, 1, 2]), groups=[7, 8, 9]).grouped
        assert resolve_units(_IC([0, 1, 2]), groups=[7, 7, 9]).grouped

    def test_non_classification_is_rejected(self):
        with pytest.raises(ValueError, match="classification datasets only"):
            resolve_units(MockODDataset([[0, 1], [2]], {0: "a", 1: "b", 2: "c"}), groups=None)

    def test_dataset_id_falls_back_to_type_name(self):
        assert _dataset_id(_IC([0, 1], name=None)) == "_IC"
        assert resolve_units(_IC([0, 1], name=None), groups=None).population == "items"

    def test_n_classes_and_sparse_declared_names(self):
        sparse = resolve_units(_IC([0, 2], declare={0: "a", 2: "c"}), groups=None)
        assert sparse.class_names == ["a", "c"]
        assert sparse.n_classes == 3
        assert resolve_units(_IC([0, 2], declare=False), groups=None).n_classes == 3

    def test_no_declared_index2label_gives_no_class_names(self):
        assert resolve_units(_IC([0, 1], declare=False), groups=None).class_names is None
        assert resolve_units(_IC([0, 1]), groups=None).class_names == ["a", "b", "c"]


@pytest.mark.required
class TestReplicas:
    def test_maps_datum_ids_to_labeled_positions(self):
        ds = _IC([0, None, 1, 2], ids=["p", "q", "r", "s"])
        units = resolve_units(ds, groups=None)
        out = resolve_replicas({"r": [2, 0], "s": np.array([1])}, ds, units)
        assert out.keys() == {1, 2}
        np.testing.assert_array_equal(out[1], [2, 0])
        assert resolve_replicas({}, ds, units) == {}

    def test_replicas_do_not_read_the_dataset(self):
        ds = _IC([0, 1, 2], ids=["a", "b", "c"])
        units = resolve_units(ds, groups=None)
        ds.reads = 0
        assert resolve_replicas({"b": [0]}, ds, units).keys() == {1}
        assert ds.reads == 0

    def test_dataset_without_ids_is_keyed_by_position(self):
        ds = _IC([0, None, 1], ids=None)
        out = resolve_replicas({2: [1]}, ds, resolve_units(ds, groups=None))
        np.testing.assert_array_equal(out[1], [1])

    def test_sparse_declared_classes_accept_the_highest_index(self):
        ds = _IC([0, 2], declare={0: "a", 2: "c"}, ids=["x", "y"])
        units = resolve_units(ds, groups=None)
        np.testing.assert_array_equal(resolve_replicas({"x": [2]}, ds, units)[0], [2])
        with pytest.raises(ValueError, match="outside"):
            resolve_replicas({"x": [3]}, ds, units)

    def test_duplicate_datum_ids_reject_replicas(self):
        ds = _IC([0, 1, 2], ids=["a", "a", "b"])
        with pytest.raises(ValueError, match="repeats ids"):
            resolve_replicas({"b": [0]}, ds, resolve_units(ds, groups=None))

    def test_rejects_unknown_unlabeled_and_out_of_range(self):
        ds = _IC([0, None, 1], ids=["a", "b", "c"])
        units = resolve_units(ds, groups=None)
        with pytest.raises(ValueError, match="not a datum id"):
            resolve_replicas({"z": [0]}, ds, units)
        with pytest.raises(ValueError, match="unlabeled"):
            resolve_replicas({"b": [0]}, ds, units)
        with pytest.raises(ValueError, match="outside"):
            resolve_replicas({"c": [3]}, ds, units)
        with pytest.raises(ValueError, match="integer"):
            resolve_replicas({"c": [0.5]}, ds, units)


@pytest.mark.required
class TestSplits:
    def _units(self):
        ds = _IC([i % 3 for i in range(12)])
        return resolve_units(ds, groups=np.repeat(np.arange(4), 3))

    def test_splits_are_group_disjoint(self):
        units = self._units()
        fit, audit = split_units(units, fit_fraction=0.5, seed=0)
        assert set(units.groups[fit]).isdisjoint(units.groups[audit])
        np.testing.assert_array_equal(np.sort(np.concatenate([fit, audit])), np.arange(12))
        again = split_units(units, fit_fraction=0.5, seed=0)
        np.testing.assert_array_equal(fit, again[0])
        assert len(set(units.groups[fit])) == 2
        other = [split_units(units, fit_fraction=0.5, seed=s)[0].tolist() for s in range(1, 8)]
        assert any(o != fit.tolist() for o in other)

    def test_empty_side_raises(self):
        with pytest.raises(ValueError, match="empty"):
            split_units(self._units(), fit_fraction=0.99, seed=0)


class _Boxes:
    def __init__(self, boxes, labels) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones(len(self.labels))


class _OD:
    """Detection dataset: per image a list of ``(box, label)``; ``factors`` per image, lists for per-box values."""

    def __init__(self, images, *, declare: bool | dict = True, factors=None) -> None:
        self._targets = [_Boxes([b for b, _ in image], [c for _, c in image]) for image in images]
        self._factors = factors or {}
        self.metadata: dict[str, Any] = {"id": "toy-od"}
        if declare:
            self.metadata["index2label"] = {0: "car", 1: "person"} if declare is True else declare

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, i: int):
        meta = {"id": f"im{i}"} | {k: v[i] for k, v in self._factors.items()}
        return np.zeros((3, 2, 2), dtype=np.float32), self._targets[i], meta


_CAR, _PERSON = [0.0, 0.0, 10.0, 10.0], [473.07, 212.13, 480.11, 219.59]


@pytest.mark.required
class TestDetectionUnits:
    def test_every_image_is_a_unit_with_its_boxes_and_labels(self):
        units = resolve_detection_units(_OD([[(_CAR, 0), (_PERSON, 1)], [], [(_PERSON, 1)]]), groups=None)
        assert len(units.boxes) == len(units.labels) == 3
        np.testing.assert_array_equal(units.boxes[0], [_CAR, _PERSON])  # float64, as the dataset gave them
        np.testing.assert_array_equal(units.labels[0], [0, 1])
        assert units.boxes[1].shape == (0, 4)  # an empty image stays in the population
        assert units.labels[1].shape == (0,)
        np.testing.assert_array_equal(units.groups, [0, 1, 2])
        assert units.item_ids.tolist() == ["im0", "im1", "im2"]
        assert units.n_classes == 2
        assert units.class_names == ["car", "person"]
        assert not units.grouped

    def test_an_image_level_factor_groups_images(self):
        ds = _OD([[(_CAR, 0)], [], [(_CAR, 0)]], factors={"site": ["a", "b", "a"]})
        units = resolve_detection_units(ds, groups="site")
        np.testing.assert_array_equal(units.groups, [0, 1, 0])
        assert units.grouped

    def test_a_box_level_factor_cannot_group_images(self):
        ds = _OD([[(_CAR, 0), (_CAR, 1)], [], [(_CAR, 0)]], factors={"occluded": [[0, 1], [], [1]]})
        with pytest.raises(ValueError, match="groups='occluded' names a metadata factor"):
            resolve_detection_units(ds, groups="occluded")

    def test_a_groups_array_is_one_per_image(self):
        ds = _OD([[(_CAR, 0)], [], [(_CAR, 0)]])
        np.testing.assert_array_equal(resolve_detection_units(ds, groups=[4, 4, 9]).groups, [0, 0, 1])
        with pytest.raises(ValueError, match="group ids"):
            resolve_detection_units(ds, groups=[0, 1])

    def test_a_view_over_a_plain_dataset_groups_by_source_index(self):
        units = resolve_detection_units(View(_OD([[(_CAR, 0)], [], [(_CAR, 1)]]), _Repeat()), groups=None)
        np.testing.assert_array_equal(units.groups, [0, 0, 1, 2, 2])
        assert units.grouped
        np.testing.assert_array_equal(units.labels[4], [1])

    def test_a_derived_view_needs_groups(self):
        with pytest.raises(ValueError, match="derived view.*groups="):
            resolve_detection_units(View(View(_OD([[(_CAR, 0)], [(_CAR, 1)]]))), groups=None)

    @pytest.mark.parametrize("box", [[0.0, 0.0, 0.0, 10.0], [5.0, 0.0, 1.0, 10.0], [0.0, 0.0, np.inf, 1.0]])
    def test_a_target_box_no_detection_could_match_is_refused(self, box):
        ds = _OD([[(_CAR, 0)], [(_CAR, 0), (box, 1)]])
        with pytest.raises(ValueError, match="^dataset item 1 targets: box 1"):
            resolve_detection_units(ds, groups=None)

    def test_a_label_outside_the_declared_classes_is_refused(self):
        with pytest.raises(ValueError, match=r"item 0 .*label 2.*\[0, 2\)"):
            resolve_detection_units(_OD([[(_CAR, 2)]]), groups=None)

    def test_without_declared_classes_the_largest_label_sets_the_count(self):
        units = resolve_detection_units(_OD([[(_CAR, 3)], []], declare=False), groups=None)
        assert units.n_classes == 4
        assert units.class_names is None

    def test_classification_is_refused(self):
        with pytest.raises(ValueError, match="object detection datasets only"):
            resolve_detection_units(_IC([0, 1, 2]), groups=None)

    def test_split_takes_detection_units(self):
        units = resolve_detection_units(_OD([[(_CAR, 0)]] * 6, declare=True), groups=[0, 0, 1, 1, 2, 2])
        fit, audit = split_units(units, fit_fraction=0.5, seed=0)
        assert sorted(np.r_[fit, audit].tolist()) == list(range(6))
        assert set(units.groups[fit]).isdisjoint(units.groups[audit])
