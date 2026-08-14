"""Tests for the shared geometry-rewrite helper used by target-mutating operations."""

from typing import Any

import numpy as np
import pytest

from dataeval.data import View
from dataeval.data._geometry import GeometryMap, rewrite_geometry
from dataeval.data._view import Operation
from dataeval.protocols import ObjectDetectionTarget


class _ODTarget:
    def __init__(self, boxes, labels, scores=None) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.asarray(scores if scores is not None else [1.0] * len(self.labels), dtype=np.float64)


def _datum(boxes, labels, metadata: dict[str, Any] | None = None, size: tuple[int, int] = (20, 20)):
    image = np.zeros((3, *size), dtype=np.float32)
    return image, _ODTarget(boxes, labels), metadata if metadata is not None else {"id": 0}


def _image(h: int, w: int):
    return np.zeros((3, h, w), dtype=np.float32)


@pytest.mark.required
class TestGeometryMapBoxes:
    def test_identity_leaves_boxes_untouched(self):
        boxes = np.array([[1.0, 2.0, 3.0, 4.0]])
        out, mask = GeometryMap(size=(20, 20)).apply_boxes(boxes)
        assert np.array_equal(out, boxes)
        assert mask.tolist() == [True]

    def test_scale_multiplies_both_corners(self):
        # 10x10 source -> 20x40 destination: sx = 4, sy = 2
        boxes = np.array([[1.0, 2.0, 3.0, 4.0]])
        out, _ = GeometryMap(size=(20, 40), scale=(4.0, 2.0)).apply_boxes(boxes)
        assert out.tolist() == [[4.0, 4.0, 12.0, 8.0]]

    def test_offset_translates_both_corners(self):
        boxes = np.array([[1.0, 2.0, 3.0, 4.0]])
        out, _ = GeometryMap(size=(20, 20), offset=(5.0, 7.0)).apply_boxes(boxes)
        assert out.tolist() == [[6.0, 9.0, 8.0, 11.0]]

    def test_scale_then_offset_applies_in_that_order(self):
        # Letterbox: scale to fit, then offset by the pad. Scale must not scale the pad.
        boxes = np.array([[1.0, 1.0, 3.0, 3.0]])
        out, _ = GeometryMap(size=(20, 20), scale=(2.0, 2.0), offset=(4.0, 0.0)).apply_boxes(boxes)
        assert out.tolist() == [[6.0, 2.0, 10.0, 6.0]]

    def test_negative_offset_is_a_crop(self):
        # Crop starting at x=5, y=3 -> every box shifts by (-5, -3).
        boxes = np.array([[6.0, 5.0, 10.0, 9.0]])
        out, mask = GeometryMap(size=(20, 20), offset=(-5.0, -3.0)).apply_boxes(boxes)
        assert out.tolist() == [[1.0, 2.0, 5.0, 6.0]]
        assert mask.tolist() == [True]

    def test_partially_outside_box_is_clipped_not_dropped(self):
        boxes = np.array([[-4.0, -4.0, 6.0, 6.0]])
        out, mask = GeometryMap(size=(10, 10)).apply_boxes(boxes)
        assert out.tolist() == [[0.0, 0.0, 6.0, 6.0]]
        assert mask.tolist() == [True]

    def test_box_overhanging_the_far_edge_is_clipped_to_the_canvas(self):
        boxes = np.array([[4.0, 4.0, 40.0, 40.0]])
        out, _ = GeometryMap(size=(10, 12)).apply_boxes(boxes)
        # canvas is (height=10, width=12): x clips to 12, y clips to 10
        assert out.tolist() == [[4.0, 4.0, 12.0, 10.0]]

    def test_fully_outside_box_is_dropped(self):
        boxes = np.array([[1.0, 1.0, 4.0, 4.0], [20.0, 20.0, 30.0, 30.0]])
        out, mask = GeometryMap(size=(10, 10)).apply_boxes(boxes)
        assert mask.tolist() == [True, False]
        assert out.tolist() == [[1.0, 1.0, 4.0, 4.0]]

    def test_box_flush_against_the_edge_is_degenerate_and_dropped(self):
        # Clips to zero width — an edge-touching box carries no pixels.
        boxes = np.array([[10.0, 2.0, 14.0, 6.0]])
        out, mask = GeometryMap(size=(10, 10)).apply_boxes(boxes)
        assert mask.tolist() == [False]
        assert out.shape == (0, 4)

    def test_zero_detections_does_not_raise(self):
        out, mask = GeometryMap(size=(10, 10), scale=(2.0, 2.0)).apply_boxes(np.zeros((0, 4)))
        assert out.shape == (0, 4)
        assert mask.shape == (0,)

    def test_non_positive_scale_is_rejected(self):
        with pytest.raises(ValueError, match="scale"):
            GeometryMap(size=(10, 10), scale=(0.0, 1.0))
        with pytest.raises(ValueError, match="scale"):
            GeometryMap(size=(10, 10), scale=(1.0, -2.0))

    def test_non_positive_size_is_rejected(self):
        with pytest.raises(ValueError, match="size"):
            GeometryMap(size=(0, 10))

    def test_clip_narrows_the_region_boxes_are_clipped_to(self):
        # Letterbox: canvas is 20x20 but only rows 5..15 are real pixels.
        boxes = np.array([[2.0, 0.0, 8.0, 20.0]])
        out, mask = GeometryMap(size=(20, 20), clip=(0.0, 5.0, 20.0, 15.0)).apply_boxes(boxes)
        assert out.tolist() == [[2.0, 5.0, 8.0, 15.0]]
        assert mask.tolist() == [True]

    def test_box_wholly_inside_the_padding_is_dropped(self):
        boxes = np.array([[2.0, 0.0, 8.0, 4.0]])
        _, mask = GeometryMap(size=(20, 20), clip=(0.0, 5.0, 20.0, 15.0)).apply_boxes(boxes)
        assert mask.tolist() == [False]

    def test_degenerate_clip_is_rejected(self):
        with pytest.raises(ValueError, match="clip"):
            GeometryMap(size=(20, 20), clip=(5.0, 5.0, 5.0, 15.0))


@pytest.mark.required
class TestRewriteGeometry:
    def test_returns_the_supplied_image(self):
        datum = _datum([[1.0, 1.0, 4.0, 4.0]], [0])
        image = _image(10, 10)
        out = rewrite_geometry(datum, image, GeometryMap(size=(10, 10)))
        assert out[0] is image

    def test_target_still_satisfies_the_object_detection_protocol(self):
        datum = _datum([[1.0, 1.0, 4.0, 4.0]], [0])
        out = rewrite_geometry(datum, _image(10, 10), GeometryMap(size=(10, 10), scale=(2.0, 2.0)))
        assert isinstance(out[1], ObjectDetectionTarget)

    def test_boxes_are_transformed_and_labels_scores_follow(self):
        datum = _datum([[1.0, 1.0, 4.0, 4.0], [50.0, 50.0, 60.0, 60.0]], [7, 9], size=(80, 80))
        out = rewrite_geometry(datum, _image(20, 20), GeometryMap(size=(20, 20)))
        assert np.asarray(out[1].boxes).tolist() == [[1.0, 1.0, 4.0, 4.0]]
        assert np.asarray(out[1].labels).tolist() == [7]
        assert np.asarray(out[1].scores).tolist() == [1.0]

    def test_per_detection_metadata_stays_length_aligned(self):
        metadata = {"id": 3, "track": [11, 12, 13], "nested": {"conf": [0.1, 0.2, 0.3]}}
        datum = _datum(
            [[1.0, 1.0, 4.0, 4.0], [50.0, 50.0, 60.0, 60.0], [2.0, 2.0, 5.0, 5.0]],
            [0, 1, 2],
            metadata=metadata,
            size=(80, 80),
        )
        out = rewrite_geometry(datum, _image(20, 20), GeometryMap(size=(20, 20)))
        assert len(np.asarray(out[1].boxes)) == 2
        assert out[2]["track"] == [11, 13]
        assert out[2]["nested"]["conf"] == [0.1, 0.3]
        assert out[2]["id"] == 3

    def test_zero_detection_datum_does_not_raise(self):
        datum = _datum(np.zeros((0, 4)), [])
        out = rewrite_geometry(datum, _image(10, 10), GeometryMap(size=(10, 10), scale=(0.5, 0.5)))
        assert len(np.asarray(out[1].boxes)) == 0

    def test_image_classification_target_passes_through_untouched(self):
        target = np.array([0.0, 1.0, 0.0])
        datum = (np.zeros((3, 20, 20), dtype=np.float32), target, {"id": 4})
        out = rewrite_geometry(datum, _image(10, 10), GeometryMap(size=(10, 10), scale=(0.5, 0.5)))
        assert out[1] is target
        assert out[2] is datum[2]

    def test_image_only_datum_passes_the_image_through(self):
        image = _image(10, 10)
        out = rewrite_geometry(np.zeros((3, 20, 20), dtype=np.float32), image, GeometryMap(size=(10, 10)))
        assert out is image

    def test_segmentation_target_is_deferred_with_a_clear_error(self):
        class _SegTarget:
            def __init__(self) -> None:
                self.mask = np.zeros((2, 20, 20), dtype=np.uint8)
                self.labels = np.asarray([0, 1])
                self.scores = np.asarray([1.0, 1.0])

        datum = (np.zeros((3, 20, 20), dtype=np.float32), _SegTarget(), {"id": 0})
        with pytest.raises(NotImplementedError, match="[Ss]egmentation"):
            rewrite_geometry(datum, _image(10, 10), GeometryMap(size=(10, 10), scale=(0.5, 0.5)))


class Halve(Operation):
    """Minimal consumer of the helper, standing in for Resize/Crop."""

    def apply(self, view: View) -> None:
        view.map(self._transform)

    @staticmethod
    def _transform(datum: Any) -> Any:
        image = np.asarray(datum[0])[:, ::2, ::2]
        h, w = image.shape[-2:]
        return rewrite_geometry(datum, image, GeometryMap(size=(h, w), scale=(0.5, 0.5)))


class _ODDataset:
    def __init__(self, n: int, boxes) -> None:
        self._n = n
        self._boxes = boxes
        self.metadata = {"id": "od", "index2label": {0: "a"}}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        return _datum(self._boxes, [0] * len(self._boxes), {"id": index}, size=(20, 20))


@pytest.mark.required
class TestHelperThroughAView:
    def test_empty_dataset_does_not_raise(self):
        view = View(_ODDataset(0, [[1.0, 1.0, 4.0, 4.0]]), [Halve()])
        assert len(view) == 0
        assert list(view) == []

    def test_boxes_are_rewritten_on_read(self):
        view = View(_ODDataset(2, [[2.0, 2.0, 6.0, 6.0]]), [Halve()])
        image, target, _ = view[0]
        assert image.shape[-2:] == (10, 10)
        assert np.asarray(target.boxes).tolist() == [[1.0, 1.0, 3.0, 3.0]]
