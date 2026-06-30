"""Tests for DetectionCrops — the object-detection -> image-classification crop view."""

from collections.abc import Sequence

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.data import DetectionCrops, Reverse, Select
from dataeval.exceptions import MaiteShapeError
from dataeval.protocols import Array, DatasetMetadata, DatumMetadata, ObjectDetectionTarget


class _ODTarget:
    def __init__(self, boxes: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones(len(self.labels), dtype=np.float32)


class _ODDataset:
    """Object-detection dataset with caller-supplied images and per-image targets."""

    def __init__(self, images, targets, index2label=None, ids=None) -> None:
        self._images = images
        self._targets = targets
        # Per-datum ids default to the positional index; pass `ids` to make them differ
        # (e.g. to check source_id traces a datum's identity, not its position).
        self._ids = list(ids) if ids is not None else list(range(len(images)))
        i2l = index2label if index2label is not None else {0: "a", 1: "b", 2: "c"}
        self.metadata = DatasetMetadata(id="toy", index2label=i2l)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Array, ObjectDetectionTarget, DatumMetadata]:
        return self._images[index], self._targets[index], {"id": self._ids[index]}


def _positional_image(h: int = 32, w: int = 32, c: int = 3) -> np.ndarray:
    """Image whose every pixel encodes its (y, x) so crops can be checked against the source."""
    ys, xs = np.mgrid[0:h, 0:w]
    plane = (ys * w + xs).astype(np.int32)
    return np.broadcast_to(plane, (c, h, w)).copy()


def _simple_dataset() -> _ODDataset:
    images = np.stack([_positional_image() for _ in range(3)])
    targets = [
        _ODTarget([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 1]),
        _ODTarget([[5, 5, 15, 25]], [1]),  # non-square: 10 wide x 20 tall
        _ODTarget([[1, 1, 3, 3], [28, 28, 32, 32]], [2, 0]),  # tiny (2x2) + edge box
    ]
    return _ODDataset(images, targets)


@pytest.mark.required
class TestStructure:
    def test_length_is_total_detections(self):
        crops = DetectionCrops(_simple_dataset())
        assert len(crops) == 5  # 2 + 1 + 2
        assert crops.n_dropped == 0

    def test_datum_shape_and_metadata(self):
        crops = DetectionCrops(_simple_dataset())
        crop, onehot, meta = crops[0]
        assert crop.ndim == 3
        assert crop.shape[0] == 3
        assert onehot.shape == (3,)
        assert onehot.dtype == np.float32
        assert int(onehot.argmax()) == 0
        assert meta == {"id": 0, "source_id": 0, "target": 0, "box": [0.0, 0.0, 10.0, 10.0]}

    def test_detection_order_matches_metadata_flattening(self):
        crops = DetectionCrops(_simple_dataset())
        # Labels read off the crop view must equal the source detections in image/detection order.
        assert [int(crops[i][1].argmax()) for i in range(len(crops))] == [0, 1, 1, 2, 0]
        # source_id / target trace back to the originating detection.
        assert [(crops[i][2].get("source_id", -1), crops[i][2].get("target", -1)) for i in range(len(crops))] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (2, 0),
            (2, 1),
        ]

    def test_aligns_one_to_one_with_metadata_class_labels(self):
        """The load-bearing contract: Metadata(crops) lines up 1:1 with the crops."""
        crops = DetectionCrops(_simple_dataset())
        class_labels = np.asarray(Metadata(crops).class_labels)
        assert len(class_labels) == len(crops)
        assert class_labels.tolist() == [int(crops[i][1].argmax()) for i in range(len(crops))]

    def test_index2label_inherited_and_metadata_id_suffixed(self):
        crops = DetectionCrops(_simple_dataset())
        assert crops.index2label == {0: "a", 1: "b", 2: "c"}
        assert crops.metadata["id"] == "toy-crops"
        assert "index2label" in crops.metadata
        assert crops.metadata["index2label"] == {0: "a", 1: "b", 2: "c"}

    def test_index2label_fallback_for_unmapped_label(self):
        images = np.stack([_positional_image()])
        ds = _ODDataset(images, [_ODTarget([[0, 0, 10, 10]], [5])], index2label={0: "a"})
        crops = DetectionCrops(ds)
        assert "index2label" in crops.metadata
        assert crops.index2label[5] == "UNDEFINED_CLASS_5"
        assert crops[0][1].shape == (6,)  # one-hot sized to the highest class index

    def test_empty_dataset(self):
        crops = DetectionCrops(_ODDataset(np.empty((0, 3, 8, 8)), []))
        assert len(crops) == 0

    def test_source_id_is_datum_id_not_position(self):
        """source_id reflects each source datum's metadata id, not its positional index."""
        images = np.stack([_positional_image() for _ in range(3)])
        targets = [
            _ODTarget([[0, 0, 10, 10]], [0]),
            _ODTarget([[5, 5, 15, 15]], [1]),
            _ODTarget([[1, 1, 11, 11]], [2]),
        ]
        # Ids deliberately differ from position (and are strings) to expose the difference.
        ds = _ODDataset(images, targets, ids=["img-a", "img-b", "img-c"])
        crops = DetectionCrops(ds)
        assert [crops[i][2].get("source_id", -1) for i in range(len(crops))] == ["img-a", "img-b", "img-c"]

    def test_source_id_survives_reindexing_view(self):
        """A re-ordering Select view renumbers positions but source_id still tracks the datum id."""
        images = np.stack([_positional_image() for _ in range(3)])
        targets = [
            _ODTarget([[0, 0, 10, 10]], [0]),
            _ODTarget([[5, 5, 15, 15]], [1]),
            _ODTarget([[1, 1, 11, 11]], [2]),
        ]
        ds = _ODDataset(images, targets, ids=["img-a", "img-b", "img-c"])
        crops = DetectionCrops(Select(ds, Reverse()))
        # Reverse maps view positions 0,1,2 onto source ids c,b,a — source_id follows the id,
        # not the (reversed) position it now sits at.
        assert [crops[i][2].get("source_id", -1) for i in range(len(crops))] == ["img-c", "img-b", "img-a"]


@pytest.mark.required
class TestFiltering:
    def test_min_size_drops_small_boxes(self):
        crops = DetectionCrops(_simple_dataset(), min_size=4)
        assert len(crops) == 4  # the 2x2 box is dropped
        assert crops.n_dropped == 1

    def test_degenerate_box_always_dropped(self):
        images = np.stack([_positional_image()])
        ds = _ODDataset(images, [_ODTarget([[5, 5, 5, 10], [0, 0, 8, 8]], [0, 1])], index2label={0: "a", 1: "b"})
        crops = DetectionCrops(ds, min_size=0)  # zero-width box dropped even with min_size=0
        assert len(crops) == 1
        assert crops.n_dropped == 1


@pytest.mark.required
class TestCropGeometry:
    def test_object_off_returns_exact_box_pixels(self):
        crops = DetectionCrops(_simple_dataset(), region="object", square="off")
        crop = crops[0][0]
        assert crop.shape == (3, 10, 10)
        np.testing.assert_array_equal(crop, _positional_image()[:, 0:10, 0:10])

    def test_off_preserves_aspect_ratio(self):
        crops = DetectionCrops(_simple_dataset(), square="off")
        assert crops[2][0].shape == (3, 20, 10)  # the 10x20 box stays rectangular

    def test_expand_and_pad_are_square(self):
        for square in ("expand", "pad"):
            crops = DetectionCrops(_simple_dataset(), square=square)
            assert crops[2][0].shape == (3, 20, 20)

    def test_expand_uses_real_pixels_pad_uses_fill(self):
        # Tall 4x12 box centered in x; squaring needs extra columns.
        img = _positional_image(40, 40)
        ds = _ODDataset(img[None], [_ODTarget([[10, 18, 14, 30]], [0])], index2label={0: "a"})
        expand = DetectionCrops(ds, square="expand", fill="zero")[0][0]
        pad = DetectionCrops(ds, square="pad", fill="zero")[0][0]
        assert not (expand == 0).any()  # expand pulls real neighboring pixels
        assert (pad == 0).any()  # pad injects synthetic fill columns
        # pad keeps the box's 4 real columns, centered in the 12-wide canvas
        off = (12 - 4) // 2
        np.testing.assert_array_equal(pad[:, :, off : off + 4], img[None][0][:, 18:30, 10:14])

    def test_padding_widens_the_box(self):
        crops = DetectionCrops(_simple_dataset(), padding=0.5, square="off")
        # 10x10 box at (0,0) widened by 50% each side -> extends to x,y in [-5,15], clamped to [0,15].
        assert crops[0][0].shape == (3, 15, 15)

    def test_fill_zero_vs_mean(self):
        img = _positional_image(40, 40)
        ds = _ODDataset(img[None], [_ODTarget([[10, 18, 14, 30]], [0])], index2label={0: "a"})
        pad_zero = DetectionCrops(ds, square="pad", fill="zero")[0][0]
        pad_mean = DetectionCrops(ds, square="pad", fill="mean")[0][0]
        assert (pad_zero == 0).any()
        # mean fill equals the per-channel mean of the real box pixels
        region = img[None][0][:, 18:30, 10:14]
        assert pad_mean[0, 0, 0] == pytest.approx(region[0].mean(), abs=1)


@pytest.mark.required
class TestSurround:
    def test_surround_masks_the_object(self):
        crops = DetectionCrops(_simple_dataset(), region="surround", padding=0.5, square="off", fill="zero")
        crop, _, meta = crops[0]
        x0, y0, x1, y1 = (int(v) for v in meta.get("box", ()))
        ox, oy = 0, 0  # box at image origin, padding clamped -> crop origin (0,0)
        # the original box region is zeroed; pixels outside it are retained (non-zero positions)
        assert (crop[:, y0 - oy : y1 - oy, x0 - ox : x1 - ox] == 0).all()
        assert (crop != 0).any()

    def test_surround_requires_padding(self):
        with pytest.raises(ValueError, match="padding > 0"):
            DetectionCrops(_simple_dataset(), region="surround")


@pytest.mark.required
class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"region": "bogus"}, "region must be"),
            ({"square": "bogus"}, "square must be"),
            ({"fill": "bogus"}, "fill must be"),
            ({"padding": -1.0}, "padding must be"),
            ({"min_size": -1}, "min_size must be"),
        ],
    )
    def test_bad_params_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DetectionCrops(_simple_dataset(), **kwargs)

    def test_non_od_dataset_rejected(self, ic_dataset):
        with pytest.raises(MaiteShapeError):
            DetectionCrops(ic_dataset([0, 1], {0: "a", 1: "b"}))
