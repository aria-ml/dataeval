import inspect
from typing import Any

import numpy as np
import pytest

from dataeval.core import compute_stats, image_stats
from dataeval.core._calculators._base import ContextNeed
from dataeval.core._calculators._dimensionstats import DimensionStatCalculator
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._image_stats import _collect_calculator_stats
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.utils._annotations import AnnotationContext
from dataeval.utils.preprocessing import to_bounding_box

pytestmark = pytest.mark.required

SHAPE = (3, 20, 40)
NAN = float("nan")
INF = float("inf")
BOX = (2, 2, 12, 12)
WIDTH = ImageStats.DIMENSION_WIDTH


def images(n: int) -> list[np.ndarray]:
    return [np.random.default_rng(i).random(SHAPE) for i in range(n)]


class _Target:
    def __init__(self, boxes: Any, labels: Any) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self._labels = np.asarray(labels)
        self.label_reads = 0
        self.scores = np.ones((len(self._labels), 1), dtype=np.float32)

    @property
    def labels(self) -> Any:
        self.label_reads += 1
        return self._labels


class _Dataset:
    metadata = DatasetMetadata(id="image_stats_labels")

    def __init__(self, items: list[tuple[Any, Any]]) -> None:
        self.targets = [_Target(boxes, labels) for boxes, labels in items]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> Any:
        return images(len(self))[i], self.targets[i], DatumMetadata(id=i)

    @property
    def label_reads(self) -> int:
        return sum(t.label_reads for t in self.targets)


@pytest.fixture
def declare(monkeypatch):
    """Make `DIMENSION_WIDTH` declare a need and record the context and focal index it is handed."""
    seen: list[tuple[AnnotationContext | None, int | None]] = []

    def install(needs: ContextNeed) -> list[tuple[AnnotationContext | None, int | None]]:
        original = DimensionStatCalculator.get_handlers

        def patched(self):
            handlers = original(self)
            width = handlers[WIDTH]

            def compute():
                seen.append((self.cache.context, self.cache.focal))
                return width.compute()

            handlers[WIDTH] = width._replace(compute=compute, needs=needs)
            return handlers

        monkeypatch.setattr(DimensionStatCalculator, "get_handlers", patched)
        return seen

    return install


@pytest.mark.parametrize("data", ["images", "dataset"])
def test_labels_require_boxes(data):
    source = images(2) if data == "images" else _Dataset([([BOX], [0]), ([BOX], [1])])

    with pytest.raises(ValueError, match=r"labels= requires boxes="):
        image_stats(source, labels=[[0], [1]], stats=ImageStats.DIMENSION)


def test_a_labels_need_with_explicit_boxes_requires_labels(declare):
    seen = declare(ContextNeed.LABELS)

    with pytest.raises(ValueError, match=r"labels="):
        image_stats(images(1), boxes=[[BOX]], stats=WIDTH)

    image_stats(images(1), boxes=[[BOX, (20, 2, 30, 12)]], labels=[[5, 6]], stats=WIDTH)
    context = seen[0][0]
    assert context is not None
    assert context.labels is not None
    assert context.labels.tolist() == [5, 6]
    assert [focal for _, focal in seen] == [None, 0, 1]


def test_a_labels_need_reads_the_datasets_own_labels(declare):
    seen = declare(ContextNeed.LABELS)

    image_stats(_Dataset([([BOX, (20, 2, 30, 12)], [7, 8])]), stats=WIDTH)

    context = seen[0][0]
    assert context is not None
    assert context.labels is not None
    assert context.labels.tolist() == [7, 8]


@pytest.mark.parametrize(
    "legacy", [ImageStats.DIMENSION, ImageStats.ALL & ~ImageStats.SPATIAL], ids=["dimension", "all_but_spatial"]
)
def test_dataset_labels_are_read_only_for_a_labels_need(declare, monkeypatch, legacy):
    items = [([BOX], [0]), ([BOX], [1])]
    plain = _Dataset(items)
    monkeypatch.setattr(
        AnnotationContext,
        "__init__",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("a context was built for a legacy request")),
    )
    image_stats(plain, stats=legacy, per_background=True)  # M6: no context, no labels, for every real flag
    monkeypatch.undo()

    declare(ContextNeed.CONTEXT)
    context_only = _Dataset(items)
    image_stats(context_only, stats=WIDTH)

    declare(ContextNeed.LABELS)
    with_labels = _Dataset(items)
    image_stats(with_labels, stats=WIDTH)

    # Runtime protocol checks may read the attribute too on some Pythons; a labels need adds exactly one read each.
    assert context_only.label_reads == plain.label_reads
    assert with_labels.label_reads - context_only.label_reads == len(items)


def test_labels_length_and_alignment():
    b = BOX
    with pytest.raises(ValueError, match=r"image 1"):
        image_stats(images(2), boxes=[[b, b], [b, b]], labels=[[0, 1], [0]], stats=ImageStats.DIMENSION)

    with pytest.raises(ValueError, match=r"images, boxes and labels differ in length: labels ran out at index 2"):
        image_stats(images(3), boxes=[[b]] * 3, labels=iter([[0], [1]]), stats=ImageStats.DIMENSION)

    # An unannotated image takes a None or empty label entry; an annotated one does not.
    image_stats(images(2), boxes=[None, [b]], labels=[None, [0]], stats=ImageStats.DIMENSION)
    with pytest.raises(ValueError, match=r"image 1.*labels"):
        image_stats(images(2), boxes=[None, [b]], labels=[None, None], stats=ImageStats.DIMENSION)

    with pytest.raises(ValueError, match="integer class ids"):
        image_stats(images(1), boxes=[[b]], labels=[[0.5]], stats=ImageStats.DIMENSION)


def test_labels_are_validated_even_when_no_flag_reads_them():
    boxes = [[BOX, (20, 2, 30, 12)]]
    with pytest.raises(ValueError, match=r"image 0.*labels"):
        image_stats(images(1), boxes=boxes, labels=[[0]], stats=ImageStats.DIMENSION)

    plain = image_stats(images(1), boxes=boxes, stats=ImageStats.DIMENSION)
    labeled = image_stats(images(1), boxes=boxes, labels=[[0, 1]], stats=ImageStats.DIMENSION)
    assert plain["stats"].keys() == labeled["stats"].keys()
    for name in plain["stats"]:
        np.testing.assert_array_equal(plain["stats"][name], labeled["stats"][name])


@pytest.mark.parametrize("per_background", [False, True])
def test_nonfinite_boxes_no_longer_crash_image_stats(per_background):
    result = image_stats(
        images(1),
        boxes=[[(NAN, 0, 10, 10), (0, 0, INF, 10)]],
        stats=ImageStats.DIMENSION_INVALID_BOX | ImageStats.PIXEL_MEAN,
        per_background=per_background,
    )

    assert result["invalid_box_count"] == [2]
    assert result["stats"]["invalid_box"].tolist() == [False, True, True]


def test_a_reversed_box_is_valid_to_the_legacy_statistics_and_invalid_to_the_context(declare):
    seen = declare(ContextNeed.CONTEXT)

    result = image_stats(
        images(1), boxes=[[(10, 0, 2, 10)]], stats=WIDTH | ImageStats.DIMENSION_INVALID_BOX, per_image=False
    )

    assert result["stats"]["invalid_box"].tolist() == [False]
    assert result["invalid_box_count"] == [0]
    context, focal = seen[0]
    assert context is not None
    assert focal is not None
    assert context.status(focal) == "invalid_box"


def test_legacy_statistics_read_the_converted_box():
    flags = WIDTH | ImageStats.DIMENSION_HEIGHT | ImageStats.DIMENSION_INVALID_BOX | ImageStats.PIXEL_MEAN
    boxes = [(2, 2, 12, 12), (10, 0, 2, 10), (-5, -5, 10, 10), (60, 0, 70, 10), (5, 5, 5, 20), (0, 0, NAN, 10)]
    (image,) = images(1)

    result = image_stats([image], boxes=[boxes], stats=flags, per_image=False)

    calculators = CalculatorRegistry.get_calculators(flags)
    for row, entry in enumerate(boxes):
        reference, *_ = _collect_calculator_stats(calculators, image, to_bounding_box(entry, SHAPE))
        for stats in reference:
            for name, values in stats.items():
                np.testing.assert_array_equal(
                    result["stats"][name][row], np.asarray(values[0]).astype(result["stats"][name].dtype)
                )


def test_compute_stats_alias_has_no_labels():
    assert "labels" not in inspect.signature(compute_stats).parameters
