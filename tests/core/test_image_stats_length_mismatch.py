import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required

BOX = [[2, 2, 8, 8]]


def _images(n: int):
    return [np.random.default_rng(i).random((3, 16, 16)) for i in range(n)]


def test_more_boxes_than_images_raises():
    with pytest.raises(ValueError, match="images ran out at index 3"):
        image_stats(_images(3), boxes=[BOX] * 4, stats=ImageStats.DIMENSION)


def test_more_images_than_boxes_raises():
    with pytest.raises(ValueError, match="boxes ran out at index 3"):
        image_stats(_images(4), boxes=[BOX] * 3, stats=ImageStats.DIMENSION)


def test_explicit_none_boxes_entry_is_an_unannotated_image():
    result = image_stats(_images(3), boxes=[BOX, None, BOX], stats=ImageStats.DIMENSION)
    assert result["image_count"] == 3
    box_items = [s.item for s in result["source_index"] if s.key is not None]
    assert box_items == [0, 2]


@pytest.mark.parametrize(("n_images", "n_boxes", "match"), [(3, 4, "images ran out"), (4, 3, "boxes ran out")])
def test_one_pass_iterables_raise_without_a_partial_result(n_images, n_boxes, match):
    with pytest.raises(ValueError, match=match):
        image_stats(iter(_images(n_images)), boxes=(BOX for _ in range(n_boxes)), stats=ImageStats.DIMENSION)


def test_ndarray_boxes_entry_is_accepted():
    boxes = np.array([[2, 2, 8, 8], [1, 1, 5, 5]])
    result = image_stats(_images(2), boxes=[boxes, boxes], stats=ImageStats.DIMENSION)
    box_items = [s.item for s in result["source_index"] if s.key is not None]
    assert box_items == [0, 0, 1, 1]
