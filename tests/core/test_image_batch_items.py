"""An item that holds several images is refused, not measured as its first image.

Fixtures: random float32 images of shape (3, 16, 16) from a seeded generator, stacked four to an item as a batch from a
data loader would be, and the same images given one per item, with and without a leading axis of size 1.
"""

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.exceptions import ShapeMismatchError
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates

pytestmark = pytest.mark.required


def _images(n: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 3, 16, 16), dtype=np.float32)


class _Items:
    def __init__(self, items: list[np.ndarray]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.items[index]


def test_image_stats_refuses_an_item_that_holds_a_batch():
    with pytest.raises(ShapeMismatchError, match="one image per"):
        image_stats([_images()], stats=ImageStats.PIXEL)


def test_duplicates_refuses_items_that_hold_batches():
    batch = _images()
    other = batch.copy()
    other[1:] = _images(3, seed=1)  # the same first image, and three different ones
    with pytest.raises(ShapeMismatchError, match="one image per"):
        Duplicates().evaluate(_Items([batch, other]))


def test_a_leading_axis_of_size_one_is_measured_as_the_image():
    images = list(_images())
    plain = image_stats(images, stats=ImageStats.PIXEL)["stats"]
    wrapped = image_stats([image[None] for image in images], stats=ImageStats.PIXEL)["stats"]
    for name in plain:
        assert np.array_equal(np.asarray(plain[name]), np.asarray(wrapped[name]), equal_nan=True), name
