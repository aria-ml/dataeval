"""A request for box-row statistics on data that has no box rows says so once per call, unless it asks for ALL."""

import warnings

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required

IMAGES = [np.random.default_rng(1).integers(0, 256, (3, 32, 40), dtype=np.uint8) for _ in range(2)]


def _messages(stats: ImageStats) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image_stats(IMAGES, boxes=[[], []], stats=stats)
    return [str(w.message) for w in caught if "measured no box rows" in str(w.message)]


@pytest.mark.parametrize("stats", [ImageStats.SPATIAL, ImageStats.CONTEXT, ImageStats.SPATIAL | ImageStats.PIXEL_MEAN])
def test_box_statistics_on_data_without_box_rows_warn_once(stats):
    messages = _messages(stats)
    assert len(messages) == 1
    assert "NaN" in messages[0]


def test_all_on_data_without_box_rows_stays_quiet():
    assert _messages(ImageStats.ALL) == []


def test_data_with_box_rows_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image_stats(IMAGES, boxes=[[[4, 4, 20, 20]], []], stats=ImageStats.SPATIAL)
    assert not [w for w in caught if "measured no box rows" in str(w.message)]
