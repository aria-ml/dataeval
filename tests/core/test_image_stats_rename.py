import warnings

import numpy as np
import pytest

from dataeval.core import compute_stats, image_stats
from dataeval.exceptions import DeprecatedWarning
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required


@pytest.fixture
def images():
    return [np.random.default_rng(i).integers(0, 256, (3, 16, 16), dtype=np.uint8) for i in range(4)]


def test_compute_stats_matches_image_stats(images):
    # The decorator warns once per process. This is the suite's only call to `compute_stats`,
    # so the warning is still unspent here whatever order tests run in.
    expected = image_stats(images, stats=ImageStats.PIXEL_MEAN | ImageStats.HASH_XXHASH)
    with pytest.warns(DeprecatedWarning, match="image_stats"):
        got = compute_stats(images, stats=ImageStats.PIXEL_MEAN | ImageStats.HASH_XXHASH)
    np.testing.assert_array_equal(got["stats"]["mean"], expected["stats"]["mean"])
    assert list(got["stats"]["xxhash"]) == list(expected["stats"]["xxhash"])
    assert got["source_index"] == expected["source_index"]


def test_image_stats_default_is_silent_and_raw(images):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default = image_stats(images, stats=ImageStats.PIXEL_MEAN)
    raw = image_stats(images, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)
    np.testing.assert_array_equal(default["stats"]["mean"], raw["stats"]["mean"])


def test_compute_stats_docstring_carries_the_directive():
    assert ".. deprecated:: 1.2" in (compute_stats.__doc__ or "")
