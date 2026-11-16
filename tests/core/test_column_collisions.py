"""One check refuses a call whose views or channel groups would give two columns, or two details, one name."""

import warnings

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required


@pytest.fixture
def image() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (4, 64, 80), dtype=np.uint8)


def test_two_channel_groups_that_give_one_column_name_raise(image):
    # `surround_nir_intensity` measures a pixel mean, and CONTEXT on `nir` names its surround intensity the same way.
    with pytest.raises(ValueError, match="surround_nir_intensity_mean"):
        image_stats(
            [image],
            boxes=[[[10, 10, 40, 40]]],
            labels=[[0]],
            stats={"nir": ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN, "surround_nir_intensity": ImageStats.PIXEL_MEAN},
            channels={"nir": [3], "surround_nir_intensity": [0]},
        )


def test_names_once_reserved_for_the_surround_measure_without_a_warning(image):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = image_stats(
            [image],
            stats={None: ImageStats.PIXEL_MEAN, "surround": ImageStats.PIXEL_MEAN, "intensity": ImageStats.PIXEL_MEAN},
            channels={"surround": [0], "intensity": [1]},
        )
    assert {"mean", "surround_mean", "intensity_mean"} <= set(result["stats"])


def test_two_details_of_one_name_raise(image):
    # Saturation on a group `surround` and CONTEXT's surround detail on a group `saturated` both give
    # `surround_saturated_valid_fraction`; before the check the second overwrote the first without a word.
    with pytest.raises(ValueError, match="surround_saturated_valid_fraction"):
        image_stats(
            [image],
            boxes=[[[10, 10, 40, 40]]],
            labels=[[0]],
            stats={
                "surround": ImageStats.PIXEL_SATURATED_FRACTION,
                "saturated": ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN,
            },
            channels={"surround": [0], "saturated": [1]},
        )
