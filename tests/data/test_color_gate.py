"""The color statistics are not gate statistics, and evaluators that request them drop them and say so."""

import numpy as np
import pytest

from dataeval.data import QualityGate
from dataeval.data._quality_gate import _numeric_scalar_stats
from dataeval.flags import ImageStats
from dataeval.quality import Outliers

pytestmark = pytest.mark.required

COLUMNS = [
    "color_lightness_mean",
    "color_chroma_mean",
    "color_achromatic_fraction",
    "color_hue_cos_mean",
    "color_hue_sin_mean",
]
IMAGES = [np.random.default_rng(i).integers(0, 256, (3, 16, 16)).astype(np.uint8) for i in range(12)]


@pytest.fixture(autouse=True)
def fresh_statistics():
    _numeric_scalar_stats.cache_clear()
    yield
    _numeric_scalar_stats.cache_clear()


@pytest.mark.parametrize("stat", COLUMNS)
def test_the_gate_refuses_a_color_statistic_and_says_why(stat: str):
    assert stat not in _numeric_scalar_stats()
    with pytest.raises(ValueError, match="color"):
        QualityGate(stat, min=0.0)


def test_the_error_names_the_declaration_a_gate_cannot_give():
    with pytest.raises(ValueError, match="red, green and blue"):
        QualityGate("color_chroma_mean", min=0.0)


def test_the_other_scalar_statistics_are_still_accepted():
    assert {"brightness", "contrast", "darkness", "sharpness"} <= set(_numeric_scalar_stats())


def test_outliers_with_all_runs_and_color_changes_nothing_in_its_result():
    with_all = Outliers(flags=ImageStats.ALL).evaluate(IMAGES).data()
    without = Outliers(flags=ImageStats.ALL & ~ImageStats.COLOR).evaluate(IMAGES).data()
    assert with_all.equals(without)


def test_outliers_asked_for_color_drops_it_and_points_to_image_stats():
    with pytest.warns(UserWarning, match="compute with image_stats"):
        result = Outliers(flags=ImageStats.PIXEL_MEAN | ImageStats.COLOR).evaluate(IMAGES)
    assert not [name for name in result.data()["metric_name"].to_list() if name.startswith("color_")]
