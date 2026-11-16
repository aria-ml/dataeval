"""The COLOR group: five flags named so that their columns are the issue's, in `ALL` and in no smaller group."""

import pytest

from dataeval.flags import ImageStats

pytestmark = pytest.mark.required

COLUMNS = [
    "color_lightness_mean",
    "color_chroma_mean",
    "color_achromatic_fraction",
    "color_hue_cos_mean",
    "color_hue_sin_mean",
]


def test_the_names_split_into_the_issues_column_names():
    # `quality._shared._stat_names` and dataeval-flow's `stat_names_for` both read a flag as <FAMILY>_<STATISTIC>.
    assert [(m.name or "").split("_", 1)[1].lower() for m in ImageStats.COLOR] == COLUMNS


def test_color_is_in_all_and_in_no_smaller_group():
    assert ImageStats.COLOR in ImageStats.ALL
    smaller = (
        ImageStats.BASIC
        | ImageStats.STANDARD
        | ImageStats.NO_HASH
        | ImageStats.PIXEL
        | ImageStats.VISUAL
        | ImageStats.DIMENSION
        | ImageStats.HASH
        | ImageStats.BLUR
        | ImageStats.SPATIAL
        | ImageStats.CONTEXT
    )
    assert not ImageStats.COLOR & smaller


def test_the_first_new_bit_follows_the_last():
    assert ImageStats.VISUAL_COLOR_LIGHTNESS_MEAN.value == 2 * ImageStats.CONTEXT_SIGNAL_CLUTTER_RATIO.value


def test_not_all_is_none():
    assert ~ImageStats.ALL == ImageStats.NONE
