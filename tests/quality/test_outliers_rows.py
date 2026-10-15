"""Locating statistics rows in a Metadata and reading their class at the metadata's view."""

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.quality._outliers_rows import classes_of, row_positions
from dataeval.types import SourceIndex
from tests.metadata.test_structurers import _mot_dataset


def _weather_metadata():
    md = Metadata(_mot_dataset([[1, 3, 2]]))
    md.add_factors({"weather": ["rain", "clear", "snow"]}, level="unit")
    return md.at("unit").classed_by("weather")


@pytest.mark.required
class TestRowPositions:
    def test_item_and_label_rows_are_found(self, get_od_dataset):
        ds = get_od_dataset(list(np.random.default_rng(0).random((4, 3, 16, 16))), 2, True)
        md = Metadata(ds)
        levels, positions = row_positions(md, [SourceIndex(1), SourceIndex(1, 0), SourceIndex(9, 0)])
        assert levels[:2] == [md.item_level, md.label_level]
        assert positions[0] == 1
        assert positions[1] >= 0
        assert positions[2] == -1

    def test_a_frame_row_is_found_by_its_unit_index(self):
        md = Metadata(_mot_dataset([[2, 1, 3]]))
        levels, positions = row_positions(md, [SourceIndex(0, 2, "unit")])
        assert levels == ["unit"]
        assert positions[0] == 2


@pytest.mark.required
class TestClassesOf:
    def test_a_detection_takes_its_frames_class_on_a_unit_view(self):
        md = _weather_metadata()
        levels, positions = row_positions(md, [SourceIndex(0, k) for k in range(6)])
        names = md.index2label
        got = [tuple(names[c] for c in cls or ()) for cls in classes_of(md, levels, positions)]
        assert got == [("rain",), ("clear",), ("clear",), ("clear",), ("snow",), ("snow",)]

    def test_a_row_above_the_view_gets_each_distinct_class_once(self, get_od_dataset):
        ds = get_od_dataset(list(np.random.default_rng(0).random((4, 3, 16, 16))), 3, True)
        md = Metadata(ds)
        levels, positions = row_positions(md, [SourceIndex(i) for i in range(4)])
        for cls in classes_of(md, levels, positions):
            assert cls is not None
            assert list(cls) == sorted(set(cls))
