import polars as pl
import pytest

from dataeval import Metadata
from dataeval.exceptions import ShapeMismatchError
from dataeval.types import SourceIndex
from tests.metadata.test_review_fixes import _od_dataset
from tests.metadata.test_video_stats_landing import _video


def _by_key(md, level, column):
    key = {"unit": "unit_index", "track": "track_id", "instance": "target_index"}[level]
    frame = md._store.frame(level)
    pairs = zip(frame["item_index"].to_list(), frame[key].to_list(), strict=True)
    return dict(zip(pairs, frame[column].to_list(), strict=True))


@pytest.mark.required
class TestKeyedAddressesMayNameSomeRows:
    def test_some_frames(self):
        md = _video()
        md.add_factors({"blur": [0.2, 1.0]}, source_index=[SourceIndex(0, 2, "unit"), SourceIndex(1, 0, "unit")])
        assert _by_key(md, "unit", "blur") == {(0, 0): None, (0, 1): None, (0, 2): 0.2, (1, 0): 1.0, (1, 1): None}

    def test_some_detections(self):
        md = _video()
        md.add_factors({"iou": [0.5, 0.7]}, source_index=[SourceIndex(0, 3, "instance"), SourceIndex(1, 1, "instance")])
        got = _by_key(md, "instance", "iou")
        assert got[(0, 3)] == 0.5
        assert got[(1, 1)] == 0.7
        assert sum(v is not None for v in got.values()) == 2

    def test_frames_and_detections_in_one_call(self):
        md = _video()
        md.add_factors({"x": [1.0, 2.0]}, source_index=[SourceIndex(0, 1, "unit"), SourceIndex(1, 2, "instance")])
        assert _by_key(md, "unit", "unit_x")[(0, 1)] == 1.0
        assert _by_key(md, "instance", "instance_x")[(1, 2)] == 2.0

    def test_some_boxes_of_an_image_dataset(self):
        """Two boxes named in an image dataset of 3 images holding 2, 3 and 1 boxes."""
        md = Metadata(_od_dataset(counts=(2, 3, 1)))
        md.add_factors({"iou": [0.9, 0.1]}, source_index=[SourceIndex(1, 2), SourceIndex(0, 0)])
        got = _by_key(md, "instance", "iou")
        assert got[(1, 2)] == 0.9
        assert got[(0, 0)] == 0.1
        assert sum(v is not None for v in got.values()) == 2


@pytest.mark.required
class TestPartialWritesFoldLikeKeyedWrites:
    def test_two_partial_writes_build_one_column(self):
        md = _video()
        md.add_factors({"blur": [0.0, 0.1]}, source_index=[SourceIndex(0, 0, "unit"), SourceIndex(0, 1, "unit")])
        md.add_factors({"blur": [1.0, 1.1]}, source_index=[SourceIndex(1, 0, "unit"), SourceIndex(1, 1, "unit")])
        assert "blur_added" not in md._store.columns
        assert _by_key(md, "unit", "blur") == {(0, 0): 0.0, (0, 1): 0.1, (0, 2): None, (1, 0): 1.0, (1, 1): 1.1}

    def test_naming_a_filled_row_again_collides(self):
        md = _video()
        md.add_factors({"blur": [0.0]}, source_index=[SourceIndex(0, 0, "unit")])
        md.add_factors({"blur": [9.0]}, source_index=[SourceIndex(0, 0, "unit")])
        assert _by_key(md, "unit", "blur_added")[(0, 0)] == 9.0

    def test_overwrite_replaces_only_named_rows_and_keeps_the_original(self):
        md = _video()
        md.add_factors({"blur": [0.0, 0.1]}, source_index=[SourceIndex(0, 0, "unit"), SourceIndex(0, 1, "unit")])
        md.add_factors({"blur": [9.0]}, source_index=[SourceIndex(0, 1, "unit")], overwrite=True)
        assert _by_key(md, "unit", "blur")[(0, 0)] == 0.0
        assert _by_key(md, "unit", "blur")[(0, 1)] == 9.0
        assert _by_key(md, "unit", "blur_original")[(0, 1)] == 0.1


@pytest.mark.required
class TestWhatPartialAddressesAreStillRefused:
    def test_unkeyed_addresses_still_cover_every_row(self):
        md = _video()
        with pytest.raises(ShapeMismatchError, match="1 sequence-level values but the metadata has 2"):
            md.add_factors({"night": [0.0]}, source_index=[SourceIndex(0)])

    def test_a_row_named_twice(self):
        md = _video()
        with pytest.raises(ValueError, match="more than once"):
            md.add_factors({"blur": [0.0, 0.1]}, source_index=[SourceIndex(0, 2, "unit"), SourceIndex(0, 2, "unit")])

    def test_a_row_the_metadata_does_not_have(self):
        md = _video()
        with pytest.raises(ValueError, match="rows this metadata does not have"):
            md.add_factors({"blur": [0.0]}, source_index=[SourceIndex(0, 7, "unit")])

    def test_nothing_is_written_when_refused(self):
        md = _video()
        before = set(md._store.columns)
        with pytest.raises(ValueError, match="more than once"):
            md.add_factors({"blur": [0.0, 0.1]}, source_index=[SourceIndex(0, 2, "unit"), SourceIndex(0, 2, "unit")])
        assert set(md._store.columns) == before


@pytest.mark.required
class TestAWriteNamingEveryRowReplacesTheColumn:
    """Nothing of the old column survives a keyed overwrite naming every row, so neither does its dtype."""

    def test_a_full_keyed_source_index_keeps_the_incoming_dtype(self):
        md = Metadata(_od_dataset(counts=(2, 3, 1)))
        source_index = [SourceIndex(item, box) for item, count in enumerate((2, 3, 1)) for box in range(count)]
        md.add_factors({"f": list("abcdef")}, source_index=source_index)
        md.add_factors({"f": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]}, source_index=source_index, overwrite=True)
        assert md._store.frame("instance")["f"].dtype == pl.Float64

    def test_a_full_key_write_keeps_the_incoming_dtype(self):
        md = _video()
        keys = {"item_index": [0, 0, 0, 1, 1], "unit_index": [0, 1, 2, 0, 1]}
        md.add_factors({**keys, "blur": list("abcde")}, level="unit", key="unit_index")
        md.add_factors({**keys, "blur": [0.0, 0.1, 0.2, 1.0, 1.1]}, level="unit", key="unit_index", overwrite=True)
        assert md._store.frame("unit")["blur"].dtype == pl.Float64
