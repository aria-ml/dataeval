"""Attaching values that name their rows by key — FE-6 issue 6.3.

``track_stats`` indexes its results by sorted track id within one sequence; a metadata
track row is keyed ``(item_index, track_index)`` in order of first appearance. The two
orders coincide only by accident, so these tests use fixtures where they *disagree* — a
fixture whose track ids happen to appear in sorted order would pass positionally and prove
nothing.
"""

import pytest

from dataeval import Metadata
from dataeval.exceptions import ShapeMismatchError
from tests.metadata.test_structurers import _mot_dataset


def _tracking(shapes):
    metadata = Metadata(_mot_dataset(shapes))
    metadata._structure()
    return metadata


@pytest.mark.required
class TestKeyedAddFactors:
    def test_values_land_on_the_rows_their_keys_name(self):
        """First appearance order is 5, 9; the values arrive sorted-id order 9, 5."""
        md = _tracking([[[5, 9], [5]]])
        assert md._store.frame("track")["track_id"].to_list() == [5, 9]
        md.add_factors({"track_ids": [9, 5], "speed": [0.9, 0.5]}, level="track", key="track_id")
        assert md._store.frame("track")["speed"].to_list() == [0.5, 0.9]

    def test_matching_is_on_item_and_key_together(self):
        """Track ids restart per sequence, so the id alone names a row in every one."""
        md = _tracking([[[5, 9], [5]], [[5], [5, 9]]])
        assert md._store.frame("track").select("item_index", "track_id").rows() == [(0, 5), (0, 9), (1, 5), (1, 9)]
        md.add_factors(
            {"track_ids": [5, 9, 5, 9], "item_index": [1, 1, 0, 0], "speed": [1.5, 1.9, 0.5, 0.9]},
            level="track",
            key="track_id",
        )
        assert md._store.frame("track")["speed"].to_list() == [0.5, 0.9, 1.5, 1.9]

    def test_a_single_item_dataset_supplies_its_own_item_index(self):
        md = _tracking([[[7, 3], [7]]])
        md.add_factors({"track_ids": [3, 7], "speed": [0.3, 0.7]}, level="track", key="track_id")
        assert md._store.frame("track")["speed"].to_list() == [0.7, 0.3]

    def test_the_singular_column_name_is_accepted_too(self):
        md = _tracking([[[7, 3], [7]]])
        md.add_factors({"track_id": [3, 7], "speed": [0.3, 0.7]}, level="track", key="track_id")
        assert md._store.frame("track")["speed"].to_list() == [0.7, 0.3]

    def test_the_key_is_consumed_rather_than_stored(self):
        """It says which row a value belongs to, not anything about the row."""
        md = _tracking([[[7, 3], [7]]])
        md.add_factors({"track_ids": [3, 7], "item_index": [0, 0], "speed": [0.3, 0.7]}, level="track", key="track_id")
        assert "track_ids" not in md._store.columns
        assert "speed" in md._store.frame("track").columns
        assert md._store.frame("track")["item_index"].to_list() == [0, 0], "the real column is untouched"

    def test_an_unnamed_row_is_null_rather_than_absent(self):
        md = _tracking([[[7, 3], [7]]])
        md.add_factors({"track_ids": [7], "speed": [0.7]}, level="track", key="track_id")
        held = md._store.frame("track")["speed"]
        assert held.to_list() == [0.7, None]
        assert len(held) == md.level_counts["track"]

    def test_the_result_is_readable_from_instance_rows(self):
        """A keyed write is an ordinary factor once placed: descendants gather it."""
        md = _tracking([[[7, 3], [7]]])
        md.add_factors({"track_ids": [3, 7], "speed": [0.3, 0.7]}, level="track", key="track_id")
        assert md._store.column("instance", "speed").to_list() == [0.7, 0.3, 0.7]


@pytest.mark.required
class TestKeyedRejections:
    def test_several_items_without_an_item_index_is_rejected(self):
        md = _tracking([[[5]], [[5]]])
        with pytest.raises(ValueError, match="which item each belongs to"):
            md.add_factors({"track_ids": [5], "speed": [1.0]}, level="track", key="track_id")

    def test_an_unknown_key_column_is_rejected(self):
        md = _tracking([[[5, 9]]])
        with pytest.raises(ValueError, match="not a column"):
            md.add_factors({"nope": [5], "speed": [1.0]}, level="track", key="nope")

    def test_missing_key_values_are_rejected(self):
        md = _tracking([[[5, 9]]])
        with pytest.raises(ValueError, match="track_ids"):
            md.add_factors({"speed": [1.0, 2.0]}, level="track", key="track_id")

    def test_duplicate_keys_are_rejected(self):
        md = _tracking([[[5, 9]]])
        with pytest.raises(ValueError, match="not unique"):
            md.add_factors({"track_ids": [5, 5], "speed": [1.0, 2.0]}, level="track", key="track_id")

    def test_a_length_disagreement_is_rejected(self):
        md = _tracking([[[5, 9]]])
        with pytest.raises(ShapeMismatchError, match="one value per key"):
            md.add_factors({"track_ids": [5, 9], "speed": [1.0]}, level="track", key="track_id")

    def test_key_without_a_level_is_rejected(self):
        md = _tracking([[[5, 9]]])
        with pytest.raises(ValueError, match="level="):
            md.add_factors({"track_ids": [5], "speed": [1.0]}, key="track_id")

    def test_key_with_source_index_is_rejected(self):
        md = _tracking([[[5, 9]]])
        with pytest.raises(ValueError, match="pass one"):
            md.add_factors({"track_ids": [5]}, level="track", key="track_id", source_index=[])

    def test_nothing_is_written_when_the_call_is_rejected(self):
        md = _tracking([[[5, 9]]])
        before = set(md._store.columns)
        with pytest.raises(ValueError, match="not unique"):
            md.add_factors({"track_ids": [5, 5], "speed": [1.0, 2.0]}, level="track", key="track_id")
        assert set(md._store.columns) == before


@pytest.mark.required
class TestTrackStatsEquivalences:
    """The two quantities computed in two places, pinned so the claim cannot rot."""

    def test_track_length_and_frame_span_are_derived_by_the_walk(self):
        md = _tracking([[[5, 9], [5], [5, 9]]])
        frame = md._store.frame("track")
        # Track 5 appears in all three frames; track 9 in the first and last, with a gap.
        assert frame["track_length"].to_list() == [3, 2]
        assert frame["frame_span"].to_list() == [3, 3]

    def test_they_are_present_without_any_keyed_write(self):
        """``track_stats``' n_appearances and track_duration would only duplicate these."""
        md = _tracking([[[5, 9], [5]]])
        assert {"track_length", "frame_span"} <= set(md._store.frame("track").columns)
