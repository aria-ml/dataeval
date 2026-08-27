"""Attaching values that name their rows by key — FE-6 issue 6.3.

``track_stats`` indexes its results by sorted track id within one sequence; a metadata
track row is keyed ``(item_index, track_index)`` in order of first appearance. The two
orders coincide only by accident, so these tests use fixtures where they *disagree* — a
fixture whose track ids happen to appear in sorted order would pass positionally and prove
nothing.
"""

import polars as pl
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

    def test_a_second_write_folds_into_the_same_column(self):
        """One sequence per call is what ``track_stats`` invites; it builds one column."""
        md = _tracking([[[5, 9], [5]], [[5], [5, 9]]])
        md.add_factors({"track_ids": [5, 9], "item_index": [0, 0], "speed": [0.5, 0.9]}, level="track", key="track_id")
        assert md._store.frame("track")["speed"].to_list() == [0.5, 0.9, None, None]

        md.add_factors({"track_ids": [5, 9], "item_index": [1, 1], "speed": [1.5, 1.9]}, level="track", key="track_id")
        assert md._store.frame("track")["speed"].to_list() == [0.5, 0.9, 1.5, 1.9]
        assert "speed_added" not in md._store.columns

    def test_a_row_written_twice_is_a_real_collision(self):
        """Two values for one row is a name collision like any other, and is renamed."""
        md = _tracking([[[5, 9], [5]]])
        md.add_factors({"track_ids": [5, 9], "speed": [0.5, 0.9]}, level="track", key="track_id")
        md.add_factors({"track_ids": [5], "speed": [99.0]}, level="track", key="track_id")
        assert md._store.frame("track")["speed"].to_list() == [0.5, 0.9]
        assert md._store.frame("track")["speed_added"].to_list() == [99.0, None]

    def test_overwrite_replaces_only_the_rows_the_keys_name(self):
        md = _tracking([[[5, 9], [5]]])
        md.add_factors({"track_ids": [5, 9], "speed": [0.5, 0.9]}, level="track", key="track_id")
        md.add_factors({"track_ids": [5], "speed": [42.0]}, level="track", key="track_id", overwrite=True)
        assert md._store.frame("track")["speed"].to_list() == [42.0, 0.9]

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

    def test_a_key_column_that_names_several_rows_is_rejected(self):
        """``instance_index`` is dense within a *frame*, so it repeats across a sequence.

        Left unchecked this was a silent wrong answer rather than an error: the match is on
        ``(item_index, key)`` against the frame, so one incoming value landed on every row
        sharing the pair.
        """
        md = _tracking([[[5, 9], [], [9, -1]]])
        assert md._store.frame("instance")["instance_index"].to_list() == [0, 1, 0, 1], "repeats"
        with pytest.raises(ValueError, match="does not name one row"):
            md.add_factors(
                {"item_index": [0, 0], "instance_index": [0, 1], "x": [1.0, 2.0]},
                level="instance",
                key="instance_index",
            )

    def test_the_rejection_names_a_column_that_would_work(self):
        md = _tracking([[[5, 9], [], [9, -1]]])
        with pytest.raises(ValueError, match="target_index"):
            md.add_factors(
                {"item_index": [0], "instance_index": [0], "x": [1.0]},
                level="instance",
                key="instance_index",
            )

    def test_the_suggestion_skips_columns_that_cannot_be_a_key(self):
        """A key names a row by one value, so a box could never be one.

        Asking polars for the distinct count of an Array column is unsupported on the
        supported floor and comes back as a panic out of its Rust side, which replaced this
        whole message with a stack trace.
        """
        md = _tracking([[[5, 9], [], [9, -1]]])
        assert md._store.frame("instance")["box"].dtype == pl.Array(pl.Float32, 4)

        with pytest.raises(ValueError, match="does not name one row") as raised:
            md.add_factors(
                {"item_index": [0], "instance_index": [0], "x": [1.0]},
                level="instance",
                key="instance_index",
            )
        assert "box" not in str(raised.value)

    def test_a_nested_column_is_refused_as_the_key_itself(self):
        """The same dtype on the other path: naming it outright, rather than being offered it."""
        md = _tracking([[[5, 9], [], [9, -1]]])
        with pytest.raises(ValueError, match=r"key='box' is a .* column .* holds several values per row"):
            md.add_factors({"item_index": [0], "x": [1.0]}, level="instance", key="box")

    def test_the_column_it_names_does_work(self):
        """The suggestion is computed from the frame, so it has to be usable."""
        md = _tracking([[[5, 9], [], [9, -1]]])
        md.add_factors(
            {"item_index": [0, 0, 0, 0], "target_index": [3, 1, 0, 2], "x": [0.3, 0.1, 0.0, 0.2]},
            level="instance",
            key="target_index",
        )
        assert md._store.frame("instance")["x"].to_list() == [0.0, 0.1, 0.2, 0.3]

    def test_nothing_is_written_when_the_key_column_is_ambiguous(self):
        md = _tracking([[[5, 9], [], [9, -1]]])
        before = set(md._store.columns)
        with pytest.raises(ValueError, match="does not name one row"):
            md.add_factors(
                {"item_index": [0], "instance_index": [0], "x": [1.0]},
                level="instance",
                key="instance_index",
            )
        assert set(md._store.columns) == before

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
