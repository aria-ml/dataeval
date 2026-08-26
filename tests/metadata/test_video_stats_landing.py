"""Where video statistics land, and how they say which row they belong to — FE-9 issue 9.2.

There is no MOT statistics path today: ``compute_stats`` treats a dataset item as one
image, and a video item is a sequence of frames, so nothing produces a ``StatsResult`` for
video. That producer is the statistics workstream's. What metadata owns is the other end —
the landing contract — and these tests stand in for the producer so that contract is pinned
before it exists rather than retrofitted afterwards.

Issue 9.1 decided that contract: video statistics name their rows with **key columns**,
not with :class:`~dataeval.types.SourceIndex`. A producer emits one array per level plus
the columns naming the rows — ``item_index`` alongside ``unit_index`` or ``track_id`` — and
each level lands in one ``add_factors`` call. ``SourceIndex`` is ``(item, target, channel)``
and can name a sequence and a detection; it cannot name a frame or a track, and the diamond
has no single ordering for it to be.

Every fixture here uses keys whose order **disagrees** with row order. A fixture where the
two coincide passes positionally and proves nothing.
"""

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from tests.metadata.test_structurers import _mot_dataset

# Two videos exercising the shapes a classification dataset never produces.
#
#   seq 0, frame 0: tracks 5 and 9
#   seq 0, frame 1: no detections at all
#   seq 0, frame 2: track 9 again — a gap — plus an untracked detection
#   seq 1, frame 0: track 7
#   seq 1, frame 1: track 7 and track 3
#
# Track ids ascend nowhere: first-appearance order is (5, 9) then (7, 3), so sorted-id
# order disagrees with row order in the second sequence and matches in the first.
_SHAPES = [[[5, 9], [], [9, -1]], [[7], [7, 3]]]


def _video():
    """A structured tracking metadata over ``_SHAPES``, with no factors attached yet."""
    metadata = Metadata(_mot_dataset(_SHAPES))
    metadata._structure()
    return metadata


def _per_frame():
    """Per-frame values, handed over in an order no level's rows are in.

    Frames arrive last-first within the first sequence and reversed within the second, so
    a positional write would scramble every one of them.
    """
    return {
        "item_index": [0, 0, 0, 1, 1],
        "unit_index": [2, 0, 1, 1, 0],
        "blur": [0.2, 0.0, 0.1, 1.1, 1.0],
    }


def _per_track():
    """Per-track values in sorted-id order within each sequence, as ``track_stats`` emits."""
    return {
        "item_index": [0, 0, 1, 1],
        "track_ids": [5, 9, 3, 7],
        "speed": [0.5, 0.9, 1.3, 1.7],
    }


def _measured(md):
    """Attach all four levels, each in the one call its level takes."""
    md.add_factors(_per_frame(), level="unit", key="unit_index")
    md.add_factors(_per_track(), level="track", key="track_id")
    md.add_factors({"night": [0.0, 1.0]}, level="sequence")
    md.add_factors({"iou": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}, level="instance")
    return md


@pytest.mark.required
class TestTheFixtureIsAdversarial:
    """If these drift, every assertion below silently stops proving anything."""

    def test_the_dataset_has_the_shapes_a_video_produces(self):
        md = _video()
        assert md.level_counts == {"sequence": 2, "unit": 5, "track": 4, "instance": 7}

    def test_key_order_disagrees_with_row_order(self):
        md = _video()
        rows = md._store.frame("track").select("item_index", "track_id").rows()
        assert rows == [(0, 5), (0, 9), (1, 7), (1, 3)], "first appearance, not sorted id"

        keys = list(zip(_per_track()["item_index"], _per_track()["track_ids"], strict=True))
        assert keys != rows, "a fixture whose orders coincide passes positionally"
        assert sorted(keys) == sorted(rows), "same rows, different order"

    def test_frame_keys_disagree_with_row_order_too(self):
        md = _video()
        rows = md._store.frame("unit").select("item_index", "unit_index").rows()
        keys = list(zip(_per_frame()["item_index"], _per_frame()["unit_index"], strict=True))
        assert keys != rows
        assert sorted(keys) == sorted(rows)


@pytest.mark.required
class TestEveryLevelReachableInOneCall:
    """One ``add_factors`` per level, nothing rearranged by the caller beforehand."""

    def test_per_frame_values_land_by_key(self):
        md = _video()
        md.add_factors(_per_frame(), level="unit", key="unit_index")
        assert md._store.frame("unit")["blur"].to_list() == [0.0, 0.1, 0.2, 1.0, 1.1]

    def test_per_track_values_land_by_key(self):
        md = _video()
        md.add_factors(_per_track(), level="track", key="track_id")
        assert md._store.frame("track")["speed"].to_list() == [0.5, 0.9, 1.7, 1.3]

    def test_per_sequence_values_land_positionally(self):
        md = _video()
        md.add_factors({"night": [0.0, 1.0]}, level="sequence")
        assert md._store.frame("sequence")["night"].to_list() == [0.0, 1.0]

    def test_per_detection_values_land_positionally(self):
        md = _video()
        md.add_factors({"iou": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}, level="instance")
        assert md._store.frame("instance")["iou"].to_list() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    def test_all_four_coexist(self):
        """Four calls, four levels, no interference between them."""
        md = _measured(_video())
        assert md._store.frame("unit")["blur"].to_list() == [0.0, 0.1, 0.2, 1.0, 1.1]
        assert md._store.frame("track")["speed"].to_list() == [0.5, 0.9, 1.7, 1.3]
        assert md._store.frame("sequence")["night"].to_list() == [0.0, 1.0]
        assert md._store.frame("instance")["iou"].to_list() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    def test_the_key_column_is_not_stored_as_a_measurement(self):
        """``unit_index`` says which frame; it is not a fact about the frame."""
        md = _video()
        md.add_factors(_per_frame(), level="unit", key="unit_index")
        assert md._store.frame("unit")["unit_index"].to_list() == [0, 1, 2, 0, 1], "untouched"
        assert "metadata_unit_index" not in md._store.columns

    def test_a_producer_needs_no_source_index(self):
        """The whole point of 9.1: nothing here labels a value with a position."""
        md = _measured(_video())
        assert md._store.frame("unit")["blur"].to_list() == [0.0, 0.1, 0.2, 1.0, 1.1]


@pytest.mark.required
class TestShapesOnlyVideoProduces:
    def test_a_frame_holding_no_detections_still_takes_a_value(self):
        """Frame 1 of sequence 0 has nothing in it. It is still a frame, and still measurable."""
        md = _video()
        md.add_factors(_per_frame(), level="unit", key="unit_index")
        empty = md._store.frame("unit").filter((pl.col("item_index") == 0) & (pl.col("unit_index") == 1))
        assert empty["blur"].to_list() == [0.1]

    def test_nothing_beneath_an_empty_frame_reads_its_value(self):
        """No instance row descends from it, so the value simply has no reader below."""
        md = _video()
        md.add_factors(_per_frame(), level="unit", key="unit_index")
        # Instances of sequence 0 sit in frames 0, 0, 2, 2 — never frame 1.
        seq0 = md._store.frame("instance").filter(pl.col("item_index") == 0)
        assert seq0["unit_index"].to_list() == [0, 0, 2, 2]
        assert md._store.column("instance", "blur").to_list() == [0.0, 0.0, 0.2, 0.2, 1.0, 1.1, 1.1]

    def test_a_track_spanning_a_gap_is_one_row(self):
        """Track 9 is in frames 0 and 2 but not 1: one track row, measured once."""
        md = _video()
        md.add_factors(_per_track(), level="track", key="track_id")
        track_9 = md._store.frame("track").filter((pl.col("item_index") == 0) & (pl.col("track_id") == 9))
        assert len(track_9) == 1
        assert track_9["speed"].to_list() == [0.9]
        assert track_9["track_length"].to_list() == [2], "two observations"
        assert track_9["frame_span"].to_list() == [3], "across three frames — the gap"

    def test_an_untracked_detection_holds_no_per_track_value(self):
        """``track_id == -1`` sits in a frame and belongs to no track, so speed is null there."""
        md = _measured(_video())
        instances = md._store.frame("instance")
        untracked = np.asarray(instances["track_id"].to_list()) == -1
        assert untracked.tolist() == [False, False, False, True, False, False, False]

        speeds = md._store.column("instance", "speed").to_list()
        assert speeds == [0.5, 0.9, 0.9, None, 1.7, 1.7, 1.3]

    def test_an_untracked_detection_still_holds_its_frames_value(self):
        """It has no track parent; it does have a frame parent, and reads from it."""
        md = _measured(_video())
        assert md._store.column("instance", "blur").to_list()[3] == 0.2

    def test_a_partly_null_per_track_factor_leaves_instance_analysis(self):
        """Documented consequence of the diamond: a partial column cannot be binned."""
        md = _measured(_video())
        assert "speed" in md.at("track").factor_names
        assert "speed" not in md.at("instance").factor_names, "null on the untracked detection"

    def test_a_sequence_whose_frames_are_all_empty(self):
        """Frames exist, tracks and detections do not. Per-frame values still land."""
        md = Metadata(_mot_dataset([[[], []], [[7]]]))
        md._structure()
        assert md.level_counts == {"sequence": 2, "unit": 3, "track": 1, "instance": 1}

        md.add_factors(
            {"item_index": [0, 0, 1], "unit_index": [1, 0, 0], "blur": [0.1, 0.0, 1.0]},
            level="unit",
            key="unit_index",
        )
        assert md._store.frame("unit")["blur"].to_list() == [0.0, 0.1, 1.0]

    def test_an_empty_sequence_aggregates_to_null_not_zero(self):
        """Nothing was measured there, which is not the same as measuring zero."""
        md = Metadata(_mot_dataset([[[], []], [[7]]]))
        md._structure()
        counted = md.agg("instance", "sequence", pl.len().alias("n_detections"))
        assert counted._store.frame("sequence")["n_detections"].to_list() == [None, 1]

    def test_siblings_do_not_see_each_others_values(self):
        """A track spans frames, so no single frame's blur is the right value for it."""
        md = _measured(_video())
        assert md._store.column("track", "blur").to_list() == [None] * 4
        assert "blur" not in md.at("track").factor_names
        assert "speed" not in md.at("unit").factor_names


@pytest.mark.required
class TestPartialAndRepeatedWrites:
    """A producer that measures some frames, or one sequence at a time."""

    def test_frames_no_key_names_are_null_rather_than_absent(self):
        """Sampling frames is expected; the column still has one value per frame."""
        md = _video()
        md.add_factors(
            {"item_index": [0, 0], "unit_index": [2, 0], "blur": [0.2, 0.0]},
            level="unit",
            key="unit_index",
        )
        held = md._store.frame("unit")["blur"]
        assert held.to_list() == [0.0, None, 0.2, None, None]
        assert len(held) == md.level_counts["unit"]

    def test_one_sequence_at_a_time_builds_one_column(self):
        md = _video()
        md.add_factors(
            {"item_index": [0, 0, 0], "unit_index": [2, 0, 1], "blur": [0.2, 0.0, 0.1]},
            level="unit",
            key="unit_index",
        )
        md.add_factors(
            {"item_index": [1, 1], "unit_index": [1, 0], "blur": [1.1, 1.0]},
            level="unit",
            key="unit_index",
        )
        assert md._store.frame("unit")["blur"].to_list() == [0.0, 0.1, 0.2, 1.0, 1.1]
        assert "blur_added" not in md._store.columns

    def test_the_same_holds_for_tracks(self):
        md = _video()
        md.add_factors(
            {"item_index": [0, 0], "track_ids": [9, 5], "speed": [0.9, 0.5]},
            level="track",
            key="track_id",
        )
        md.add_factors(
            {"item_index": [1, 1], "track_ids": [3, 7], "speed": [1.3, 1.7]},
            level="track",
            key="track_id",
        )
        assert md._store.frame("track")["speed"].to_list() == [0.5, 0.9, 1.7, 1.3]
        assert "speed_added" not in md._store.columns


@pytest.mark.required
class TestTheRoundTripSurvivesTheOperations:
    """Landed values are ordinary factors: ``where``, ``having`` and ``agg`` all read them."""

    def test_where_narrows_on_a_per_frame_value(self):
        md = _measured(_video())
        kept = md.where(pl.col("blur") > 0.05, level="unit")
        assert kept._store.frame("unit")["blur"].to_list() == [0.1, 0.2, 1.0, 1.1]
        assert kept.level_counts["sequence"] == 2, "where never filters upwards"
        assert kept.level_counts["track"] == 4, "tracks are siblings of frames, not below them"

    def test_having_narrows_sequences_on_a_per_track_value(self):
        md = _measured(_video())
        kept = md.having(pl.col("speed") > 1.5, level="track")
        assert kept._store.frame("sequence")["item_index"].to_list() == [1]
        assert kept.level_counts == {"sequence": 1, "unit": 2, "track": 2, "instance": 3}

    def test_having_keeps_the_whole_sequence_not_only_the_matching_track(self):
        md = _measured(_video())
        kept = md.having(pl.col("speed") > 1.5, level="track")
        assert kept._store.frame("track")["speed"].to_list() == [1.7, 1.3], "track 3 rides along"

    def test_agg_rolls_frames_up_into_sequences(self):
        md = _measured(_video())
        rolled = md.agg("unit", "sequence", pl.mean("blur").alias("mean_blur"))
        assert rolled._store.frame("sequence")["mean_blur"].to_list() == pytest.approx([0.1, 1.05])

    def test_agg_rolls_detections_up_into_tracks(self):
        md = _measured(_video())
        rolled = md.agg("instance", "track", pl.len().alias("n_obs"))
        assert rolled._store.frame("track")["n_obs"].to_list() == [1, 2, 2, 1]

    def test_agg_over_a_sibling_column_needs_unique_by(self):
        """Averaging per-frame blur over a track's detections weights frames by crowding."""
        md = _measured(_video())
        with pytest.raises(ValueError, match="would weight it by that fan-out"):
            md.agg("instance", "track", pl.col("blur").mean().alias("mean_blur"))

        rolled = md.agg("instance", "track", pl.col("blur").mean().alias("mean_blur"), unique_by="unit")
        assert rolled._store.frame("track")["mean_blur"].to_list() == pytest.approx([0.0, 0.1, 1.05, 1.1])

    def test_values_survive_a_narrowing_and_a_rollup_together(self):
        md = _measured(_video())
        rolled = md.having(pl.col("speed") > 1.5, level="track").agg(
            "unit", "sequence", pl.mean("blur").alias("mean_blur")
        )
        assert rolled._store.frame("sequence")["mean_blur"].to_list() == pytest.approx([1.05])


@pytest.mark.required
class TestTheProducerShouldNotRecomputeThese:
    """Two per-track quantities the structuring walk already derives — FE-9 issue 9.3.

    ``track_length == n_appearances`` and ``frame_span == track_duration``. A video
    statistic duplicating one of these is duplicating it, not adding to it.
    """

    def test_both_are_present_before_any_factor_is_attached(self):
        md = _video()
        assert {"track_length", "frame_span"} <= set(md._store.frame("track").columns)

    def test_they_differ_exactly_where_a_track_has_a_gap(self):
        md = _video()
        frame = md._store.frame("track")
        assert frame.select("item_index", "track_id").rows() == [(0, 5), (0, 9), (1, 7), (1, 3)]
        assert frame["track_length"].to_list() == [1, 2, 2, 1]
        assert frame["frame_span"].to_list() == [1, 3, 2, 1], "track 9 spans the gap"

    def test_aggregating_detections_recomputes_track_length_a_third_way(self):
        """Correct, and not the route to prefer: the walk already counted it."""
        md = _video()
        rolled = md.agg("instance", "track", pl.len().alias("n_obs"))
        frame = rolled._store.frame("track")
        assert frame["n_obs"].to_list() == frame["track_length"].to_list()
