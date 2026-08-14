"""Positional foreign keys: both representations must mean exactly the same thing.

:class:`~dataeval._metadata._links.LinkIndex` picks between a run-length and a general form by
inspecting the positions, so the two are interchangeable by construction — every test
here that fixes behavior is run against both, and the reference is the clamp-then-null
semantics propagation has always had.
"""

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._metadata._links import GatherLink, LinkIndex, RunLengthLink, to_series
from dataeval._metadata._structurers import select_structurer
from tests.embeddings.test_embeddings import MockDataset
from tests.metadata.test_structurers import _mot_dataset, _od_target

# (label, positions, parent_len) — grouped and ascending selects run-length, anything
# else selects the general form.
CASES = [
    ("grouped dense", [0, 0, 1, 1, 1, 2], 3),
    ("grouped, parent with no children", [0, 0, 2, 2, 2], 3),
    ("grouped, trailing empty parents", [0, 0], 4),
    ("unsorted", [2, 0, 1, 0], 3),
    ("missing ancestors", [0, -1, 1, -1, 2], 3),
    ("all missing", [-1, -1, -1], 3),
    ("no child rows", [], 3),
    ("no rows at either level", [], 0),
]


def _expected(positions, values):
    """The reference: a negative position yields null, anything else gathers."""
    return [None if position < 0 else values[position] for position in positions]


@pytest.mark.required
@pytest.mark.parametrize(("label", "positions", "parent_len"), CASES)
class TestBothRepresentationsAgree:
    def test_broadcast_matches_clamp_then_null(self, label, positions, parent_len):
        values = np.arange(parent_len, dtype=np.int64) * 10
        link = LinkIndex.of(positions, parent_len)
        assert link.broadcast("x", values).to_list() == _expected(positions, values.tolist())

    def test_positions_round_trip(self, label, positions, parent_len):
        link = LinkIndex.of(positions, parent_len)
        assert link.positions().tolist() == list(positions)

    def test_counts_are_children_per_parent(self, label, positions, parent_len):
        link = LinkIndex.of(positions, parent_len)
        present = [position for position in positions if position >= 0]
        assert link.counts().tolist() == np.bincount(present, minlength=parent_len).tolist()
        assert len(link.counts()) == parent_len

    def test_lengths_are_reported(self, label, positions, parent_len):
        link = LinkIndex.of(positions, parent_len)
        assert link.child_len == len(positions)
        assert link.parent_len == parent_len


@pytest.mark.required
class TestRepresentationChoice:
    """Picked by inspection, because a caller's declaration could be wrong silently."""

    def test_grouped_ascending_and_total_is_run_length(self):
        assert isinstance(LinkIndex.of([0, 0, 1, 2, 2], 3), RunLengthLink)

    def test_unsorted_is_a_gather(self):
        assert isinstance(LinkIndex.of([1, 0, 2], 3), GatherLink)

    def test_any_missing_ancestor_forces_a_gather(self):
        # A run-length form has no way to say "this child has no parent" — the runs
        # tile the child rows exhaustively.
        assert isinstance(LinkIndex.of([0, -1, 1], 2), GatherLink)

    def test_run_length_stores_one_number_per_parent(self):
        """The memory claim: size follows the parent level, not the child level."""
        link = LinkIndex.of(np.repeat(np.arange(1000), 50), 1000)
        assert isinstance(link, RunLengthLink)
        assert link._offsets.size == 1001
        assert link.child_len == 50_000


@pytest.mark.required
class TestBroadcastDtypes:
    """Nested dtypes are the case a scatter-based null cannot serve."""

    @pytest.mark.parametrize("positions", [[0, 0, 1], [0, -1, 1]])
    def test_fixed_width_arrays_survive(self, positions):
        box = np.arange(8, dtype=np.float32).reshape(2, 4)
        result = LinkIndex.of(positions, 2).broadcast("box", box)
        assert result.to_list() == _expected(positions, box.tolist())

    @pytest.mark.parametrize("positions", [[0, 0, 1], [0, -1, 1]])
    def test_strings_survive(self, positions):
        values = np.array(["a", "b"])
        result = LinkIndex.of(positions, 2).broadcast("s", values)
        assert result.to_list() == _expected(positions, values.tolist())

    def test_a_polars_series_is_accepted(self):
        result = LinkIndex.of([0, 0, 1], 2).broadcast("x", pl.Series([1.5, 2.5]))
        assert result.to_list() == [1.5, 1.5, 2.5]

    def test_the_result_takes_the_requested_name(self):
        assert LinkIndex.of([0, -1], 1).broadcast("chosen", np.array([1])).name == "chosen"


@pytest.mark.required
class TestRestrict:
    def test_dropping_children_keeps_the_survivors_pointing_at_the_same_parents(self):
        link = LinkIndex.of([0, 0, 1, 1, 2], 3)
        restricted = link.restrict(np.array([1, 4], dtype=np.intp), np.arange(3, dtype=np.intp))
        assert restricted.positions().tolist() == [0, 2]

    def test_dropping_a_parent_renumbers_the_rest(self):
        link = LinkIndex.of([0, 1, 2], 3)
        # Parent 1 is dropped, so its child goes too — as the survivor rule requires.
        restricted = link.restrict(np.array([0, 2], dtype=np.intp), np.array([0, -1, 1], dtype=np.intp))
        assert restricted.positions().tolist() == [0, 1]
        assert restricted.parent_len == 2

    def test_an_existing_missing_ancestor_is_carried_through(self):
        link = LinkIndex.of([0, -1, 1], 2)
        restricted = link.restrict(np.arange(3, dtype=np.intp), np.arange(2, dtype=np.intp))
        assert restricted.positions().tolist() == [0, -1, 1]

    def test_keeping_a_child_whose_parent_was_dropped_is_rejected(self):
        """I5: a filter must never fabricate a no-ancestor marker."""
        link = LinkIndex.of([0, 1, 2], 3)
        with pytest.raises(RuntimeError, match="fabricate a no-ancestor marker"):
            link.restrict(np.arange(3, dtype=np.intp), np.array([0, -1, 1], dtype=np.intp))

    def test_restricting_onto_a_parent_level_with_no_survivors(self):
        """Nothing to renumber onto, and every child already carries the marker."""
        link = LinkIndex.of([-1, -1], 0)
        restricted = link.restrict(np.array([0, 1], dtype=np.intp), np.array([], dtype=np.intp))
        assert restricted.positions().tolist() == [-1, -1]
        assert restricted.parent_len == 0

    def test_restriction_can_change_representation(self):
        """Filtering an unsorted edge down to an ordered one is free to tighten."""
        link = LinkIndex.of([1, 0, 2], 3)
        assert isinstance(link, GatherLink)
        restricted = link.restrict(np.array([1, 2], dtype=np.intp), np.arange(3, dtype=np.intp))
        assert isinstance(restricted, RunLengthLink)
        assert restricted.positions().tolist() == [0, 2]


@pytest.mark.required
class TestCompose:
    """A child's grandparent is its parent's parent, so only edges need storing."""

    def test_two_edges_chain(self):
        child_parent = LinkIndex.of([0, 0, 1, 1, 2], 3)
        parent_grand = LinkIndex.of([0, 0, 1], 2)
        assert child_parent.compose(parent_grand).positions().tolist() == [0, 0, 0, 0, 1]

    def test_a_missing_parent_means_a_missing_grandparent(self):
        # Carried through rather than looked up: parent_grand[-1] would read the last row.
        child_parent = LinkIndex.of([0, -1, 1], 2)
        parent_grand = LinkIndex.of([0, 1], 2)
        assert child_parent.compose(parent_grand).positions().tolist() == [0, -1, 1]

    def test_edges_that_do_not_meet_are_rejected(self):
        with pytest.raises(ValueError, match="do not meet at the same level"):
            LinkIndex.of([0, 1, 2], 3).compose(LinkIndex.of([0, 1], 2))

    def test_composition_reports_the_far_level(self):
        composed = LinkIndex.of([0, 0, 1], 2).compose(LinkIndex.of([0, 0], 1))
        assert composed.child_len == 3
        assert composed.parent_len == 1

    def test_an_empty_level_in_between_leaves_every_row_marked(self):
        """A tracking dataset in which nothing is tracked has no ``track`` rows at all.

        Every detection's track position is then the marker, and there is nothing above
        to look it up in — the clamped index would run off an empty array.
        """
        no_track = LinkIndex.of([-1, -1, -1], 0)
        track_to_sequence = LinkIndex.of([], 2)
        assert no_track.compose(track_to_sequence).positions().tolist() == [-1, -1, -1]


@pytest.mark.required
class TestPositionsMustFitTheParentLevel:
    """Both forms report the parent length they were given, or refuse to be built.

    A run-length form derives its parent count from the runs, so an out-of-range position
    would silently extend the parent level while the general form kept the declared
    length. Interchangeable means agreeing about this too.
    """

    @pytest.mark.parametrize("positions", [[0, 1, 5], [5, 1, 0], [0, 3]])
    def test_a_position_past_the_parent_level_is_rejected(self, positions):
        with pytest.raises(ValueError, match="names a row beyond the 3 row"):
            LinkIndex.of(positions, 3)

    def test_a_parent_level_with_no_rows_admits_only_markers(self):
        assert LinkIndex.of([-1, -1], 0).parent_len == 0
        with pytest.raises(ValueError, match="names a row beyond the 0 row"):
            LinkIndex.of([0], 0)


@pytest.mark.required
class TestFirstKnown:
    """Only a diamond has two routes, and they differ only where one stops short."""

    @staticmethod
    def _diamond():
        """Two sequences: seq 0 holds frames 0-1 and track 0, seq 1 holds frame 2 and track 1.

        Five detections; detection 2 is untracked, so it reaches its sequence through
        its frame and not through a track.
        """
        return (
            LinkIndex.of([0, 0, 1, 1, 2], 3).compose(LinkIndex.of([0, 0, 1], 2)),  # via unit
            LinkIndex.of([0, 0, -1, 0, 1], 2).compose(LinkIndex.of([0, 1], 2)),  # via track
        )

    def test_routes_agree_wherever_both_are_total(self):
        via_unit, via_track = self._diamond()
        both = via_track.positions() >= 0
        assert np.array_equal(via_unit.positions()[both], via_track.positions()[both])

    def test_the_first_route_that_knows_wins(self):
        via_unit, via_track = self._diamond()
        assert LinkIndex.first_known([via_unit, via_track]).positions().tolist() == [0, 0, 0, 0, 1]

    def test_a_later_route_fills_what_an_earlier_one_lacks(self):
        via_unit, via_track = self._diamond()
        assert LinkIndex.first_known([via_track, via_unit]).positions().tolist() == [0, 0, 0, 0, 1]

    def test_a_single_route_is_returned_as_is(self):
        only = LinkIndex.of([0, -1, 1], 2)
        assert LinkIndex.first_known([only]).positions().tolist() == [0, -1, 1]

    def test_no_routes_is_rejected(self):
        with pytest.raises(ValueError, match="empty set of routes"):
            LinkIndex.first_known([])

    def test_routes_between_different_levels_are_rejected(self):
        with pytest.raises(ValueError, match="between the same two levels"):
            LinkIndex.first_known([LinkIndex.of([0, 1], 2), LinkIndex.of([0, 1, 1], 2)])


@pytest.mark.required
class TestCompositionMatchesStoredAncestry:
    """Storing only the schema's edges must lose nothing.

    Today every block records a position array for *every* ancestor. Keeping only the
    parent edges and composing the rest is the same information in less space — but only
    if the composition agrees with what was stored, including across the diamond, where
    two routes exist and one of them stops short at an untracked detection.
    """

    @staticmethod
    def _composed(data, schema):
        """Rebuild every ancestor link from the parent edges alone."""
        blocks = {block.level: block for block in data.blocks}
        sizes = {block.level: block.size for block in data.blocks}
        edges = {
            (level, parent): LinkIndex.of(blocks[level].ancestor_pos[parent], sizes[parent])
            for level in schema.levels
            for parent in schema.parents_of(level)
        }
        built = {}
        for level in schema.levels:
            for ancestor in schema.ancestors(level):
                routes = []
                for path in schema.paths(level, ancestor):
                    link, current = None, level
                    for step in path:
                        edge = edges[(current, step)]
                        link = edge if link is None else link.compose(edge)
                        current = step
                    routes.append(link)
                built[(level, ancestor)] = LinkIndex.first_known(routes)
        return built

    def _check(self, dataset):
        structurer = select_structurer(dataset, None)
        data = structurer.build(dataset)
        blocks = {block.level: block for block in data.blocks}
        composed = self._composed(data, structurer.levels)
        assert composed, "no ancestor links to check"
        for (level, ancestor), link in composed.items():
            stored = np.asarray(blocks[level].ancestor_pos[ancestor], dtype=np.intp)
            assert link.positions().tolist() == stored.tolist(), f"{level} -> {ancestor}"

    def test_object_detection(self, get_od_dataset):
        self._check(get_od_dataset(8, targets_per_image=3, metadata=[{"w": "a"}] * 8))

    def test_image_classification(self, get_od_dataset):
        self._check(get_od_dataset(8, metadata=[{"w": "a"}] * 8))

    def test_tracking_with_untracked_detections(self):
        """The diamond's two routes, where the track branch cannot reach the sequence."""
        self._check(_mot_dataset([[2, 0, [1, -1]], [[0, -1, 2]]], [{"w": "a"}] * 2))

    def test_tracking_with_every_detection_tracked(self):
        self._check(_mot_dataset([[2, 1], [1, 3]], [{"w": "a"}] * 2))


@pytest.mark.required
class TestMetadataStore:
    """The normalized store must carry everything the flat frame does.

    The store holds each level's own rows and the schema's own edges; the flat frame
    is those plus the gathers that carry ancestor factors downwards. Until that
    equivalence holds, nothing can safely read from the store instead.
    """

    @staticmethod
    def _metadata(dataset):
        metadata = Metadata(dataset)
        metadata._structure()
        return metadata

    def _check(self, dataset):
        metadata = self._metadata(dataset)
        flat = metadata.dataframe
        for level in metadata.levels:
            rows = flat.filter(pl.col("level") == level)
            native = metadata._store.frame(level)
            assert native.height == rows.height, level
            # A level's own columns are stored verbatim.
            for column in native.columns:
                assert native[column].to_list() == rows[column].to_list(), f"{level}.{column}"
            # An ancestor's factors reproduce by broadcast. Reserved key columns are
            # deliberately excluded: a block writes only its own level's keys, so they
            # are null above rather than inherited, and broadcasting them would differ.
            for ancestor in metadata._levels.ancestors(level):
                link = metadata._store.link(level, ancestor)
                for column in metadata._factors_by_level[ancestor]:
                    broadcast = link.broadcast(column, metadata._store.frame(ancestor)[column])
                    assert broadcast.to_list() == rows[column].to_list(), f"{ancestor}->{level}.{column}"

    def test_image_classification(self, get_od_dataset):
        self._check(get_od_dataset(6, metadata=[{"w": "a", "n": 1.0}] * 6))

    def test_object_detection(self, get_od_dataset):
        self._check(get_od_dataset(6, targets_per_image=3, metadata=[{"w": "a", "n": 1.0}] * 6))

    def test_tracking_with_untracked_detections(self):
        self._check(_mot_dataset([[2, 0, [1, -1]], [[0, 2], [1]]], [{"w": "a", "n": 1.0}] * 2))

    def test_only_schema_edges_are_stored(self):
        """Ancestor pairs beyond the direct edges are composed, not held."""
        metadata = self._metadata(_mot_dataset([[2, 1], [1]], [{"w": "a"}] * 2))
        assert set(metadata._store.links) == {
            ("unit", "sequence"),
            ("track", "sequence"),
            ("instance", "unit"),
            ("instance", "track"),
        }
        assert ("instance", "sequence") not in metadata._store.links
        assert metadata._store.link("instance", "sequence") is not None

    def test_a_dataset_where_nothing_is_tracked_still_composes(self):
        """No ``track`` rows at all, so the track branch of the diamond reaches nothing."""
        metadata = self._metadata(_mot_dataset([[[-1, -1]], [[-1]]], [{"w": "a"}] * 2))
        assert metadata._store.links[("instance", "track")].parent_len == 0
        assert metadata._store.link("instance", "sequence").positions().tolist() == [0, 0, 1]

    def test_a_factor_added_later_lands_in_its_level_frame(self):
        """The store holds every fact once, including the ones added after structuring."""
        metadata = self._metadata(_mot_dataset([[2, 1], [1]], [{"w": "a"}] * 2))
        metadata.add_factors({"brightness": np.arange(3, dtype=np.float64)}, level="unit")
        assert "brightness" in metadata._factors_by_level["unit"]
        assert metadata._store.frame("unit")["brightness"].to_list() == [0.0, 1.0, 2.0]

    def test_composed_links_are_memoized(self):
        metadata = self._metadata(_mot_dataset([[2, 1], [1]], [{"w": "a"}] * 2))
        first = metadata._store.link("instance", "sequence")
        assert metadata._store.link("instance", "sequence") is first

    def test_a_level_that_is_not_above_is_rejected(self):
        metadata = self._metadata(_mot_dataset([[2, 1], [1]], [{"w": "a"}] * 2))
        with pytest.raises(ValueError, match="is not above"):
            metadata._store.link("unit", "track")

    def test_a_view_is_independent_of_its_source(self, get_od_dataset):
        """A write on either side must be invisible to the other.

        Stated as behavior and without naming a field, on purpose. ``at()`` used to
        hand-copy every mutable container the store was made of, and the next such
        container added would have been missed silently. The store is immutable and a
        writer rebinds it, so there is no list to keep complete — and this test would
        fail if one came back.
        """
        metadata = self._metadata(get_od_dataset(4, targets_per_image=2, metadata=[{"w": "a"}] * 4))
        view = metadata.at("unit")

        metadata.add_factors({"only_on_source": np.arange(4, dtype=np.float64)}, level="unit")
        assert "only_on_source" not in view.dataframe.columns
        assert "only_on_source" not in view._store.columns

        view.add_factors({"only_on_view": np.arange(4, dtype=np.float64)}, level="unit")
        assert "only_on_view" not in metadata.dataframe.columns
        assert "only_on_view" not in metadata._store.columns


@pytest.mark.required
class TestToSeries:
    """A 2-D column becomes a fixed-width Array, including when it has no rows.

    The no-rows case is the one that needed stating: on the supported polars floor
    ``pl.Series(name, np.empty((0, 4)))`` raises rather than inferring an empty
    ``Array(_, 4)``, and an object-detection dataset whose items carry no detections
    produces precisely that array for its ``box`` column.
    """

    def test_a_populated_2d_column_is_fixed_width(self):
        series = to_series("box", np.zeros((2, 4), dtype=np.float32))
        assert series.dtype == pl.Array(pl.Float32, 4)
        assert len(series) == 2

    def test_an_empty_2d_column_keeps_its_width_and_inner_dtype(self):
        series = to_series("box", np.zeros((0, 4), dtype=np.float32))
        assert series.dtype == pl.Array(pl.Float32, 4)
        assert len(series) == 0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
    def test_the_inner_dtype_follows_the_array(self, dtype):
        empty = to_series("x", np.zeros((0, 3), dtype=dtype))
        populated = to_series("x", np.zeros((2, 3), dtype=dtype))
        assert empty.dtype == populated.dtype

    def test_one_dimensional_and_non_array_values_are_unchanged(self):
        assert to_series("x", np.arange(3)).to_list() == [0, 1, 2]
        assert to_series("x", ["a", "b"]).to_list() == ["a", "b"]
        assert to_series("x", np.empty(0, dtype=np.float32)).dtype == pl.Float32

    def test_a_dataset_with_no_detections_structures(self):
        """The end-to-end shape that made the floor fail: an empty (0, 4) box column.

        Built from detection targets holding nothing, rather than by asking the fixture
        for zero targets per image — that produces a classification dataset, which has
        no box column to be empty.
        """
        dataset = MockDataset(np.zeros((4, 3, 16, 16)), [_od_target(0)] * 4, [{"w": "a"}] * 4)
        metadata = Metadata(dataset)
        metadata._structure()
        assert metadata._store.frame("instance").height == 0
        assert metadata.dataframe.height == 4
