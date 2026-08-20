"""Survivor closure and link remapping — FE-5 issue 5.1.

A filter's whole risk is that it renumbers rows correctly at one level and not at another,
which changes no shape and raises nothing. These tests compare the store's answer against a
brute-force oracle that never touches a :class:`LinkIndex`: it keys every row by the reserved
columns the flat frame carries and resolves ancestry by matching those keys, so the two
implementations share nothing but the dataset.
"""

import logging

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._metadata._links import RunLengthLink
from tests.metadata.test_structurers import _mot_dataset

# How to name a row of each level, and how to name the ancestor a row points at, using only
# the reserved columns the flat frame carries. ``item_index`` is the dataset item, so it
# doubles as the sequence key; the rest are positions within that item.
ROW_KEY = {
    "sequence": ("item_index",),
    "unit": ("item_index", "unit_index"),
    "track": ("item_index", "track_index"),
    "instance": ("item_index", "unit_index", "instance_index"),
}
ANCESTOR_KEY = {
    ("unit", "sequence"): ("item_index",),
    ("track", "sequence"): ("item_index",),
    ("instance", "sequence"): ("item_index",),
    ("instance", "unit"): ("item_index", "unit_index"),
    ("instance", "track"): ("item_index", "track_index"),
}


def _keys(metadata, level, columns):
    """One key tuple per row at ``level``, read off that level's own rows."""
    frame = metadata.rows_at(level)
    held = [frame[name].to_list() for name in columns]
    return [tuple(column[row] for column in held) for row in range(frame.height)]


def _ancestor_keys(metadata, level, ancestor):
    """The ancestor each row names, or None where it names none.

    ``-1`` in a key column is the no-ancestor marker — an untracked detection — and is the
    one case that must not be read as a row identity.
    """
    keys = _keys(metadata, level, ANCESTOR_KEY[(level, ancestor)])
    return [None if -1 in key else key for key in keys]


def _oracle(metadata, level, mask, mode):
    """Survivors by key matching, with no LinkIndex, no positions and no schema order."""
    schema = metadata._levels
    present = [name for name in schema if name in metadata._store.frames]
    rows = {name: _keys(metadata, name, ROW_KEY[name]) for name in present}
    ancestors = {name: schema.ancestors(name) for name in present}

    keep: dict[str, set] = {}
    if mode == "where":
        keep[level] = {key for key, flag in zip(rows[level], mask, strict=True) if flag}
        for above in ancestors[level]:
            if above in present:
                keep[above] = set(rows[above])
    else:
        for above in ancestors[level]:
            if above not in present:
                continue
            reached = _ancestor_keys(metadata, level, above)
            keep[above] = {key for key, flag in zip(reached, mask, strict=True) if flag and key is not None}

    # Fixpoint rather than one topological pass, so the oracle does not assume the ordering
    # the implementation relies on. It converges because survival only ever shrinks.
    changed = True
    while changed:
        changed = False
        for name in present:
            if name in keep and name == level and mode == "where":
                continue
            if name in keep and name in ancestors[level] and mode in {"where", "having"}:
                continue
            survivors = set()
            for row, key in enumerate(rows[name]):
                parents = [(parent, _ancestor_keys(metadata, name, parent)[row]) for parent in schema.parents_of(name)]
                if all(reached is None or reached in keep.get(parent, set()) for parent, reached in parents):
                    survivors.add(key)
            if keep.get(name) != survivors:
                keep[name] = survivors
                changed = True
    return {name: [row for row, key in enumerate(rows[name]) if key in keep[name]] for name in present}


def _invariants(store, before, keep):
    """I4 and I5, checked against the store rather than inside it."""
    for (child, parent), link in store.links.items():
        positions = link.positions()
        assert link.child_len == store.height(child), f"{child}->{parent}: child_len"
        assert link.parent_len == store.height(parent), f"{child}->{parent}: parent_len"
        if positions.size:
            # I4: in range, or the marker.
            assert positions.min() >= -1, f"{child}->{parent}: position below -1"
            assert positions.max() < max(link.parent_len, 1), f"{child}->{parent}: position past the parent"
        if child in keep and (child, parent) in before.links:
            # I5: a filter never manufactures a marker.
            was = before.links[(child, parent)].positions()[keep[child]]
            assert not np.any((was >= 0) & (positions < 0)), f"{child}->{parent}: fabricated a no-ancestor marker"


FIXTURES = {
    # A frame with no detections, a track spanning two frames, and an untracked detection.
    "diamond": [[2, 0, [1, -1]], [[0, 2], [1]]],
    # Every detection tracked, so instance->track is total.
    "dense": [[3, 3], [2]],
    # Untracked throughout: instance->track is entirely markers.
    "untracked": [[[-1, -1]], [[-1]]],
    # A sequence whose every frame is empty.
    "empty_sequence": [[0, 0], [2]],
}


@pytest.fixture(params=sorted(FIXTURES))
def tracking(request):
    metadata = Metadata(_mot_dataset(FIXTURES[request.param]))
    metadata._structure()
    return request.param, metadata


def _masks(height):
    """A spread of masks: nothing, everything, and a few partial cuts."""
    yield "none", np.zeros(height, dtype=np.bool_)
    yield "all", np.ones(height, dtype=np.bool_)
    if height:
        first = np.zeros(height, dtype=np.bool_)
        first[0] = True
        yield "first", first
        alternating = np.zeros(height, dtype=np.bool_)
        alternating[::2] = True
        yield "alternating", alternating
        last = np.zeros(height, dtype=np.bool_)
        last[-1] = True
        yield "last", last


@pytest.mark.required
class TestSurvivorClosure:
    """The closure agrees with key matching, on every level of every fixture."""

    def test_where_matches_the_oracle(self, tracking):
        name, metadata = tracking
        store = metadata._store
        for level in store.frames:
            for label, mask in _masks(store.height(level)):
                keep = store.surviving_where(level, mask)
                expected = _oracle(metadata, level, mask, "where")
                for other in expected:
                    assert keep[other].tolist() == expected[other], f"{name}: where({level},{label}) at {other}"

    def test_having_matches_the_oracle(self, tracking):
        name, metadata = tracking
        store = metadata._store
        for level in store.frames:
            if not metadata._levels.ancestors(level):
                continue
            for label, mask in _masks(store.height(level)):
                keep = store.surviving_having(level, mask)
                expected = _oracle(metadata, level, mask, "having")
                for other in expected:
                    assert keep[other].tolist() == expected[other], f"{name}: having({level},{label}) at {other}"

    def test_where_leaves_strict_ancestors_whole(self, tracking):
        """``where`` does not filter upwards, however much of the level it removes."""
        name, metadata = tracking
        store = metadata._store
        for level in store.frames:
            for _, mask in _masks(store.height(level)):
                keep = store.surviving_where(level, mask)
                for above in metadata._levels.ancestors(level):
                    if above in store.frames:
                        assert len(keep[above]) == store.height(above), f"{name}: where({level}) cut {above}"

    def test_a_row_with_no_ancestor_is_never_dropped_for_lacking_one(self, tracking):
        """I5 at the closure rather than at the edge: an untracked detection has nothing to lose."""
        name, metadata = tracking
        store = metadata._store
        if "instance" not in store.frames or "track" not in store.frames:
            pytest.skip("no tracking diamond")
        untracked = np.flatnonzero(store.positions_from("instance", "track") < 0)
        if not untracked.size:
            pytest.skip("every detection is tracked")
        everything = np.ones(store.height("unit"), dtype=np.bool_)
        keep = store.surviving_where("unit", everything)
        assert set(untracked.tolist()) <= set(keep["instance"].tolist()), name


@pytest.mark.required
class TestRestrict:
    def test_invariants_hold_after_every_filter(self, tracking):
        name, metadata = tracking
        store = metadata._store
        for level in store.frames:
            for label, mask in _masks(store.height(level)):
                keep = store.surviving_where(level, mask)
                _invariants(store.restrict(keep), store, keep)
                if metadata._levels.ancestors(level):
                    having = store.surviving_having(level, mask)
                    _invariants(store.restrict(having), store, having)
                assert label

    def test_restricted_frames_are_the_surviving_rows(self, tracking):
        name, metadata = tracking
        store = metadata._store
        for level in store.frames:
            for _, mask in _masks(store.height(level)):
                keep = store.surviving_where(level, mask)
                restricted = store.restrict(keep)
                for other, survivors in keep.items():
                    # Same guard the store uses: polars 1.0.0 panics indexing a frame that
                    # holds a fixed-width Array column with an empty index.
                    expected = store.frame(other)[survivors] if survivors.size else store.frame(other).head(0)
                    assert restricted.frame(other).equals(expected), f"{name}: {other}"

    def test_a_run_length_edge_survives_as_a_run_length_edge(self, tracking):
        """Survivors stay grouped: a parent's children die with it, so no run is orphaned."""
        name, metadata = tracking
        store = metadata._store
        run_length = [edge for edge, link in store.links.items() if isinstance(link, RunLengthLink)]
        assert run_length, f"{name}: fixture exercises no run-length edge"
        for level in store.frames:
            for _, mask in _masks(store.height(level)):
                restricted = store.restrict(store.surviving_where(level, mask))
                for edge in run_length:
                    assert isinstance(restricted.links[edge], RunLengthLink), f"{name}: {edge} degraded"

    def test_a_cascade_matches_the_single_filter_it_composes_to(self, tracking):
        """Two filters in sequence are one filter over the rows both would have kept.

        The renumbering is what makes this non-trivial: the second filter's mask indexes
        the *restricted* rows, so agreeing with the direct filter means the first restrict
        left every edge pointing where the second one reads it.
        """
        name, metadata = tracking
        height = metadata._store.height("unit")
        if height < 2:
            pytest.skip("too few rows to cascade")
        store = metadata._store

        drop_last = np.ones(height, dtype=np.bool_)
        drop_last[-1] = False
        once = store.restrict(store.surviving_where("unit", drop_last))

        drop_first = np.ones(once.height("unit"), dtype=np.bool_)
        drop_first[0] = False
        twice = once.restrict(once.surviving_where("unit", drop_first))

        both = np.ones(height, dtype=np.bool_)
        both[[0, -1]] = False
        direct = store.restrict(store.surviving_where("unit", both))

        assert dict(twice.counts) == dict(direct.counts), name
        for level in twice.frames:
            assert twice.frame(level).equals(direct.frame(level)), f"{name}: {level}"
        for edge, link in twice.links.items():
            assert link.positions().tolist() == direct.links[edge].positions().tolist(), f"{name}: {edge}"

    def test_restrict_rejects_a_partial_survivor_mapping(self, tracking):
        name, metadata = tracking
        store = metadata._store
        keep = store.surviving_where("unit", np.ones(store.height("unit"), dtype=np.bool_))
        del keep[next(iter(store.frames))]
        with pytest.raises(ValueError, match="every level that has a frame"):
            store.restrict(keep)
        assert name

    def test_a_filtered_store_still_resolves_and_flattens(self, tracking):
        """The point of remapping: the store keeps working, not merely keeps its shape."""
        name, metadata = tracking
        store = metadata._store
        mask = np.zeros(store.height("unit"), dtype=np.bool_)
        mask[::2] = True
        restricted = store.restrict(store.surviving_where("unit", mask))
        flat = restricted.flat()
        assert flat.height == sum(restricted.counts.values()), name
        assert list(flat.columns) == list(store.column_order), name
        for level in restricted.frames:
            resolved = restricted.resolve(level)
            assert resolved.height == restricted.height(level), f"{name}: {level}"
            assert resolved.equals(flat.filter(pl.col("level") == level)), f"{name}: {level}"


@pytest.mark.required
class TestWhereAndHaving:
    """The public filters — FE-5 issue 5.2."""

    def test_having_drops_a_sibling_whose_track_holds_no_match(self):
        """The acceptance case, and the one that distinguishes ``having`` from ``where``.

        One frame holds a person on one track and a car on another; a second frame
        continues only the person's track. Matching on the person keeps the frame the car
        is in — so the car's *frame* survives — but not the car's track, and a detection
        whose track was dropped goes with it.
        """
        metadata = Metadata(_mot_dataset([[[10, 20], [10]]]))
        metadata._structure()
        labels = metadata.class_labels.tolist()
        assert labels == [0, 1, 0], "fixture: person, car, person"

        kept = metadata.having(pl.col("class_label") == 0, level="instance")
        assert kept.class_labels.tolist() == [0, 0], "the car survived its own track being dropped"
        assert kept.level_counts["unit"] == 2, "both frames are kept — the car's frame holds a person"
        assert kept.level_counts["track"] == 1, "only the person's track is kept"

    def test_where_keeps_the_matching_rows_themselves(self):
        """``where`` with the same predicate keeps the person rows and every frame."""
        metadata = Metadata(_mot_dataset([[[10, 20], [10]]]))
        metadata._structure()
        kept = metadata.where(pl.col("class_label") == 0, level="instance")
        assert kept.class_labels.tolist() == [0, 0]
        assert kept.level_counts["unit"] == 2
        assert kept.level_counts["track"] == 2, "where does not filter upwards or sideways"

    def test_neither_filter_mutates_the_source(self, tracking):
        name, metadata = tracking
        before = dict(metadata.level_counts)
        metadata.where(pl.col("item_index") == 0, level="unit")
        if metadata._levels.ancestors("instance"):
            metadata.having(pl.col("class_label") == 0, level="instance")
        assert dict(metadata.level_counts) == before, name
        assert not metadata._is_filtered, name

    def test_a_filtered_instance_says_so(self, tracking):
        name, metadata = tracking
        assert not metadata._is_filtered, name
        assert metadata.where(pl.col("item_index") == 0, level="unit")._is_filtered, name

    def test_a_descendant_column_is_rejected_and_names_having(self):
        """The dangerous shape: resolve() would answer null on every row rather than raise."""
        metadata = Metadata(_mot_dataset([[2, 2]]))
        metadata._structure()
        with pytest.raises(ValueError, match="having"):
            metadata.where(pl.col("class_label") == 0, level="unit")

    def test_a_sibling_branch_column_is_rejected(self):
        metadata = Metadata(_mot_dataset([[2, 2]]))
        metadata._structure()
        with pytest.raises(ValueError, match="different branch"):
            metadata.where(pl.col("track_id") > 0, level="unit")

    def test_an_unknown_column_is_rejected(self, tracking):
        _, metadata = tracking
        with pytest.raises(ValueError, match="not a column"):
            metadata.where(pl.col("no_such_factor") > 0, level="unit")

    def test_a_non_boolean_predicate_is_rejected(self, tracking):
        _, metadata = tracking
        with pytest.raises(ValueError, match="boolean"):
            metadata.where(pl.col("item_index") + 1, level="unit")

    def test_an_aggregate_predicate_is_rejected(self, tracking):
        _, metadata = tracking
        with pytest.raises(ValueError, match="one value per row"):
            metadata.where(pl.col("item_index").max() > 0, level="unit")

    def test_having_at_the_coarsest_level_is_rejected(self, tracking):
        _, metadata = tracking
        with pytest.raises(ValueError, match="coarsest level"):
            metadata.having(pl.col("item_index") == 0, level="sequence")

    def test_orphaned_rows_are_reported(self, caplog):
        """Cutting frames leaves the tracks whose observations were all in them."""
        metadata = Metadata(_mot_dataset([[2, 2], [[5, 6]]]))
        metadata._structure()
        with caplog.at_level(logging.INFO, logger="dataeval.metadata"):
            metadata.where(pl.col("item_index") == 0, level="unit")
        assert "with no remaining rows below them" in caplog.text
        assert "'track'" in caplog.text

    def test_an_already_childless_row_is_not_reported(self, caplog):
        """An empty frame is a shape the dataset arrived in, not something the filter did."""
        metadata = Metadata(_mot_dataset([[2, 0]]))
        metadata._structure()
        with caplog.at_level(logging.INFO, logger="dataeval.metadata"):
            metadata.where(pl.col("item_index") == 0, level="unit")
        assert "no remaining rows below" not in caplog.text

    def test_the_filter_defaults_to_the_current_view(self, tracking):
        name, metadata = tracking
        at_unit = metadata.at("unit")
        explicit = metadata.where(pl.col("item_index") == 0, level="unit")
        implicit = at_unit.where(pl.col("item_index") == 0)
        assert dict(implicit.level_counts) == dict(explicit.level_counts), name

    def test_filtering_keeps_the_bin_edges_it_was_given(self):
        """Filtering is not re-structuring: the surviving rows keep the bins they had."""
        metadata = Metadata(_mot_dataset([[4, 4], [[1, 2, 3]]]))
        metadata.continuous_factor_bins = {"time_s": 3}
        metadata._structure()
        metadata.factor_data  # noqa: B018  # bin
        before = metadata._store.frame("unit")["time_s↕"].to_list()
        kept = metadata.where(pl.col("item_index") == 0, level="unit")
        survivors = kept._store.frame("unit")["time_s↕"].to_list()
        assert survivors == before[: len(survivors)]

    def test_filtering_can_expose_a_factor_that_partial_ancestry_hid(self):
        """I5's consequence: a filter adds factors to the analysis and never removes one.

        A track factor is unreadable at the instance view while some detection is
        untracked. Dropping those detections makes the column total, so the factor joins
        ``factor_names`` — which is only safe because a filter cannot create an untracked
        detection where there was none.
        """
        metadata = Metadata(_mot_dataset([[[0, -1], [0, 1]]]))
        metadata._structure()
        assert metadata._store.partial_ancestry("track", "instance"), "fixture has an untracked detection"
        before = set(metadata.factor_names)

        kept = metadata.where(pl.col("track_index") >= 0, level="instance")
        assert not kept._store.partial_ancestry("track", "instance")
        assert before <= set(kept.factor_names), "a filter removed a factor from the analysis"


@pytest.mark.required
class TestPartialAncestryIsResolvedByFiltering:
    """FE-5 issue 5.4: ``where`` is what ``how="inner"`` would have been.

    The keyword was never implemented. ``where`` seeds at ``instance`` and nothing sits
    below it, so filtering on the track marker leaves exactly the tracked detections —
    which is the whole of what an inner join at that view would have done, without a
    second filtering path or its own row-count bookkeeping.
    """

    def test_track_factors_return_to_factor_data_after_the_filter(self):
        metadata = Metadata(_mot_dataset([[[0, -1], [0, 1]]]))
        metadata._structure()
        track_factors = set(metadata._factors_by_level["track"])
        assert track_factors, "fixture carries track-level factors"

        assert metadata._store.partial_ancestry("track", "instance")
        assert not (track_factors & set(metadata.factor_names)), "hidden while a detection is untracked"

        tracked = metadata.where(pl.col("track_index") >= 0, level="instance")
        assert not tracked._store.partial_ancestry("track", "instance")
        assert track_factors <= set(tracked.factor_names), "track factors did not return"

    def test_the_filter_keeps_exactly_the_tracked_rows(self):
        metadata = Metadata(_mot_dataset([[[0, -1], [0, 1]]]))
        metadata._structure()
        tracked = metadata.where(pl.col("track_index") >= 0, level="instance")
        assert tracked.level_counts["instance"] == 3, "one of four detections was untracked"
        assert (tracked._store.positions_from("instance", "track") >= 0).all()

    def test_a_fully_tracked_dataset_needs_no_filter(self):
        """The (b) half of the diamond fixture: nothing to resolve, nothing hidden."""
        metadata = Metadata(_mot_dataset([[[0, 1], [0, 1]]]))
        metadata._structure()
        assert not metadata._store.partial_ancestry("track", "instance")
        assert set(metadata._factors_by_level["track"]) <= set(metadata.factor_names)

    def test_class_labels_stay_aligned_with_factor_data_across_the_filter(self):
        """What ``how="inner"`` would have had to keep in lockstep by hand."""
        metadata = Metadata(_mot_dataset([[[0, -1], [0, 1]]]))
        metadata._structure()
        tracked = metadata.where(pl.col("track_index") >= 0, level="instance")
        assert len(tracked.class_labels) == tracked.factor_data.shape[0]
        assert len(tracked.class_labels) == tracked.level_counts["instance"]


@pytest.mark.required
class TestIsFilteredAndSelectedItems:
    """FE-5 issue 5.5: keeping the dataset side in correspondence, or refusing to guess."""

    def test_selected_items_names_the_surviving_items(self):
        metadata = Metadata(_mot_dataset([[2, 1], [[0, 2], [1]], [1]]))
        metadata._structure()
        kept = metadata.where(pl.col("item_index") != 1, level="sequence")
        assert kept.selected_items().tolist() == [0, 2]
        assert kept.item_count == 2, "item_count follows the filter"

    def test_selected_items_raises_when_the_filter_cut_below_an_item(self):
        """No dataset subset reproduces four frames of a video."""
        metadata = Metadata(_mot_dataset([[2, 1, 1], [[0, 2]]]))
        metadata._structure()
        partial = metadata.where(pl.col("unit_index") == 0, level="unit")
        with pytest.raises(ValueError, match="below the item level"):
            partial.selected_items()

    def test_the_cut_is_remembered_across_a_later_whole_item_filter(self):
        """A second, clean filter does not make the first one reproducible again."""
        metadata = Metadata(_mot_dataset([[2, 1, 1], [[0, 2]]]))
        metadata._structure()
        partial = metadata.where(pl.col("unit_index") == 0, level="unit")
        again = partial.where(pl.col("item_index") == 0, level="sequence")
        with pytest.raises(ValueError, match="below the item level"):
            again.selected_items()

    def test_an_unfiltered_metadata_names_every_item(self, tracking):
        name, metadata = tracking
        assert metadata.selected_items().tolist() == list(range(metadata.item_count)), name

    def test_is_filtered_is_public_and_survives_a_view_move(self, tracking):
        name, metadata = tracking
        kept = metadata.where(pl.col("item_index") == 0, level="unit")
        assert kept.is_filtered, name
        assert kept.at("instance").is_filtered, name
        assert not metadata.at("instance").is_filtered, name

    def test_an_aggregate_of_a_filtered_metadata_stays_filtered(self, tracking):
        name, metadata = tracking
        kept = metadata.where(pl.col("item_index") == 0, level="unit")
        assert kept.agg("instance", "unit", pl.len().alias("n")).is_filtered, name


@pytest.mark.required
class TestEmbeddingEvaluatorsRefuseFilteredMetadata:
    """Per D5, refused outright rather than warned — and whenever a filtered metadata is
    involved at all, since an evaluator that computes its own embeddings from the bound
    dataset produces exactly the same mismatch as one handed embeddings directly."""

    @pytest.fixture
    def filtered(self, get_od_dataset):
        metadata = Metadata(get_od_dataset(6, targets_per_image=2))
        metadata._structure()
        return metadata.where(pl.col("item_index") < 3, level="unit")

    def test_coverage_refuses(self, filtered):
        from dataeval.scope import Coverage

        with pytest.raises(ValueError, match="selected_items"):
            Coverage().evaluate(filtered)

    def test_outliers_refuses_a_filtered_metadata_argument(self, filtered):
        from dataeval.quality import Outliers

        with pytest.raises(ValueError, match="selected_items"):
            Outliers().evaluate(np.zeros((3, 3, 16, 16), dtype=np.float32), metadata=filtered)

    def test_prioritize_refuses(self, filtered):
        from dataeval.scope import Prioritize

        with pytest.raises(ValueError, match="selected_items"):
            Prioritize().evaluate(filtered)

    def test_an_unfiltered_metadata_is_not_refused(self, get_od_dataset):
        from dataeval._helpers import reject_filtered_metadata

        metadata = Metadata(get_od_dataset(6, targets_per_image=2))
        metadata._structure()
        reject_filtered_metadata(metadata, "Coverage")
        reject_filtered_metadata(None, "Coverage")
        reject_filtered_metadata(np.zeros(3), "Coverage")


@pytest.mark.required
def test_a_multi_column_predicate_is_rejected(get_od_dataset):
    """`pl.col('a', 'b')` asks two questions at once; a filter answers exactly one."""
    from dataeval._metadata._filters import evaluate

    metadata = Metadata(get_od_dataset(4, metadata=[{"a": 1.0, "b": 2.0}] * 4))
    metadata._structure()
    with pytest.raises(ValueError, match="must answer one column"):
        evaluate(metadata._store, metadata._item_level, pl.col("a", "b"))
