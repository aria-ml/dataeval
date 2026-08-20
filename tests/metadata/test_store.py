"""What the normalized store must hold, and what it must not.

The store's claim is that every fact is held once, at the granularity it was measured,
and that the flat frame is the store plus the gathers. These tests pin the parts of that
claim that are invisible from the outside — a column at the wrong level, or a full-height
copy of a per-image value, changes no result while undoing the whole point.
"""

import dataclasses

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._metadata._columns import to_col
from dataeval.types import FactorInfo
from tests.metadata.test_structurers import _mot_dataset


def _tasks(get_od_dataset):
    """One structured Metadata per task, each carrying factors at more than one level."""
    classification = Metadata(get_od_dataset(6, metadata=[{"w": "a", "n": 1.0}] * 6))
    detection = Metadata(get_od_dataset(6, targets_per_image=3, metadata=[{"w": "a", "n": 1.0}] * 6))
    tracking = Metadata(_mot_dataset([[2, 0, [1, -1]], [[0, 2], [1]]], [{"w": "a", "n": 1.0}] * 2))
    for metadata in (classification, detection, tracking):
        metadata._structure()
    return {"IC": classification, "OD": detection, "MOT": tracking}


@pytest.fixture
def tasks(get_od_dataset):
    return _tasks(get_od_dataset)


@pytest.mark.required
class TestNothingIsHeldTwice:
    """A level's frame carries that level's own columns and no others."""

    def test_no_frame_carries_another_levels_factor(self, tasks):
        for task, metadata in tasks.items():
            metadata.factor_data  # noqa: B018  # bin, so companions exist to be misplaced
            for level in metadata.levels:
                held = set(metadata._store.frame(level).columns)
                for other in metadata.levels:
                    if other == level:
                        continue
                    strays = held & metadata._factors_by_level[other]
                    assert not strays, f"{task}: {level} frame holds {other}-level factor(s) {strays}"

    def test_no_companion_column_is_full_height(self, tasks):
        """A bin column travels as its factor does — written once, gathered down.

        Held at full height instead, a per-image factor's bins would cost one value per
        detection, which is the storage the normalized model exists to remove.
        """
        for task, metadata in tasks.items():
            info = metadata.factor_info
            for name, factor_info in info.items():
                companion = to_col(name, factor_info)
                if companion == name:
                    continue
                level = metadata._factor_level(name)
                source = metadata._store.source_of(level, companion)
                assert source == level, f"{task}: {companion} is not native to {level}"
                assert metadata._store.frame(level).height == metadata.level_counts[level]

    def test_a_coarse_factors_companion_costs_fewer_values_than_a_copy(self, tasks):
        """Issue 4.1's acceptance, stated as the saving rather than as the placement.

        The height check above holds however the column is stored, because a level's
        frame is that level's height by construction. What 4.1 buys is that a factor
        measured per image is *binned* per image: on a dataset with more detections
        than images, its companion must be shorter than the label level's rows, and a
        companion written to the flat frame could not be.
        """
        for task, metadata in tasks.items():
            finest = metadata.level_counts[metadata.label_level]
            coarse = [
                (name, factor_info)
                for name, factor_info in metadata.factor_info.items()
                if metadata.level_counts[metadata._factor_level(name)] < finest
            ]
            if not coarse:
                continue
            for name, factor_info in coarse:
                companion = to_col(name, factor_info)
                level = metadata._factor_level(name)
                held = metadata._store.frame(level)[companion]
                assert len(held) == metadata.level_counts[level] < finest, f"{task}: {companion}"

    def test_companion_columns_stay_int64(self, tasks):
        """Per A5 the ``UInt16`` narrowing was withdrawn.

        A digitized categorical's ordinal comes from ``np.unique(..., return_inverse=True)``
        and is unbounded by the number of bins, so the width has to hold an arbitrary
        cardinality. Moving the column to its own level is what shrinks it.
        """
        for task, metadata in tasks.items():
            for name, factor_info in metadata.factor_info.items():
                companion = to_col(name, factor_info)
                if companion == name:
                    continue
                assert metadata._store.dtype_of(companion) == pl.Int64, f"{task}: {companion}"

    def test_a_factor_added_later_lands_only_at_its_level(self, tasks):
        metadata = tasks["OD"]
        metadata.add_factors({"added": np.arange(metadata.level_counts["unit"], dtype=np.float64)}, level="unit")
        assert "added" in metadata._store.frame("unit").columns
        assert "added" not in metadata._store.frame("instance").columns
        # Still readable from the instance rows, by gather rather than by a stored copy.
        assert metadata._store.column("instance", "added").null_count() == 0


@pytest.mark.required
class TestTheFlatFrameIsDerived:
    def test_rows_at_matches_a_filter_of_the_flat_frame(self, tasks):
        for task, metadata in tasks.items():
            flat = metadata.dataframe
            for level in metadata.levels:
                expected = flat.filter(pl.col("level") == level)
                assert metadata.rows_at(level).equals(expected), f"{task}.{level}"

    def test_the_flat_frame_follows_the_store(self, tasks):
        """No stored copy to go stale: a write to the store shows up in the frame."""
        metadata = tasks["OD"]
        before = metadata.dataframe
        metadata.add_factors({"late": np.arange(metadata.level_counts["unit"], dtype=np.float64)}, level="unit")
        assert "late" not in before.columns
        assert "late" in metadata.dataframe.columns

    def test_a_write_releases_the_memoized_frame(self, tasks):
        """Retired on the write, not on the next read.

        The flat frame is the largest object either side holds — a denormalized copy of
        every level — so a rebind that left it referenced would keep that copy alive for
        as long as nothing asked for :attr:`dataframe` again, which for a caller that
        only reads arrays is forever.
        """
        for task, metadata in tasks.items():
            metadata.dataframe  # noqa: B018  # memoize it
            assert metadata._flat is not None, task
            metadata.add_factors({f"x_{task}": np.arange(metadata.level_counts["unit"])}, level="unit")
            assert metadata._flat is None, f"{task}: stale frame still referenced after a write"

    def test_dropping_the_bins_releases_the_memoized_frame(self, tasks):
        """The write a caller is least aware of: no ``_store =`` appears at the call site.

        ``_reset_bins`` rebinds through ``without_columns``, and re-binding writes each
        companion back. Both are stores the memoized frame no longer describes.
        """
        for task, metadata in tasks.items():
            metadata.factor_data  # noqa: B018  # bin, so there are companions to drop
            metadata.dataframe  # noqa: B018  # memoize against the binned store
            assert metadata._flat is not None, task
            metadata._reset_bins()
            assert metadata._flat is None, f"{task}: stale frame survived _reset_bins"
            binned = metadata.dataframe
            metadata.factor_data  # noqa: B018  # re-bins, writing the companions back
            assert metadata._flat is None, f"{task}: stale frame survived re-binning"
            assert not binned.is_empty()

    def test_resolving_a_level_whose_column_is_unordered_raises(self, tasks):
        """The flat frame's contract is ``column_order``; a column outside it would vanish.

        Not reachable through any current writer — every one of them extends the order —
        but ``resolve`` is where the flat frame's column set is decided, and a factor
        silently missing from it reads as one that was never added rather than one lost.
        """
        store = tasks["OD"]._store
        ghost = pl.Series("ghost", range(store.height("unit")))
        smuggled = dataclasses.replace(store, frames={**store.frames, "unit": store.frame("unit").with_columns(ghost)})
        with pytest.raises(RuntimeError, match="ghost"):
            smuggled.resolve("unit")


@pytest.mark.required
class TestProjectionReadsOnlyWhatItAsksFor:
    """Issue 3.4's one way to get this wrong.

    ``_project`` must not route through ``rows_at``, whose wide default broadcasts every
    ancestor column and so rebuilds exactly the flat frame the store exists to avoid.
    """

    def test_a_projection_materializes_no_other_column(self, tasks):
        for task, metadata in tasks.items():
            info = metadata._factor_info
            if not info:
                continue
            name, factor_info = next(iter(info.items()))
            column = to_col(name, factor_info)
            selected = metadata._store.select(metadata.view, [column])
            assert selected.columns == [column], task
            assert selected.height == metadata.level_counts[metadata.view]

    def test_a_projection_never_calls_rows_at(self, tasks):
        """The direct pin, since a wide read and a narrow one return the same values.

        Nothing about the *result* of ``_project`` distinguishes the two routes — only
        the work done to get there — so this asserts the route.
        """
        for task, metadata in tasks.items():
            metadata.factor_data  # noqa: B018  # bin up front, outside the spy's window
            metadata.rows_at = lambda level, task=task: pytest.fail(f"{task}: _project widened via rows_at({level!r})")
            try:
                assert metadata.factor_data.shape[0] == metadata.level_counts[metadata.view]
                metadata.filter_by_factor(lambda *_: True)
            finally:
                del metadata.rows_at

    def test_the_projected_values_match_the_wide_read(self, tasks):
        for task, metadata in tasks.items():
            # Binned first: the companion columns only exist once _factor_info has run,
            # and the wide read has to be taken after them or it is a different frame.
            info = metadata._factor_info
            wide = metadata.rows_at(metadata.view)
            for name, factor_info in info.items():
                column = to_col(name, factor_info)
                narrow = metadata._store.select(metadata.view, [column])[column]
                assert narrow.to_list() == wide[column].to_list(), f"{task}.{column}"


@pytest.mark.required
class TestTheProjectionLinesUpWithTheLabels:
    """Issue 3.4's acceptance: the array-shaped accessors describe one set of rows."""

    def test_factor_data_matches_class_labels_at_the_default_view(self, tasks):
        for task, metadata in tasks.items():
            assert metadata.factor_data.shape[0] == len(metadata.class_labels), task

    def test_item_indices_matches_factor_data_at_every_view(self, tasks):
        for task, metadata in tasks.items():
            for level in metadata.levels:
                view = metadata.at(level)
                assert len(view.item_indices) == view.factor_data.shape[0], f"{task}.{level}"
                assert len(view) == view.factor_data.shape[0], f"{task}.{level}"


@pytest.mark.required
class TestTheLabelsAreHeldOnce:
    """Issue 3.5. The labels are a column of the label level's frame, and only that."""

    def test_class_labels_is_the_label_levels_column(self, tasks):
        for task, metadata in tasks.items():
            column = metadata.rows_at(metadata.label_level)["class_label"].to_numpy()
            assert np.array_equal(metadata.class_labels, column), task
            assert metadata.class_labels.dtype == np.intp, task

    def test_no_second_copy_of_the_labels_survives(self, tasks):
        """The field is gone, not merely unread.

        Two copies is what makes a relational operation able to misalign labels against
        factor rows — the worst answer this class can give — so the pin is that there is
        nowhere for a second copy to live rather than that the two currently agree.
        """
        for task, metadata in tasks.items():
            assert not hasattr(metadata, "_class_labels"), task

    def test_labels_follow_a_write_to_the_label_level(self, tasks):
        """Derived, so a store that moves takes the labels with it."""
        metadata = tasks["OD"]
        level = metadata.label_level
        flipped = metadata.class_labels[::-1].copy()
        metadata._store = metadata._store.with_column(level, pl.Series("class_label", flipped), propagates=False)
        assert np.array_equal(metadata.class_labels, flipped)

    def test_class_labels_is_empty_rather_than_absent_without_a_dataset(self):
        metadata = Metadata.from_factors({"a": np.arange(6.0)})
        assert metadata.class_labels.tolist() == [0] * 6


@pytest.mark.required
class TestStoreDegenerateLevels:
    """A level the schema knows but that holds no rows must answer, not raise."""

    @staticmethod
    def _missing_a_level(metadata) -> tuple:
        """The MOT store with its finest level's rows dropped, plus that level's name."""
        store = metadata._store
        gone = list(store.frames)[-1]
        return dataclasses.replace(store, frames={k: v for k, v in store.frames.items() if k != gone}), gone

    def test_source_of_an_unpopulated_level_is_none(self, tasks):
        store, gone = self._missing_a_level(tasks["MOT"])
        assert store.source_of(gone, "anything") is None

    def test_with_column_on_an_unpopulated_level_is_a_no_op(self, tasks):
        store, gone = self._missing_a_level(tasks["MOT"])
        assert store.with_column(gone, pl.Series("added", [1])) is store

    def test_surviving_having_skips_unpopulated_ancestors(self, tasks):
        """An ancestor with no rows contributes nothing rather than failing the walk."""
        store = tasks["MOT"]._store
        level = list(store.frames)[-1]
        thinned = dataclasses.replace(
            store, frames={k: v for k, v in store.frames.items() if k in (level, list(store.frames)[0])}
        )
        mask = np.ones(thinned.frames[level].height, dtype=np.bool_)
        assert level in thinned.surviving_having(level, mask)

    def test_flat_of_an_empty_store_keeps_the_column_order(self, tasks):
        store = dataclasses.replace(tasks["MOT"]._store, frames={})
        flat = store.flat()
        assert flat.height == 0
        assert tuple(flat.columns) == tuple(store.column_order)


@pytest.mark.required
class TestToCol:
    """Which column a reader of a factor should select, given what binning produced."""

    def test_an_unbinned_undigitized_factor_reads_its_own_name(self):
        assert to_col("brightness", FactorInfo("continuous")) == "brightness"

    def test_binning_is_ignored_when_the_caller_asks_for_the_raw_column(self):
        assert to_col("brightness", FactorInfo("continuous", is_binned=True), is_binned=False) == "brightness"


@pytest.mark.required
def test_surviving_where_skips_unpopulated_ancestors(tasks):
    """An ancestor with no frame is simply absent from the answer, not an error."""
    store = tasks["MOT"]._store
    level = list(store.frames)[-1]
    thinned = dataclasses.replace(store, frames={level: store.frames[level]})
    mask = np.zeros(thinned.frames[level].height, dtype=np.bool_)
    mask[0] = True
    survivors = thinned.surviving_where(level, mask)
    assert set(survivors) == {level}
