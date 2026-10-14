"""The level resolver Duplicates and Outliers share, and the helper that reads a result frame's kinds."""

import re

import polars as pl
import pytest

from dataeval._helpers import lexical_categorical
from dataeval.quality._shared import resolve_levels, row_kinds


def _plan(levels=None, per_image=None, per_target=None, *, tracking=False, **names):
    return resolve_levels(levels, per_image, per_target, tracking, caller="Outliers.evaluate", **names)


@pytest.mark.required
class TestImpliedLevels:
    @pytest.mark.parametrize(
        ("per_image", "per_target", "expected"),
        [
            (None, None, {"item"}),
            (True, False, {"item"}),
            (True, True, {"item", "target"}),
            (False, True, {"target"}),
        ],
    )
    def test_an_image_dataset(self, per_image, per_target, expected):
        assert _plan(per_image=per_image, per_target=per_target).levels == frozenset(expected)

    @pytest.mark.parametrize(
        ("per_image", "per_target", "expected"),
        [
            (None, None, {"sequence", "unit"}),
            (True, True, {"sequence", "unit", "instance", "track"}),
            (False, True, {"instance", "track"}),
        ],
    )
    def test_a_tracking_dataset(self, per_image, per_target, expected):
        assert _plan(per_image=per_image, per_target=per_target, tracking=True).levels == frozenset(expected)

    def test_the_flags_travel_with_the_plan(self):
        plan = _plan(per_image=False, per_target=True)
        assert (plan.per_image, plan.per_target) == (False, True)

    def test_no_flag_at_all_resolves_to_nothing(self):
        # Callers refuse this before resolving, so the resolver does not have to.
        assert _plan(per_image=False, per_target=False).levels == frozenset()


@pytest.mark.required
class TestExplicitLevels:
    @pytest.mark.parametrize(
        ("levels", "per_image", "per_target"),
        [("item", True, False), ("target", False, True), (["item", "target"], True, True)],
    )
    def test_image_names_derive_the_flags(self, levels, per_image, per_target):
        plan = _plan(levels)
        assert (plan.per_image, plan.per_target) == (per_image, per_target)

    @pytest.mark.parametrize(
        ("level", "per_target"), [("sequence", False), ("unit", False), ("track", True), ("instance", True)]
    )
    def test_tracking_names_always_measure_frames(self, level, per_target):
        plan = _plan(level, tracking=True)
        assert plan.per_image is True
        assert plan.per_target is per_target
        assert plan.levels == frozenset({level})

    def test_a_tuple_and_a_repeated_name_are_a_set(self):
        assert _plan(("track", "track"), tracking=True).levels == frozenset({"track"})


def _exactly(message):
    return f"^{re.escape(message)}$"


@pytest.mark.required
class TestErrors:
    def test_both_spellings(self):
        message = (
            "Outliers.evaluate: pass either `levels` or `per_image`/`per_target`, not both -- they are two "
            "spellings of one thing, and reconciling them silently would hide whichever lost. `levels` for this "
            "dataset accepts ['item', 'target']."
        )
        with pytest.raises(ValueError, match=_exactly(message)):
            _plan("item", per_image=True)

    def test_an_unknown_name_says_what_the_task_accepts(self):
        message = (
            "Outliers.evaluate: ['bogus', 'other'] are not a level of a tracking dataset; "
            "it accepts ['sequence', 'unit', 'track', 'instance']."
        )
        with pytest.raises(ValueError, match=_exactly(message)):
            _plan(["bogus", "other"], tracking=True)

    def test_an_empty_levels(self):
        with pytest.raises(ValueError, match="must name at least one level"):
            _plan([])

    def test_the_subject_replaces_the_word_dataset(self):
        with pytest.raises(ValueError, match="not a level of an image metadata"):
            _plan("bogus", subject="metadata")

    def test_the_duplicates_messages_are_unchanged(self):
        unknown = "Duplicates.evaluate: ['bogus'] is not a level of an image dataset; it accepts ['item', 'target']."
        with pytest.raises(ValueError, match=_exactly(unknown)):
            resolve_levels("bogus", None, None, False, caller="Duplicates.evaluate")
        empty = "Duplicates.evaluate: `levels` must name at least one level."
        with pytest.raises(ValueError, match=_exactly(empty)):
            resolve_levels([], None, None, True, caller="Duplicates.evaluate")


@pytest.mark.required
class TestRowKinds:
    @staticmethod
    def _frame():
        return pl.DataFrame(
            {"item_index": [0, 1, 2, 3], "target_index": [None, 4, 5, None], "level": ["unit", None, "track", None]},
            schema={"item_index": pl.Int64, "target_index": pl.Int64, "level": lexical_categorical()},
        )

    @pytest.mark.parametrize(
        ("drop", "expected"),
        [
            ([], ["unit", "instance", "track", None]),
            (["level"], [None, "instance", "instance", None]),
            (["target_index"], ["unit", None, "track", None]),
            (["level", "target_index"], [None, None, None, None]),
        ],
    )
    def test_a_missing_column_reads_as_all_null(self, drop, expected):
        assert row_kinds(self._frame().drop(drop)).to_list() == expected

    def test_an_empty_frame(self):
        assert row_kinds(self._frame().clear()).to_list() == []
        assert row_kinds(self._frame().clear().drop("level", "target_index")).to_list() == []
