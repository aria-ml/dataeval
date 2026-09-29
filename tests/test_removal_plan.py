import dataclasses

import numpy as np
import pytest

from dataeval.types import RemovalPlan, SourceIndex


@pytest.mark.required
class TestRemovalPlan:
    """A set of addresses to remove, spelled once each."""

    def test_ints_are_stored_as_item_addresses(self):
        assert RemovalPlan([3, 1]).discard == frozenset({SourceIndex(1), SourceIndex(3)})

    def test_numpy_integers_are_ints(self):
        assert RemovalPlan(np.array([2, 5])).discard == frozenset({SourceIndex(2), SourceIndex(5)})

    def test_two_spellings_of_one_row_are_one_address(self):
        plan = RemovalPlan([
            SourceIndex(3, 7),
            SourceIndex(3, 7, "instance"),
            SourceIndex(4),
            SourceIndex(4, None, "sequence"),
        ])
        assert plan.discard == frozenset({SourceIndex(3, 7), SourceIndex(4)})

    def test_a_frame_or_track_address_keeps_its_level(self):
        plan = RemovalPlan([SourceIndex(0, 12, "unit"), SourceIndex(0, 5, "track")])
        assert plan.discard == frozenset({SourceIndex(0, 12, "unit"), SourceIndex(0, 5, "track")})

    def test_union_keeps_every_row_and_collapses_spellings(self):
        combined = RemovalPlan([SourceIndex(3, 7)]) | RemovalPlan([SourceIndex(3, 7, "instance"), 1])
        assert combined == RemovalPlan([1, SourceIndex(3, 7)])

    def test_union_with_something_else_is_refused(self):
        with pytest.raises(TypeError):
            _ = RemovalPlan([1]) | {2}

    def test_iteration_is_in_address_order(self):
        plan = RemovalPlan([SourceIndex(2), SourceIndex(0, 3), SourceIndex(0), SourceIndex(0, 1)])
        assert list(plan) == [SourceIndex(0), SourceIndex(0, 1), SourceIndex(0, 3), SourceIndex(2)]

    def test_an_int_finds_the_item_it_names(self):
        plan = RemovalPlan([3, SourceIndex(5, 2)])
        assert 3 in plan
        assert 4 not in plan

    def test_a_numpy_integer_finds_the_item_it_names(self):
        plan = RemovalPlan([3])
        assert np.int64(3) in plan
        assert np.int64(4) not in plan

    def test_any_spelling_of_a_row_finds_it(self):
        plan = RemovalPlan([SourceIndex(5, 2), SourceIndex(6, None, "sequence")])
        assert SourceIndex(5, 2, "instance") in plan
        assert SourceIndex(6) in plan
        assert SourceIndex(5, 1) not in plan

    def test_a_detection_inside_a_named_item_is_not_itself_named(self):
        assert SourceIndex(3, 0) not in RemovalPlan([3])

    def test_something_that_is_not_an_address_is_not_in_a_plan(self):
        plan = RemovalPlan([3])
        assert "x" not in plan
        assert (3, 0) not in plan
        assert None not in plan

    def test_length_counts_distinct_rows(self):
        assert len(RemovalPlan([1, 1, SourceIndex(1)])) == 1

    def test_an_empty_plan_is_falsy(self):
        assert not RemovalPlan()

    def test_plans_compare_and_hash_by_their_rows(self):
        assert RemovalPlan([1, 2]) == RemovalPlan([SourceIndex(2), 1])
        assert len({RemovalPlan([1]), RemovalPlan([SourceIndex(1)])}) == 1

    def test_a_plan_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            RemovalPlan([1]).discard = frozenset()  # type: ignore[misc]

    def test_repr_counts_each_level(self):
        plan = RemovalPlan([0, 1, SourceIndex(2, 0), SourceIndex(2, 4, "unit"), SourceIndex(2, 9, "track")])
        assert repr(plan) == "RemovalPlan(items=2, instances=1, units=1, tracks=1)"

    def test_repr_of_an_empty_plan(self):
        assert repr(RemovalPlan()) == "RemovalPlan()"

    def test_something_that_is_not_an_address_is_refused(self):
        with pytest.raises(TypeError, match="int or a SourceIndex"):
            RemovalPlan([(3, 7)])  # type: ignore[list-item]
