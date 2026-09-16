"""Roll-up recipes shipped by a producer and applied on add_factors."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.types import Aggregator
from tests.metadata.test_structurers import _mot_dataset


def _video(frames_per_sequence=(5, 5)):
    """Structured tracking metadata: two sequences of five frames each."""
    metadata = Metadata(_mot_dataset([[1] * n for n in frames_per_sequence]))
    metadata._structure()
    return metadata


def _result(values, aggregations=()):
    """A minimal stats result placing one per-frame factor, with declarations."""
    return {
        "stats": {"pan_speed": np.asarray(values, dtype=float)},
        "aggregations": tuple(aggregations),
    }


def _two_factor_result(pan_speed, other, aggregations=()):
    """A stats result placing two per-frame factors, with declarations."""
    return {
        "stats": {"pan_speed": np.asarray(pan_speed, dtype=float), "other": np.asarray(other, dtype=float)},
        "aggregations": tuple(aggregations),
    }


MEDIAN = Aggregator("median", "unit", "sequence", ("pan_speed",), min_coverage=0.25)


@pytest.mark.required
class TestDeclarationsApply:
    def test_a_declared_roll_up_runs_on_add(self):
        md = _video()
        md.add_factors(cast("Any", _result([10.0, 12.0, 14.0, 16.0, 18.0] * 2, [MEDIAN])), level="unit")
        assert "pan_speed_median" in md.factor_names
        assert md.rows_at("sequence")["pan_speed_median"].to_list() == [14.0, 14.0]

    def test_the_source_factor_is_still_stored(self):
        md = _video()
        md.add_factors(cast("Any", _result([1.0] * 10, [MEDIAN])), level="unit")
        assert "pan_speed" in md.factor_names
        assert md.rows_at("unit").height == 10

    def test_a_result_with_no_declarations_rolls_up_nothing(self):
        md = _video()
        md.add_factors(cast("Any", _result([1.0] * 10)), level="unit")
        assert "pan_speed_median" not in md.factor_names

    def test_declarations_register_even_when_the_result_adds_no_new_values(self):
        # A later call with no new factors, only a declaration over one already stored, is
        # still work to do, not a no-op the empty-factors early return should discard.
        md = _video()
        md.add_factors(cast("Any", _result([1.0] * 10)), level="unit")
        empty = {"stats": {}, "aggregations": (MEDIAN,)}
        md.add_factors(cast("Any", empty), level="unit")
        assert "pan_speed_median" in md.factor_names

    def test_adding_twice_does_not_double_apply(self):
        md = _video()
        md.add_factors(cast("Any", _result([1.0] * 10, [MEDIAN])), level="unit")
        md.add_factors({"other": np.arange(10, dtype=float)}, level="unit")
        rolled = [n for n in md.factor_names if n.startswith("pan_speed_median")]
        assert rolled == ["pan_speed_median"]

    def test_a_second_call_with_its_own_declarations_does_not_duplicate_the_first(self):
        # The second call must also carry a declaration. A plain mapping never replays, so
        # it would not exercise `_replay_aggregations` running a second time onto a store
        # that already holds the first call's rolled-up column.
        md = _video()
        md.add_factors(cast("Any", _result([1.0] * 10, [MEDIAN])), level="unit")
        other_max = Aggregator("max", "unit", "sequence", ("other",), min_coverage=0.25)
        second = {"stats": {"other": np.arange(10, dtype=float)}, "aggregations": (other_max,)}
        md.add_factors(cast("Any", second), level="unit")
        rolled = {n for n in md.factor_names if n.startswith(("pan_speed_median", "other_max"))}
        assert rolled == {"pan_speed_median", "other_max"}

    def test_overwriting_the_source_factor_refreshes_its_roll_up(self):
        # A roll-up already on file must be recomputed when its source is re-landed, not
        # left holding the value computed against the old contents. It must land back under
        # its own name, not a duplicate `_agg`-suffixed one.
        md = _video()
        md.add_factors(cast("Any", _result([10.0, 12.0, 14.0, 16.0, 18.0] * 2, [MEDIAN])), level="unit")
        assert md.rows_at("sequence")["pan_speed_median"].to_list() == [14.0, 14.0]

        md.add_factors({"pan_speed": np.array([100.0, 200.0, 300.0, 400.0, 500.0] * 2)}, level="unit", overwrite=True)

        assert md.rows_at("sequence")["pan_speed_median"].to_list() == [300.0, 300.0]
        assert [n for n in md.factor_names if n.startswith("pan_speed_median")] == ["pan_speed_median"]


@pytest.mark.required
class TestCoverageGate:
    """min_coverage is what makes an untrusted frame abstain rather than mislead."""

    def test_a_partly_trusted_sequence_answers(self):
        md = _video()
        values = [10.0, 12.0, np.nan, np.nan, 14.0] + [np.nan] * 5
        md.add_factors(cast("Any", _result(values, [MEDIAN])), level="unit")
        assert md.rows_at("sequence")["pan_speed_median"].to_list()[0] == 12.0

    def test_a_mostly_untrusted_sequence_abstains(self):
        md = _video()
        values = [1.0] * 5 + [1.0, np.nan, np.nan, np.nan, np.nan]
        md.add_factors(cast("Any", _result(values, [MEDIAN])), level="unit")
        assert md.rows_at("sequence")["pan_speed_median"].to_list()[1] is None

    def test_the_default_coverage_would_poison_the_whole_sequence(self):
        # Guards the 0.25 default: at min_coverage=1.0 a single null nulls the destination.
        md = _video()
        strict = Aggregator("median", "unit", "sequence", ("pan_speed",))
        values = [10.0, 12.0, np.nan, 14.0, 16.0] * 2
        md.add_factors(cast("Any", _result(values, [strict])), level="unit")
        assert md.rows_at("sequence")["pan_speed_median"].to_list() == [None, None]


@pytest.mark.required
class TestOverrides:
    def test_aggregate_false_suppresses_them(self):
        md = _video()
        md.add_factors(cast("Any", _result([1.0] * 10, [MEDIAN])), level="unit", aggregate=False)
        assert "pan_speed_median" not in md.factor_names

    def test_aggregate_false_on_a_plain_mapping_is_not_an_error(self):
        md = _video()
        md.add_factors({"a": np.arange(10, dtype=float)}, level="unit", aggregate=False)
        assert "a" in md.factor_names

    def test_how_swaps_one_reduction_and_keeps_the_rest(self):
        md = _video()
        values = [10.0, 12.0, 14.0, 16.0, 18.0] * 2
        md.add_factors(cast("Any", _result(values, [MEDIAN])), level="unit", how={"pan_speed": "max"})
        assert "pan_speed_max" in md.factor_names
        assert "pan_speed_median" not in md.factor_names
        assert md.rows_at("sequence")["pan_speed_max"].to_list() == [18.0, 18.0]

    def test_aggregations_replaces_wholesale(self):
        md = _video()
        replacement = [Aggregator("min", "unit", "sequence", ("pan_speed",), min_coverage=0.25)]
        values = [10.0, 12.0, 14.0, 16.0, 18.0] * 2
        md.add_factors(cast("Any", _result(values, [MEDIAN])), level="unit", aggregations=replacement)
        assert "pan_speed_min" in md.factor_names
        assert "pan_speed_median" not in md.factor_names

    def test_how_overrides_only_the_named_factor_in_a_multi_factor_declaration(self):
        md = _video()
        multi = Aggregator("median", "unit", "sequence", ("pan_speed", "other"), min_coverage=0.25)
        pan_speed = [10.0, 12.0, 14.0, 16.0, 18.0] * 2
        other = [1.0, 2.0, 3.0, 4.0, 5.0] * 2
        result = _two_factor_result(pan_speed, other, [multi])
        md.add_factors(cast("Any", result), level="unit", how={"pan_speed": "max"})
        assert "pan_speed_max" in md.factor_names
        assert "pan_speed_median" not in md.factor_names
        # `other` was not named in `how`, so it keeps the declaration's own reduction.
        assert "other_median" in md.factor_names
        assert "other_max" not in md.factor_names
        assert md.rows_at("sequence")["pan_speed_max"].to_list() == [18.0, 18.0]
        assert md.rows_at("sequence")["other_median"].to_list() == [3.0, 3.0]

    def test_how_naming_two_reductions_for_one_declaration_raises(self):
        md = _video()
        multi = Aggregator("median", "unit", "sequence", ("pan_speed", "other"), min_coverage=0.25)
        result = _two_factor_result([1.0] * 10, [2.0] * 10, [multi])
        with pytest.raises(ValueError, match="declared together as"):
            md.add_factors(cast("Any", result), level="unit", how={"pan_speed": "max", "other": "min"})

    def test_how_naming_an_unknown_factor_raises(self):
        md = _video()
        with pytest.raises(ValueError, match="does not declare"):
            md.add_factors(cast("Any", _result([1.0] * 10, [MEDIAN])), level="unit", how={"nope": "max"})

    def test_how_without_declarations_raises(self):
        md = _video()
        with pytest.raises(ValueError, match="declares no roll-ups"):
            md.add_factors(cast("Any", _result([1.0] * 10)), level="unit", how={"pan_speed": "max"})

    def test_how_and_aggregations_together_raise(self):
        md = _video()
        with pytest.raises(ValueError, match="mutually exclusive"):
            md.add_factors(
                cast("Any", _result([1.0] * 10, [MEDIAN])),
                level="unit",
                how={"pan_speed": "max"},
                aggregations=[MEDIAN],
            )


@pytest.mark.required
class TestRoundTrip:
    def test_declared_roll_ups_survive_save_and_load(self, tmp_path):
        dataset = _mot_dataset([[1] * 5, [1] * 5])
        md = Metadata(dataset)
        md._structure()
        values = [10.0, 12.0, 14.0, 16.0, 18.0] * 2
        md.add_factors(cast("Any", _result(values, [MEDIAN])), level="unit")

        path = tmp_path / "metadata.dem"
        md.save(path)
        restored = Metadata.load(path, dataset)

        # (a) Rolled-up output column and its values survived
        assert "pan_speed_median" in restored.factor_names
        assert restored.rows_at("sequence")["pan_speed_median"].to_list() == [14.0, 14.0]

        # (b) Aggregator recipes survived: check the recipe itself, not just the output
        assert "pan_speed_median" in restored._aggregations
        assert restored._aggregations["pan_speed_median"].how == "median"
        assert restored._aggregations["pan_speed_median"].min_coverage == 0.25

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Carried recipes are discarded before they can be satisfied: add_factors calls "
            "_structure() first, which replays while the roll-up's source factor has not "
            "landed yet, and the unanswerable recipe is dropped rather than retained. "
            "Pre-existing, not introduced by declared aggregations."
        ),
    )
    def test_restored_recipes_replay_on_a_different_dataset(self, tmp_path):
        """The reason recipes are persisted at all: a restored metadata can measure new data."""
        first = _mot_dataset([[1] * 5, [1] * 5])
        md = Metadata(first)
        md._structure()
        md.add_factors(cast("Any", _result([10.0, 12.0, 14.0, 16.0, 18.0] * 2, [MEDIAN])), level="unit")

        path = tmp_path / "metadata.dem"
        md.save(path)
        restored = Metadata.load(path, first)

        # Bind to a DIFFERENT shape so a stale cached column could not satisfy the assertion
        second = _mot_dataset([[1] * 3, [1] * 7, [1] * 4])
        fresh = restored.new(second)
        # Add pan_speed values for the 14 unit rows, with distinct per-sequence medians
        # Sequence 0 (3 rows): [1.0, 2.0, 3.0] -> median 2.0
        # Sequence 1 (7 rows): [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0] -> median 13.0
        # Sequence 2 (4 rows): [100.0, 101.0, 102.0, 103.0] -> median 101.5
        values = [1.0, 2.0, 3.0] + [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0] + [100.0, 101.0, 102.0, 103.0]
        # Pass NO declarations: the roll-up comes from the *carried* recipe
        fresh.add_factors(cast("Any", _result(values, [])), level="unit")

        # The carried recipe must have replayed: pan_speed_median exists on new data
        assert "pan_speed_median" in fresh.factor_names
        # And it was computed for the new structure: 3 sequences
        assert fresh.rows_at("sequence").height == 3
        # The per-sequence medians are from the new data, computed by the carried recipe
        assert fresh.rows_at("sequence")["pan_speed_median"].to_list() == [2.0, 13.0, 101.5]
