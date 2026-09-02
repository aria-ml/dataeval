"""Named reductions: what a roll-up's name promises, and what it refuses.

The claim under test is that a name carries three things an expression does not — a value
type, an answer for an empty group, and a stable output name — and that the named surface
computes exactly what the expression surface computes when told the same thing.
"""

import logging

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._metadata._reductions import REDUCTIONS, admits, identity_of, lookup, resolve
from dataeval.types import Aggregator, FactorLevelSchema
from tests.metadata.test_structurers import _mot_dataset

SCHEMA = FactorLevelSchema.of("sequence", "unit", "track", "instance")


def _tracking(**factors):
    """A tracking Metadata with per-sequence factors, plus any extra levels asked for."""
    metadata = Metadata(_mot_dataset([[2, 0, [1, -1]], [[0, 2], [1]]], [{"w": "a", "n": 1.0}] * 2))
    metadata._structure()
    for level, values in factors.items():
        metadata.add_factors(values, level=level)  # type: ignore[arg-type]
    return metadata


@pytest.mark.required
class TestAggregatorIsADeclaration:
    """Everything decidable without a dataset is decided without one."""

    def test_the_output_name_pairs_the_factor_with_the_reduction(self):
        assert Aggregator("mean", "unit", "sequence").name_for("brightness") == "brightness_mean"

    def test_a_route_that_is_not_the_default_enters_the_name(self):
        """The name is the only durable record of the operation, so it has to carry this.

        Two roll-ups of one factor to one level by different routes are different
        questions; a name that could not tell them apart would leave them distinguished
        only by a uniqueness suffix.
        """
        assert Aggregator("mean", "instance", "sequence", via="track").name_for("x") == "x_mean_via_track"

    def test_a_suffix_overrides_the_derived_name(self):
        assert Aggregator("mean", "unit", "sequence", suffix="_avg").name_for("x") == "x_avg"

    def test_a_bare_string_factor_set_is_rejected(self):
        """``str`` is a sequence of ``str``, so this would silently become one factor per letter."""
        with pytest.raises(TypeError, match="not the bare string"):
            Aggregator("mean", "unit", "sequence", "brightness")  # type: ignore[arg-type]

    def test_rolling_into_the_level_it_rolls_from_is_rejected(self):
        with pytest.raises(ValueError, match="cannot both be 'unit'"):
            Aggregator("mean", "unit", "unit")

    def test_a_target_below_the_source_is_rejected_against_a_bare_schema(self):
        with pytest.raises(ValueError, match="does not sit above"):
            Aggregator("mean", "sequence", "instance").validate(SCHEMA)

    def test_a_sibling_target_is_rejected_against_a_bare_schema(self):
        with pytest.raises(ValueError, match="does not sit above"):
            Aggregator("mean", "unit", "track").validate(SCHEMA)

    def test_a_unique_by_below_the_source_is_rejected(self):
        with pytest.raises(ValueError, match="must be 'track' itself"):
            Aggregator("mean", "track", "sequence", unique_by="instance").validate(SCHEMA)

    def test_an_unresolved_source_validates_only_what_it_can(self):
        Aggregator("mean", None, "sequence").validate(SCHEMA)

    def test_an_unresolved_aggregator_has_no_source_to_give(self):
        with pytest.raises(ValueError, match="has not been resolved"):
            Aggregator("mean", None, "sequence").rolls_from  # noqa: B018


@pytest.mark.required
class TestTheRegistry:
    """A reduction's name is a contract, so every name has to state one."""

    def test_an_unknown_reduction_lists_the_known_ones(self):
        with pytest.raises(ValueError, match=r"'meen' is not a reduction. The reductions are \['all'"):
            lookup("meen")

    @pytest.mark.parametrize("how", sorted(REDUCTIONS))
    def test_every_reduction_declares_a_domain_and_a_kind(self, how):
        reduction = lookup(how)
        assert reduction.domain in {"numeric", "orderable", "boolean", "any"}
        assert reduction.kind in {"positional", "temporal"}

    @pytest.mark.parametrize("how", sorted(REDUCTIONS))
    def test_only_a_temporal_reduction_can_be_gap_sensitive(self, how):
        """A gap is a property of an ordering, so a reduction that reads none cannot see one."""
        reduction = lookup(how)
        assert reduction.kind == "temporal" or not reduction.gap_sensitive

    @pytest.mark.parametrize(
        ("how", "dtype", "expected"),
        [
            ("mean", pl.Float64(), True),
            ("mean", pl.String(), False),
            ("mean", pl.Boolean(), False),
            ("max", pl.String(), True),
            ("max", pl.Datetime("us"), True),
            ("any", pl.Boolean(), True),
            ("any", pl.Int64(), False),
            ("mode", pl.String(), True),
            ("mode", pl.Array(pl.Float64, 4), False),
            ("count", pl.Null(), False),
        ],
    )
    def test_a_reduction_admits_only_the_values_it_is_about(self, how, dtype, expected):
        assert admits(lookup(how), dtype) is expected

    @pytest.mark.parametrize(
        ("how", "identity"),
        [("count", 0), ("sum", 0), ("n_unique", 0), ("any", False), ("all", True)],
    )
    def test_the_reductions_with_an_identity_declare_it(self, how, identity):
        assert identity_of(Aggregator(how, "instance", "unit")) == identity

    @pytest.mark.parametrize("how", ["mean", "median", "min", "max", "std", "var", "mode", "first", "last"])
    def test_the_reductions_without_one_answer_undefined(self, how):
        assert identity_of(Aggregator(how, "instance", "unit")) is None


@pytest.mark.required
class TestResolution:
    """Turning a rule into the concrete roll-ups it names, against a dataset."""

    @staticmethod
    def _resolve(aggregator, metadata):
        levels = {name: level for level, names in metadata._factors_by_level.items() for name in names}
        dtypes = {name: metadata._store.dtype_of(name) for name in levels}
        native = {level: frozenset(metadata._store.frame(level).columns) for level in metadata._store.frames}
        return resolve(aggregator, metadata._levels, levels, dtypes, native)

    def test_the_source_is_inferred_from_where_the_factor_is_defined(self):
        metadata = _tracking()
        (one,) = self._resolve(Aggregator("mean", None, "sequence", ("time_s",)), metadata)
        assert one.rolls_from == "unit"

    def test_inference_records_itself_as_a_fit(self):
        """Resolution reads a dataset, so it is derived rather than declared — like bin edges."""
        metadata = _tracking()
        (one,) = self._resolve(Aggregator("mean", None, "sequence", ("time_s",)), metadata)
        assert one.provenance == "derived"

    def test_a_fully_specified_aggregator_passes_through_unchanged(self):
        metadata = _tracking()
        stated = Aggregator("mean", "unit", "sequence", ("time_s",))
        assert self._resolve(stated, metadata) == (stated,)

    def test_factors_at_different_levels_become_one_roll_up_each(self):
        metadata = _tracking(track={"speed": np.arange(5.0)})
        resolved = self._resolve(Aggregator("mean", None, "sequence"), metadata)
        assert {one.rolls_from for one in resolved} == {"unit", "track"}

    def test_factors_sharing_a_level_share_one_roll_up(self):
        metadata = _tracking()
        (one,) = self._resolve(Aggregator("mean", "unit", "sequence"), metadata)
        assert set(one.factors) >= {"time_s", "width", "height"}

    def test_a_named_factor_the_reduction_does_not_apply_to_is_refused(self):
        """Naming a factor is a request, so it is answered rather than filtered."""
        metadata = _tracking(instance={"kind": np.array(["a"] * 7)})
        with pytest.raises(ValueError, match="'mean' does not apply to 'kind'"):
            self._resolve(Aggregator("mean", None, "unit", ("kind",)), metadata)

    def test_the_refusal_names_the_reductions_that_would_apply(self):
        metadata = _tracking(instance={"kind": np.array(["a"] * 7)})
        with pytest.raises(ValueError, match=r"Reductions that apply to 'kind' are \['changes'"):
            self._resolve(Aggregator("mean", None, "unit", ("kind",)), metadata)

    def test_an_unnamed_factor_the_reduction_does_not_apply_to_is_passed_over(self, caplog):
        """An empty factor set is a rule, so selecting is what it is for."""
        caplog.set_level(logging.INFO, logger="dataeval.metadata")
        metadata = _tracking(instance={"kind": np.array(["a"] * 7), "area": np.arange(7.0)})
        (one,) = self._resolve(Aggregator("mean", None, "unit"), metadata)
        assert "kind" not in one.factors
        assert "passed over ['kind']" in caplog.text

    def test_an_unknown_factor_is_rejected(self):
        with pytest.raises(ValueError, match="'nope' is not a factor of this metadata"):
            self._resolve(Aggregator("mean", None, "sequence", ("nope",)), _tracking())

    def test_a_factor_the_target_does_not_sit_above_is_rejected(self):
        with pytest.raises(ValueError, match="Cannot roll 'w' up into 'sequence'"):
            self._resolve(Aggregator("mean", None, "sequence", ("w",)), _tracking())

    def test_a_rule_that_selects_nothing_is_rejected(self):
        metadata = _tracking()
        with pytest.raises(ValueError, match="Nothing to roll up into 'sequence' with 'any'"):
            self._resolve(Aggregator("any", None, "sequence"), metadata)


@pytest.mark.required
class TestAggregate:
    """The public surface, and its agreement with the expression form beneath it."""

    def test_a_named_reduction_lands_under_a_derived_name(self):
        rolled = _tracking().aggregate("time_s", level="sequence", how="mean")
        assert rolled.at("sequence").rows_at("sequence")["time_s_mean"].to_list() == [0.5, 0.25]

    def test_a_mapping_gives_one_reduction_per_factor(self):
        rolled = _tracking().aggregate(level="sequence", how={"time_s": "min", "width": "max"})
        assert {"time_s_min", "width_max"} <= set(rolled.factor_names)

    def test_a_mapping_must_cover_every_factor_named_beside_it(self):
        with pytest.raises(ValueError, match=r"\['width'\] are not among its keys"):
            _tracking().aggregate("width", level="sequence", how={"time_s": "min"})

    def test_an_aggregator_carries_the_modifiers_the_keywords_do_not(self):
        declaration = Aggregator("mean", "instance", "sequence", ("width",), unique_by="unit", via="track")
        rolled = _tracking().aggregate(declaration)
        assert "width_mean_via_track" in rolled.factor_names

    def test_a_call_of_only_aggregators_needs_no_level(self):
        rolled = _tracking().aggregate(Aggregator("mean", "unit", "sequence", ("time_s",)))
        assert "time_s_mean" in rolled.factor_names

    def test_a_destination_level_is_required_otherwise(self):
        with pytest.raises(ValueError, match="needs a destination level"):
            _tracking().aggregate("time_s", how="mean")

    def test_factors_at_two_levels_land_in_one_call(self):
        metadata = _tracking(track={"speed": np.arange(5.0)})
        rolled = metadata.aggregate("time_s", "speed", level="sequence", how="mean")
        assert {"time_s_mean", "speed_mean"} <= set(rolled.factor_names)

    def test_the_source_is_not_mutated(self):
        metadata = _tracking()
        before = set(metadata.factor_names)
        metadata.aggregate("time_s", level="sequence", how="mean")
        assert set(metadata.factor_names) == before

    def test_the_factor_records_the_level_it_was_rolled_up_from(self):
        rolled = _tracking().aggregate("time_s", level="sequence", how="mean")
        assert rolled.factor_info["time_s_mean"].aggregated_from == "unit"


@pytest.mark.required
class TestIdentityElements:
    """A destination with nothing beneath it: zero where that is a measurement, null where it is not."""

    @staticmethod
    def _empty_frames():
        """A tracking dataset whose first two frames hold no detections at all."""
        metadata = Metadata(_mot_dataset([[0, 0], [2]]))
        metadata._structure()
        metadata.add_factors({"area": np.arange(2.0)}, level="instance")
        return metadata

    def test_a_count_of_nothing_is_zero(self):
        rolled = self._empty_frames().aggregate("area", level="unit", how="count")
        assert rolled.at("unit").rows_at("unit")["area_count"].to_list() == [0, 0, 2]

    def test_a_mean_of_nothing_is_null(self):
        rolled = self._empty_frames().aggregate("area", level="unit", how="mean")
        assert rolled.at("unit").rows_at("unit")["area_mean"].to_list() == [None, None, 0.5]

    def test_the_expression_form_reaches_the_same_answer_when_told_the_same_thing(self):
        """Decision 2: no difference that cannot be expressed, only a default one form can know.

        ``agg`` cannot infer that a count of no rows is zero, because an arbitrary
        expression has no identity to infer. Given it, the two agree exactly.
        """
        metadata = self._empty_frames()
        named = metadata.aggregate("area", level="unit", how="count")
        spelled = metadata.agg("instance", "unit", pl.col("area").count().alias("area_count"), empty=0)
        assert (
            named.at("unit").rows_at("unit")["area_count"].to_list()
            == spelled.at("unit").rows_at("unit")["area_count"].to_list()
        )

    def test_without_an_identity_the_expression_form_leaves_it_null(self):
        metadata = self._empty_frames()
        spelled = metadata.agg("instance", "unit", pl.col("area").count().alias("n"))
        assert spelled.at("unit").rows_at("unit")["n"].to_list() == [None, None, 2]

    def test_an_identity_reaches_a_list_valued_column_too(self):
        """The branch written for the nested case was the one case it could not serve.

        Iterating a nested Series yields inner ``Series``, so mixing those with a plain
        Python ``empty`` gave polars a list of two kinds and it refused to build one.
        """
        metadata = self._empty_frames()
        spelled = metadata.agg("instance", "unit", pl.col("area").alias("areas"), empty=[])
        assert spelled.at("unit").rows_at("unit")["areas"].to_list() == [[], [], [0.0, 1.0]]


@pytest.mark.required
class TestCoverage:
    """How few recorded values is too few, and how that differs from having none at all."""

    @staticmethod
    def _partly_recorded():
        """Five frames; the third frame's two detections record only one area between them.

        Frame 1 holds no detections at all, which is a different kind of nothing from frame
        2's half-recorded pair and has to answer differently.
        """
        metadata = Metadata(_mot_dataset([[2, 0, [1, -1]], [[0, 2], [1]]]))
        metadata._structure()
        metadata.add_factors({"area": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])}, level="instance")
        return metadata

    def _mean(self, min_coverage):
        rolled = self._partly_recorded().aggregate(
            Aggregator("mean", "instance", "unit", ("area",), min_coverage=min_coverage),
        )
        return rolled.at("unit").rows_at("unit")["area_mean"].to_list()

    def test_all_or_nothing_nulls_a_partly_recorded_destination(self):
        assert self._mean(1.0) == [1.5, None, None, 5.5, 7.0]

    def test_a_threshold_the_destination_meets_lets_it_answer(self):
        assert self._mean(0.5) == [1.5, None, 4.0, 5.5, 7.0]

    def test_no_threshold_summarizes_whatever_was_recorded(self):
        assert self._mean(0.0) == [1.5, None, 4.0, 5.5, 7.0]

    def test_a_reduction_about_missingness_ignores_the_threshold(self):
        """Counting the values present is the right answer however many were absent."""
        rolled = self._partly_recorded().aggregate(
            Aggregator("count", "instance", "unit", ("area",), min_coverage=1.0),
        )
        assert rolled.at("unit").rows_at("unit")["area_count"].to_list() == [2, 0, 1, 2, 1]

    def test_an_under_covered_destination_is_null_and_an_empty_one_is_the_identity(self):
        """The two nothings are not the same, and only one of them earns an identity element.

        A frame with no detections measured nothing, so ``sum`` is zero. A frame whose
        detections did not record their area measured *something* and did not record it, so
        there is no total to give.
        """
        rolled = self._partly_recorded().aggregate(
            Aggregator("sum", "instance", "unit", ("area",), min_coverage=1.0),
        )
        assert rolled.at("unit").rows_at("unit")["area_sum"].to_list() == [3.0, 0.0, None, 11.0, 7.0]

    def test_a_threshold_outside_zero_to_one_is_rejected(self):
        with pytest.raises(ValueError, match=r"lies in \[0, 1\]"):
            Aggregator("mean", "unit", "sequence", min_coverage=1.5)

    def test_the_expression_form_keeps_summarizing_what_is_there(self):
        """``agg`` defaults to no threshold, so what it answered before, it still answers."""
        metadata = self._partly_recorded()
        spelled = metadata.agg("instance", "unit", pl.col("area").mean().alias("area_mean"))
        assert spelled.at("unit").rows_at("unit")["area_mean"].to_list() == [1.5, None, 4.0, 5.5, 7.0]

    def test_the_expression_form_can_ask_for_the_same_threshold(self):
        metadata = self._partly_recorded()
        spelled = metadata.agg("instance", "unit", pl.col("area").mean().alias("area_mean"), min_coverage=1.0)
        assert spelled.at("unit").rows_at("unit")["area_mean"].to_list() == self._mean(1.0)


@pytest.mark.required
class TestNaNIsAMissingValue:
    """DataEval spells an unrecorded number NaN; polars calls it a value. Roll-ups follow DataEval."""

    @staticmethod
    def _one_missing():
        metadata = Metadata(_mot_dataset([[3, 3], [2]]))
        metadata._structure()
        height = metadata.level_counts["instance"]
        values = np.arange(1.0, height + 1.0)
        values[0] = np.nan
        metadata.add_factors({"area": values}, level="instance")
        return metadata

    def test_one_unrecorded_value_does_not_poison_the_summary(self):
        """The failure this prevents: a whole sequence reading NaN because one frame lacked a value."""
        rolled = self._one_missing().aggregate(
            Aggregator("mean", "instance", "sequence", ("area",), min_coverage=0.0),
        )
        answers = rolled.at("sequence").rows_at("sequence")["area_mean"].to_list()
        assert all(answer is not None and not np.isnan(answer) for answer in answers)

    def test_an_unrecorded_value_is_not_counted_as_present(self):
        rolled = self._one_missing().aggregate(Aggregator("count", "instance", "sequence", ("area",)))
        counted = sum(rolled.at("sequence").rows_at("sequence")["area_count"].to_list())
        assert counted == self._one_missing().level_counts["instance"] - 1


@pytest.mark.required
class TestTheReport:
    """A rolled-up column cannot explain its own nulls, so the counts stay beside it."""

    @staticmethod
    def _rolled(**kwargs):
        metadata = Metadata(_mot_dataset([[2, 0, [1, -1]], [[0, 2], [1]]]))
        metadata._structure()
        metadata.add_factors({"area": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])}, level="instance")
        return metadata.aggregate(Aggregator("mean", "instance", "unit", ("area",), **kwargs))

    def test_the_record_names_what_was_done_and_where(self):
        record = self._rolled().last_aggregation[0]
        assert (record.source, record.target, record.how, record.via) == ("instance", "unit", "mean", None)

    def test_the_record_names_the_column_it_produced(self):
        assert self._rolled().last_aggregation[0].outputs == ("area_mean",)

    def test_the_lowest_coverage_seen_is_reported_even_when_no_threshold_acted(self):
        """The number is what tells a caller which threshold would have been answerable."""
        assert self._rolled(min_coverage=0.0).last_aggregation[0].coverage_of("area_mean") == 0.5

    def test_the_destinations_nulled_for_coverage_are_counted(self):
        assert self._rolled(min_coverage=1.0).last_aggregation[0].uncovered == (1,)

    def test_nothing_is_counted_as_uncovered_when_no_threshold_acted(self):
        assert self._rolled(min_coverage=0.0).last_aggregation[0].uncovered == (0,)

    def test_destinations_with_nothing_beneath_them_are_counted_apart(self):
        """One frame holds no detections; that is a different null from an under-covered one."""
        record = self._rolled(min_coverage=1.0).last_aggregation[0]
        assert (record.childless, record.uncovered) == (1, (1,))

    def test_a_complete_route_excludes_nobody(self):
        record = self._rolled().last_aggregation[0]
        assert (record.no_ancestor, record.took_part) == (0, 7)

    def test_a_routed_roll_up_counts_the_rows_the_branch_does_not_reach(self):
        metadata = Metadata(_mot_dataset([[2, 0, [1, -1]], [[0, 2], [1]]]))
        metadata._structure()
        metadata.add_factors({"area": np.arange(7.0)}, level="instance")
        rolled = metadata.aggregate(Aggregator("mean", "instance", "sequence", ("area",), via="track"))
        assert rolled.last_aggregation[0].no_ancestor == 1

    def test_a_metadata_no_roll_up_produced_reports_nothing(self):
        assert _tracking().last_aggregation == ()

    def test_the_source_keeps_its_own_report(self):
        metadata = _tracking()
        metadata.aggregate("time_s", level="sequence", how="mean")
        assert metadata.last_aggregation == ()

    def test_two_source_levels_report_one_record_each(self):
        metadata = _tracking(track={"speed": np.arange(5.0)})
        rolled = metadata.aggregate("time_s", "speed", level="sequence", how="mean")
        assert {record.source for record in rolled.last_aggregation} == {"unit", "track"}

    def test_the_record_names_the_column_after_a_collision_rename(self):
        """A record naming the name that was asked for would point at no column."""
        metadata = _tracking()
        once = metadata.aggregate("time_s", level="sequence", how="mean")
        twice = once.aggregate("time_s", level="sequence", how="mean")
        assert twice.last_aggregation[0].outputs == ("time_s_mean_agg",)

    def test_an_unknown_output_has_no_coverage_to_give(self):
        with pytest.raises(ValueError, match="is not one of this roll-up's outputs"):
            self._rolled().last_aggregation[0].coverage_of("nope")

    def test_the_reason_for_a_null_column_is_announced(self, caplog):
        caplog.set_level(logging.INFO, logger="dataeval.metadata")
        self._rolled(min_coverage=1.0)
        assert "min_coverage at or below that" in caplog.text


@pytest.mark.required
class TestReviewRegressions:
    """Cases a review found: each one answered wrongly or crashed before it was pinned."""

    @staticmethod
    def _untracked():
        """Two sequences whose every detection is untracked, so no track row exists."""
        metadata = Metadata(_mot_dataset([[[-1, -1]], [[-1]]]))
        metadata._structure()
        metadata.add_factors({"area": np.arange(3.0)}, level="instance")
        return metadata

    def test_a_route_that_reaches_nothing_answers_null_rather_than_raising(self):
        """The default named surface, on the case ``via`` exists for."""
        rolled = self._untracked().aggregate(Aggregator("mean", "instance", "sequence", ("area",), via="track"))
        assert rolled.at("sequence").rows_at("sequence")["area_mean_via_track"].to_list() == [None, None]

    def test_a_filter_that_empties_the_source_level_answers_null(self):
        metadata = self._untracked().where(pl.col("area") > 100.0, level="instance")
        rolled = metadata.aggregate("area", level="sequence", how="mean")
        assert set(rolled.at("sequence").rows_at("sequence")["area_mean"].to_list()) == {None}

    def test_an_identity_reaches_only_the_destinations_that_had_nothing(self):
        """A column with no dtype to widen is still filled positionally, not wholesale."""
        metadata = Metadata(_mot_dataset([[2, 0], [2]]))
        metadata._structure()
        rolled = metadata.agg("instance", "unit", pl.lit(None).alias("z"), empty=0)
        assert rolled.at("unit").rows_at("unit")["z"].to_list() == [None, 0, None]

    def test_mode_ignores_the_missing_value_rather_than_answering_it(self):
        """polars counts a null as a value and sorts it first, so it used to win every tie."""
        metadata = Metadata(_mot_dataset([[2, 0], [2]]))
        metadata._structure()
        metadata.add_factors({"area": np.array([1.0, np.nan, 5.0, 5.0])}, level="instance")
        rolled = metadata.aggregate(Aggregator("mode", "instance", "unit", ("area",), min_coverage=0.0))
        assert rolled.at("unit").rows_at("unit")["area_mode"].to_list() == [1.0, None, 5.0]

    def test_n_unique_does_not_count_absence_as_a_distinct_value(self):
        """It declares itself unharmed by missing values, so it must not be inflated by them."""
        metadata = Metadata(_mot_dataset([[2, 0], [2]]))
        metadata._structure()
        metadata.add_factors({"area": np.array([1.0, np.nan, 5.0, 5.0])}, level="instance")
        rolled = metadata.aggregate(Aggregator("n_unique", "instance", "unit", ("area",)))
        assert rolled.at("unit").rows_at("unit")["area_n_unique"].to_list() == [1, 0, 1]

    def test_a_declared_aggregator_is_refused_for_what_the_named_form_refuses(self):
        """The checks must not be a property of which surface the caller entered by."""
        metadata = _tracking(instance={"kind": np.array(["a"] * 7)})
        with pytest.raises(ValueError, match="'mean' does not apply to 'kind'"):
            metadata.aggregate(Aggregator("mean", "instance", "unit", ("kind",)))

    def test_a_declared_aggregator_naming_no_such_factor_is_refused(self):
        with pytest.raises(ValueError, match="'nope' is not a factor of this metadata"):
            _tracking().aggregate(Aggregator("mean", "instance", "unit", ("nope",)))

    def test_a_declared_aggregator_keeps_its_declared_provenance(self):
        metadata = _tracking()
        stated = Aggregator("mean", "unit", "sequence", ("time_s",))
        (one,) = TestResolution._resolve(stated, metadata)
        assert one.provenance == "declared"

    @pytest.mark.parametrize("via", ["bogus", "track", "unit"])
    def test_a_route_that_does_not_exist_is_refused_with_no_dataset(self, via):
        """The declaration's whole point is being wrong before any data is in hand."""
        with pytest.raises(ValueError, match="Unknown level|No route from|every route starts at"):
            Aggregator("mean", "unit", "sequence", via=via).validate(SCHEMA)

    def test_a_route_that_does_exist_validates_against_a_bare_schema(self):
        Aggregator("mean", "instance", "sequence", via="track").validate(SCHEMA)

    @pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan")])
    def test_a_threshold_that_is_not_a_share_is_refused_on_both_surfaces(self, threshold):
        with pytest.raises(ValueError, match=r"lies in \[0, 1\]"):
            Aggregator("mean", "unit", "sequence", min_coverage=threshold)
        with pytest.raises(ValueError, match=r"lies in \[0, 1\]"):
            _tracking().agg("unit", "sequence", pl.col("time_s").mean(), min_coverage=threshold)

    def test_took_part_counts_the_rows_that_were_summarized(self):
        """``unique_by`` drops rows so they are not summarized twice; they did not take part."""
        metadata = _tracking()
        rolled = metadata.aggregate(Aggregator("mean", "instance", "sequence", ("time_s",), unique_by="unit"))
        record = rolled.last_aggregation[0]
        # Seven detections collapse to one row per frame that holds any — four of the five
        # frames — so counting the seven would sit on a different denominator from the
        # coverage measured in the same pass.
        assert (record.took_part, metadata.level_counts["instance"]) == (4, 7)


@pytest.mark.required
class TestTemporalReductions:
    """A function of the ordered series, not of the bag of values."""

    @staticmethod
    def _sequence(values, shape=None):
        """One video whose frames carry ``values``, evenly spaced in time."""
        metadata = Metadata(_mot_dataset(shape or [[1] * len(values)]))
        metadata._structure()
        metadata.add_factors({"b": np.asarray(values, dtype=np.float64)}, level="unit")
        return metadata

    def _roll(self, values, how, **kwargs):
        metadata = self._sequence(values)
        rolled = metadata.aggregate(Aggregator(how, "unit", "sequence", ("b",), **kwargs))
        return rolled.at("sequence").rows_at("sequence")[f"b_{how}"].to_list()

    def test_variability_separates_drift_from_jitter_where_variance_cannot(self):
        """The reason this reduction exists, asserted directly.

        A slow ramp and an alternating signal are built here to have the same variance; what
        distinguishes them is how far the value moves between one reading and the next.
        """
        drift = [0.0, 1.0, 2.0, 3.0]
        swing = float(np.sqrt(np.var(drift)))
        jitter = [-swing, swing, -swing, swing]
        assert np.isclose(np.var(drift), np.var(jitter))
        assert self._roll(drift, "variability")[0] < self._roll(jitter, "variability")[0]

    def test_variance_cannot_and_is_left_positional(self):
        """``var`` reads a bag of values, so it is not the temporal question and does not pretend to be."""
        assert lookup("var").kind == "positional"
        assert lookup("variability").kind == "temporal"

    def test_trend_reads_the_slope_against_the_ordering_key(self):
        """Per second, not per frame: the fixture's frames are half a second apart."""
        assert self._roll([0.0, 1.0, 2.0, 3.0], "trend") == [2.0]
        assert self._roll([3.0, 2.0, 1.0, 0.0], "trend") == [-2.0]

    def test_trend_is_flat_for_a_signal_that_only_moves(self):
        assert self._roll([0.0, 3.0, 3.0, 0.0], "trend") == [0.0]

    def test_changes_counts_transitions(self):
        assert self._roll([1.0, 1.0, 2.0, 2.0, 1.0], "changes") == [2]

    def test_changes_answers_zero_for_a_signal_that_never_moves(self):
        assert self._roll([1.0, 1.0, 1.0], "changes") == [0]

    def test_longest_run_finds_the_longest_stretch_of_one_value(self):
        assert self._roll([1.0, 2.0, 2.0, 2.0, 3.0], "longest_run") == [3]

    def test_a_tolerance_admits_values_close_enough_to_call_unchanged(self):
        values = [0.10, 0.11, 0.12, 0.90]
        assert self._roll(values, "longest_run") == [1]
        assert self._roll(values, "longest_run", options={"tolerance": ("constant", (None, 0.05))}) == [3]

    def _fitted(self, values, spec):
        """The absolute tolerance a spec resolves to against one series."""
        metadata = self._sequence(values)
        declared = Aggregator("longest_run", "unit", "sequence", ("b",), options={"tolerance": spec})
        (one,) = TestResolution._resolve(declared, metadata)
        (ready,) = metadata._with_tolerance(one)
        return ready.options["tolerance"]

    def test_a_relative_tolerance_scales_with_the_changes_observed(self):
        """The point of the relative form: the same spec transfers where an absolute number does not.

        An absolute tolerance is a constant somebody measured on one dataset, and it means
        something different on the next. A relative one is read off the changes each dataset
        actually shows, so the same declaration asks the same question of both.
        """
        spec = ("iqr", (None, 1.5))
        small = self._fitted([0.0, 0.1, 0.2, 0.5], spec)
        large = self._fitted([0.0, 10.0, 20.0, 50.0], spec)
        assert large == pytest.approx(small * 100)

    def test_an_absolute_tolerance_does_not_scale(self):
        spec = ("constant", (None, 0.05))
        assert self._fitted([0.0, 0.1, 0.2, 0.5], spec) == self._fitted([0.0, 10.0, 20.0, 50.0], spec)

    def test_a_relative_tolerance_with_nothing_to_be_relative_to_is_refused(self):
        """A spread of zero gives a threshold no bound, which is a refusal rather than an answer."""
        with pytest.raises(ValueError, match="says nothing about which"):
            self._roll([0.0, 1.0, 2.0, 3.0, 4.0, 40.0], "longest_run", options={"tolerance": ("iqr", (None, 1.5))})

    def test_a_resolved_tolerance_records_the_number_it_fitted(self):
        metadata = self._sequence([0.10, 0.11, 0.12, 0.90])
        declared = Aggregator("longest_run", "unit", "sequence", ("b",), options={"tolerance": ("iqr", (None, 1.5))})
        (one,) = TestResolution._resolve(declared, metadata)
        (ready,) = metadata._with_tolerance(one)
        assert isinstance(ready.options["tolerance"], float)
        assert ready.provenance == "derived"

    def test_a_bare_number_tolerance_is_refused_with_both_spellings(self):
        """Under ThresholdLike a bare number is a multiplier, so the naive reading is the wrong one."""
        with pytest.raises(ValueError, match=r"\('constant', \(None, 0.05\)\).*\('iqr', \(None, 0.05\)\)"):
            self._roll([0.1, 0.2], "longest_run", options={"tolerance": 0.05})

    def test_a_two_sided_tolerance_is_refused(self):
        with pytest.raises(ValueError, match="has only one side"):
            self._roll([0.1, 0.2], "longest_run", options={"tolerance": ("constant", (0.01, 0.05))})

    def test_an_option_the_reduction_does_not_read_is_refused(self):
        with pytest.raises(ValueError, match="'mean' takes no options"):
            self._roll([0.1, 0.2], "mean", options={"tolerance": ("constant", (None, 0.05))})


@pytest.mark.required
class TestTemporalOrdering:
    """A temporal reduction needs an ordering, and will not invent one from row order."""

    @staticmethod
    def _tracking():
        metadata = _tracking(track={"speed": np.arange(5.0)})
        metadata.add_factors({"brightness": np.arange(5.0)}, level="unit")
        return metadata

    def test_the_ordering_is_inferred_from_the_source_level(self):
        (one,) = TestResolution._resolve(Aggregator("trend", None, "sequence", ("brightness",)), self._tracking())
        assert one.rolls_by == "time_s"

    def test_a_level_that_carries_no_ordering_refuses_the_reduction(self):
        """``track`` under ``sequence`` is the case: tracks are a set, not a series."""
        with pytest.raises(ValueError, match="carries no ordering to read them in"):
            self._tracking().aggregate("speed", level="sequence", how="trend")

    def test_an_ordering_the_source_does_not_hold_itself_is_refused(self):
        """An ordering read down from an ancestor repeats across the fan-out, so it orders nothing."""
        with pytest.raises(ValueError, match="is not a column 'instance' holds itself"):
            self._tracking().aggregate(Aggregator("trend", "instance", "track", ("brightness",), order_by="time_s"))

    def test_a_positional_reduction_needs_no_ordering(self):
        (one,) = TestResolution._resolve(Aggregator("mean", None, "sequence", ("speed",)), self._tracking())
        assert one.order_by is None

    def test_an_unresolved_aggregator_names_no_ordering_to_give(self):
        with pytest.raises(ValueError, match="names no ordering column"):
            Aggregator("trend", "unit", "sequence").rolls_by  # noqa: B018


@pytest.mark.required
class TestGapsAreReported:
    """Filtering a series leaves a gapped one, and the answer looks the same either way."""

    @staticmethod
    def _alternating():
        metadata = Metadata(_mot_dataset([[1, 1, 1, 1]]))
        metadata._structure()
        metadata.add_factors({"b": np.array([0.0, 1.0, 0.0, 1.0])}, level="unit")
        return metadata

    def test_an_evenly_spaced_series_reports_no_gaps(self):
        rolled = self._alternating().aggregate(Aggregator("changes", "unit", "sequence", ("b",)))
        assert rolled.last_aggregation[0].gaps == 0

    def test_a_filtered_series_reports_the_steps_the_filter_left(self):
        """The trap: the answer is smaller and nothing about it says a reading is missing."""
        filtered = self._alternating().where(pl.col("unit_index") != 1, level="unit")
        rolled = filtered.aggregate(Aggregator("changes", "unit", "sequence", ("b",)))
        assert rolled.at("sequence").rows_at("sequence")["b_changes"].to_list() == [1]
        assert rolled.last_aggregation[0].gaps == 1

    def test_a_positional_reduction_reports_no_gaps_at_all(self):
        """It reads no ordering, so it cannot see one — reporting a number would suggest it did."""
        rolled = self._alternating().aggregate(Aggregator("mean", "unit", "sequence", ("b",)))
        assert rolled.last_aggregation[0].gaps == 0

    def test_the_reductions_a_gap_distorts_say_so(self, caplog):
        caplog.set_level(logging.INFO, logger="dataeval.metadata")
        filtered = self._alternating().where(pl.col("unit_index") != 1, level="unit")
        filtered.aggregate(Aggregator("changes", "unit", "sequence", ("b",)))
        assert "not evenly spaced" in caplog.text

    def test_a_rate_is_not_distorted_by_a_gap_and_says_nothing(self, caplog):
        """``variability`` divides by the key delta, so a missing reading changes nothing it reports."""
        caplog.set_level(logging.INFO, logger="dataeval.metadata")
        filtered = self._alternating().where(pl.col("unit_index") != 1, level="unit")
        filtered.aggregate(Aggregator("variability", "unit", "sequence", ("b",)))
        assert "not evenly spaced" not in caplog.text


@pytest.mark.required
class TestAKeyThatRepeatsIsNotAnInfinitelyFastSeries:
    """Two rows sharing an ordering key have no time between them, not zero time."""

    @staticmethod
    def _detections():
        """Frames holding several detections, so the inferred ordering repeats per frame."""
        metadata = Metadata(_mot_dataset([[2, 2]]))
        metadata._structure()
        metadata.add_factors(
            {"area": np.arange(metadata.level_counts["instance"], dtype=np.float64) * 3.0}, level="instance"
        )
        return metadata

    def test_a_repeated_key_does_not_make_the_whole_answer_infinite(self):
        """Dividing by the zero step answered ``inf`` for every sequence in the dataset."""
        rolled = self._detections().aggregate("area", level="sequence", how="variability")
        answers = rolled.at("sequence").rows_at("sequence")["area_variability"].to_list()
        assert all(np.isfinite(value) for value in answers)

    def test_a_repeated_key_does_not_read_as_a_gap_at_every_step(self):
        """The tightest step is the sampling interval, and a duplicate is not one."""
        rolled = self._detections().aggregate("area", level="sequence", how="variability")
        assert rolled.last_aggregation[0].gaps == 0


@pytest.mark.required
class TestAMissingValueIsReadTheSameWayByEveryReduction:
    """``changes`` and ``longest_run`` both read the values that were recorded."""

    @staticmethod
    def _gapped(values):
        metadata = Metadata(_mot_dataset([[1] * len(values)]))
        metadata._structure()
        metadata.add_factors({"b": np.asarray(values, dtype=np.float64)}, level="unit")
        return metadata

    def _roll(self, values, how):
        rolled = self._gapped(values).aggregate(
            Aggregator(how, "unit", "sequence", ("b",), min_coverage=0.0),
        )
        return rolled.at("sequence").rows_at("sequence")[f"b_{how}"].to_list()

    def test_a_change_across_an_unrecorded_reading_is_not_concealed(self):
        """Both comparisons touching the null answered null and the sum skipped both."""
        assert self._roll([1.0, 1.0, np.nan, 2.0, 2.0], "changes") == [1]

    def test_a_run_is_not_broken_by_an_unrecorded_reading(self):
        """``changes`` read straight through it while ``longest_run`` called it a break."""
        assert self._roll([1.0, 1.0, np.nan, 1.0, 1.0], "longest_run") == [4]

    def test_the_two_agree_about_where_a_run_ends(self):
        values = [1.0, 1.0, np.nan, 2.0, 2.0]
        assert self._roll(values, "changes") == [1]
        assert self._roll(values, "longest_run") == [2]

    def test_a_trend_reads_the_keys_of_the_rows_it_actually_paired(self):
        """``cov`` drops the unpaired rows; a variance over all of them flattened the slope."""
        values = [0.0, 1.0, 2.0, 3.0, np.nan]
        metadata = self._gapped(values)
        times = np.asarray(metadata.rows_at("unit")["time_s"].to_list(), dtype=np.float64)
        recorded = ~np.isnan(np.asarray(values))
        expected = np.polyfit(times[recorded], np.asarray(values)[recorded], 1)[0]
        rolled = metadata.aggregate(Aggregator("trend", "unit", "sequence", ("b",), min_coverage=0.0))
        assert rolled.at("sequence").rows_at("sequence")["b_trend"].to_list() == pytest.approx([expected])


@pytest.mark.required
class TestAFittedToleranceIsReplayedRatherThanRefitted:
    """A resolved aggregator carries the number, which is the whole point of resolving it."""

    @staticmethod
    def _sequence(values):
        metadata = Metadata(_mot_dataset([[1] * len(values)]))
        metadata._structure()
        metadata.add_factors({"b": np.asarray(values, dtype=np.float64)}, level="unit")
        return metadata

    def _fitted(self):
        metadata = self._sequence([0.10, 0.11, 0.12, 0.90])
        declared = Aggregator("longest_run", "unit", "sequence", ("b",), options={"tolerance": ("iqr", (None, 1.5))})
        (one,) = TestResolution._resolve(declared, metadata)
        (ready,) = metadata._with_tolerance(one)
        return ready

    def test_replaying_it_against_a_second_dataset_reuses_the_number(self):
        """Re-resolving read the fitted distance as a fresh threshold spec and refused it."""
        fitted = self._fitted()
        rolled = self._sequence([0.10, 0.50, 0.90, 1.30]).aggregate(fitted)
        assert rolled.at("sequence").rows_at("sequence")["b_longest_run"].to_list() == [4]

    def test_resolving_a_fit_again_leaves_it_a_fit(self):
        """Relabelling it ``declared`` claimed a caller wrote what a resolution measured."""
        fitted = self._fitted()
        metadata = self._sequence([0.10, 0.50, 0.90, 1.30])
        (again,) = TestResolution._resolve(fitted, metadata)
        assert again.provenance == "derived"
        assert again.options["tolerance"] == fitted.options["tolerance"]


@pytest.mark.required
class TestARuleSkipsALevelItCannotRead:
    """A named factor is a request; an empty factor set is a rule, and a rule selects."""

    @staticmethod
    def _tracked():
        metadata = Metadata(_mot_dataset([[2, 1], [1]]))
        metadata._structure()
        metadata.add_factors({"bright": np.arange(metadata.level_counts["unit"], dtype=np.float64)}, level="unit")
        return metadata

    def test_a_level_with_no_ordering_is_passed_over_rather_than_refusing_the_call(self):
        """``track`` carries no ordering, and one such level took the whole call down."""
        rolled = self._tracked().aggregate(level="sequence", how="variability")
        assert "bright_variability" in rolled.at("sequence").rows_at("sequence").columns

    def test_the_level_it_passed_over_is_named(self, caplog):
        caplog.set_level(logging.INFO, logger="dataeval.metadata")
        self._tracked().aggregate(level="sequence", how="variability")
        assert "carries no ordering" in caplog.text

    def test_a_factor_the_caller_names_at_such_a_level_is_still_refused(self):
        """They asked for that factor, so the level's silence is an answer they need."""
        metadata = self._tracked()
        with pytest.raises(ValueError, match="carries no ordering"):
            metadata.aggregate(Aggregator("variability", "track", "sequence", ("track_length",)))


@pytest.mark.required
class TestAggregatorIsAValue:
    def test_it_can_be_put_in_a_set(self):
        """``frozen=True`` generates a hash, and the mapping field made it raise at the call."""
        one = Aggregator("mean", "unit", "sequence", ("b",), options={"tolerance": ("iqr", (None, 1.5))})
        two = Aggregator("mean", "unit", "sequence", ("b",), options={"tolerance": ("iqr", (None, 1.5))})
        assert len({one, two}) == 1


@pytest.mark.required
class TestARowWithNoOrderingKeyHasNoPlaceInTheSeries:
    """Both ends of a series are a fabrication for a reading nothing placed in time."""

    @staticmethod
    def _counts(untimed: bool):
        """Four frames holding 1, 2, 2, 1 detections, optionally with the last one untimed."""
        from tests.metadata.test_structurers import _undeclared

        dataset = _mot_dataset([[1, 2, 2, 1]])
        if untimed:
            dataset = _undeclared(dataset, 0, 3, "time_s")
        metadata = Metadata(dataset, partial_factors=True)
        return metadata.agg("instance", "unit", pl.len().alias("n"))

    def _changes(self, untimed: bool):
        rolled = self._counts(untimed).aggregate(Aggregator("changes", "unit", "sequence", ("n",), min_coverage=0.0))
        return rolled.at("sequence").rows_at("sequence")["n_changes"].to_list()

    def test_a_fully_timed_series_reads_in_time_order(self):
        assert self._changes(untimed=False) == [2]

    def test_an_untimed_reading_is_not_read_as_the_earliest_one(self):
        """polars sorts nulls first, so the untimed frame joined the front of the series and
        its value merged with the first reading: 1, 2, 2, 1 reported one change, not two."""
        assert self._changes(untimed=True) == [1]

    def test_it_answers_over_the_readings_that_do_have_a_position(self):
        """1, 2, 2 is one change — the honest answer over what was placed in time."""
        counts = self._counts(untimed=True).at("unit").rows_at("unit")
        assert counts["n"].to_list() == [1, 2, 2, 1]
        assert np.isnan(counts["time_s"].to_list()[3])


@pytest.mark.required
class TestAToleranceNeedsADistanceToMeasure:
    """A tolerance asks how far apart two readings are, and not every value type answers."""

    def test_a_tolerance_on_a_string_factor_is_refused_by_name(self):
        """polars answered with 'sub operation not supported for dtypes str and str' and a
        dump of the internal sort expression — the raw-polars leak names exist to prevent."""
        metadata = Metadata(_mot_dataset([[1, 1], [1, 1]], [{"w": "a"}, {"w": "b"}]))
        options = {"tolerance": ("constant", (None, 0.5))}
        declared = Aggregator("longest_run", "unit", "sequence", ("w",), options=options)
        with pytest.raises(ValueError, match="needs values that can be subtracted"):
            metadata.aggregate(declared)

    def test_a_factor_that_is_its_own_ordering_is_read_once(self):
        """Value and key resolved to one column, and polars refused the duplicate."""
        metadata = Metadata(_mot_dataset([[2, 1], [1]]))
        declared = Aggregator(
            "longest_run", "unit", "sequence", ("time_s",), options={"tolerance": ("constant", (None, 0.1))}
        )
        assert metadata.aggregate(declared).at("sequence").rows_at("sequence")["time_s_longest_run"].to_list() == [1, 1]

    @pytest.mark.parametrize("spec", [("iqr", (None, 1.5)), ("modzscore", (None, 3.0)), ("zscore", (None, 2.0))])
    def test_a_relative_tolerance_with_nothing_to_measure_says_so(self, spec):
        """`iqr` indexed into an empty array and raised IndexError out of the threshold; the
        z-scores resolved to NaN, which no comparison is ever true against — so every
        destination silently reported its whole length as one unbroken run."""
        metadata = Metadata(_mot_dataset([[1], [1]]))
        declared = Aggregator("longest_run", "unit", "sequence", ("width",), options={"tolerance": spec})
        with pytest.raises(ValueError, match="shows none to measure|says nothing about which"):
            metadata.aggregate(declared)


@pytest.mark.required
class TestAnAggregatorTakesTheOptionsItDocuments:
    def test_options_reads_back_as_a_mapping_whatever_was_passed(self):
        """The docstring offered ``None`` and ``__post_init__`` raised on it, so a caller
        following the signature got a TypeError out of the constructor."""
        assert Aggregator("mean", "instance", "unit", options=None).options == {}  # type: ignore[arg-type]
        assert Aggregator("mean", "instance", "unit").options == {}
