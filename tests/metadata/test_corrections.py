"""Repairing a column the walk could not read, and keeping the repair.

The claim under test is that a correction is a *declaration*: applied to the values the
walk kept rather than to a second pass over the dataset, replacing rather than accumulating,
and recorded so that it survives an archive and carries to the next dataset.
"""

from datetime import date, datetime

import numpy as np
import pytest

from dataeval import Metadata
from dataeval._metadata._corrections import apply
from dataeval._metadata._metadata import _kinds
from dataeval.types import ParseDateTime, ParseValue, Remap, Rescale, Unusable
from tests.metadata.test_structurers import _mot_dataset


def _md(values, **kwargs):
    return Metadata(_mot_dataset([[1]] * len(values), [{"d": value} for value in values]), **kwargs)


@pytest.mark.required
class TestApplyingCorrections:
    """The value rules, exercised without a dataset in the way."""

    def test_a_named_value_takes_its_replacement(self):
        assert apply(["N", "NE"], [Remap("d", {"N": 0, "NE": 45})]) == [0, 45]

    def test_a_range_key_replaces_a_whole_band(self):
        """Which is how a sentinel is retired, and what a rescale cannot express."""
        assert apply([-500, 5], [Remap("d", {(-1000, 0): -1})]) == [-1, 5]

    def test_the_catch_all_answers_for_everything_unnamed(self):
        assert apply(["N", "SSE", 1.0], [Remap("d", {"N": 0, None: -1})]) == [0, -1, -1]

    def test_without_a_catch_all_an_unnamed_value_is_unchanged(self):
        """A partial mapping is a partial mapping."""
        assert apply(["N", "SSE"], [Remap("d", {"N": 0})]) == [0, "SSE"]

    @pytest.mark.parametrize("absent", [None, float("nan")])
    def test_an_absent_row_is_never_matched_catch_all_included(self, absent):
        """Absence is not a value; the reserved missing code is what says so."""
        (result,) = apply([absent], [Remap("d", {None: -1})])
        assert result is None or np.isnan(result)

    def test_a_named_value_beats_a_range_that_contains_it(self):
        assert apply([5], [Remap("d", {5: "exact", (0, 10): "range"})]) == ["exact"]

    def test_a_boolean_is_not_the_number_one(self):
        assert apply([True, 1], [Remap("d", {1: "one"})]) == [True, "one"]

    def test_a_rescale_converts_the_values_in_its_range(self):
        assert apply([500, 1500], [Rescale("d", over=(0, 1000), multiply=0.3048)]) == [152.4, 1500]

    def test_only_the_first_matching_rescale_applies(self):
        """Overlapping ranges resolve to the one declared first rather than compounding."""
        overlapping = [Rescale("d", over=(0, 10), multiply=2), Rescale("d", over=(0, 100), multiply=100)]
        assert apply([5], overlapping) == [10.0]

    def test_a_remap_and_a_rescale_compose_on_one_factor(self):
        """Read the text into numbers, then convert them, in one declaration."""
        assert apply(["N", 90], [Remap("d", {"N": 0}), Rescale("d", multiply=2)]) == [0.0, 180.0]

    def test_text_is_in_no_range(self):
        assert apply(["N"], [Rescale("d", over=(None, None), multiply=2)]) == ["N"]

    def test_a_numeral_is_a_number_for_a_range(self):
        """Metadata through JSON is all text, and a range written over numbers reaches it."""
        assert apply(["500"], [Rescale("d", over=(0, 1000), multiply=2)]) == [1000.0]


@pytest.mark.required
class TestRepairMakesAHeldBackColumnAFactor:
    def test_it_becomes_a_factor(self):
        md = _md([1.0, "N", 2.0, "NE"])
        assert "d" in md.unusable

        md.repair([Remap("d", {None: -1, "N": 0, "NE": 45})])

        assert md.unusable == {}
        assert md.rows_at("sequence")["d"].to_list() == [-1, 0, -1, 45]

    def test_it_stops_being_reported_as_dropped(self):
        md = _md([1.0, "N"]).repair([Remap("d", {None: -1, "N": 0})])
        assert md.dropped_factors == {}

    def test_it_mutates_and_returns_itself(self):
        """A declaration about how the dataset is read, not a question about its rows."""
        md = _md([1.0, "N"])
        assert md.repair([Remap("d", {None: -1})]) is md

    def test_a_repair_that_leaves_it_mixed_leaves_it_unusable(self):
        """Rather than being quietly completed by a rule nobody wrote."""
        md = _md([1.0, "N", "NE"]).repair([Remap("d", {"N": 0})])
        assert md.unusable["d"].reasons == ("mixed_types",)
        assert "d" not in md.factor_names

    def test_the_declaration_is_readable(self):
        md = _md([1.0, "N"]).repair([Remap("d", {None: -1})])
        assert md.repairs == (Remap("d", {None: -1}),)

    def test_declaring_again_replaces_rather_than_accumulates(self):
        """So re-running a cell is safe and the record matches what was asked for."""
        md = _md([1.0, "N"]).repair([Remap("d", {None: -1})])
        md.repair([Remap("d", {None: -9, "N": 0})])
        assert len(md.repairs) == 1
        assert md.rows_at("sequence")["d"].to_list() == [-9, 0]


@pytest.mark.required
class TestRepairAlsoConvertsAFactorThatWasAlreadyReadable:
    """A column in the wrong units is not unusable, and is corrected the same way."""

    @staticmethod
    def _altitudes():
        dataset = _mot_dataset([[1]] * 3, [{"alt": 500.0}, {"alt": 1500.0}, {"alt": 3000.0}])
        return Metadata(dataset)

    @staticmethod
    def _both_units():
        return [Rescale("alt", over=(0, 1000), multiply=0.3048), Rescale("alt", over=(1000, None), multiply=0.001)]

    def test_the_column_is_rewritten(self):
        md = self._altitudes().repair(self._both_units())
        assert md.rows_at("sequence")["alt"].to_list() == [152.4, 1.5, 3.0]

    def test_declaring_it_twice_converts_once(self):
        """Read from the values the dataset wrote, not from the corrected column."""
        md = self._altitudes().repair(self._both_units()).repair(self._both_units())
        assert md.rows_at("sequence")["alt"].to_list() == [152.4, 1.5, 3.0]

    def test_dropping_it_restores_what_was_written(self):
        md = self._altitudes().repair(self._both_units())
        md.unrepair("alt")
        assert md.rows_at("sequence")["alt"].to_list() == [500, 1500, 3000]


@pytest.mark.required
class TestUnrepair:
    def test_dropping_one_factor_leaves_the_others(self):
        dataset = _mot_dataset([[1]] * 2, [{"d": 1.0, "e": 2.0}, {"d": "N", "e": 4.0}])
        md = Metadata(dataset).repair([Remap("d", {None: -1, "N": 0}), Rescale("e", multiply=10)])
        md.unrepair("e")
        assert [c.factor for c in md.repairs] == ["d"]
        assert md.rows_at("sequence")["e"].to_list() == [2.0, 4.0]

    def test_dropping_everything_takes_no_arguments(self):
        md = _md([1.0, "N"]).repair([Remap("d", {None: -1})])
        md.unrepair()
        assert md.repairs == ()
        assert "d" in md.unusable

    def test_a_correction_naming_nothing_is_refused(self):
        with pytest.raises(ValueError, match="Cannot repair"):
            _md([1.0, "N"]).repair([Remap("nope", {"a": 1})])


@pytest.mark.required
class TestARepairIsRecordedProvenance:
    def test_it_survives_the_archive(self, tmp_path):
        dataset = _mot_dataset([[1]] * 3, [{"d": 1.0}, {"d": "N"}, {"d": 2.0}])
        Metadata(dataset).repair([Remap("d", {None: -1, "N": 0})]).save(tmp_path / "m.dem")

        back = Metadata.load(tmp_path / "m.dem", dataset)
        assert back.repairs == (Remap("d", {None: -1, "N": 0}),)
        assert back.rows_at("sequence")["d"].to_list() == [-1, 0, -1]
        assert back.unusable == {}

    def test_dropping_it_after_a_round_trip_restores_the_original(self, tmp_path):
        """Which needs the pre-repair values in the archive, not just the corrected column."""
        dataset = _mot_dataset([[1]] * 2, [{"alt": 500.0}, {"alt": 1500.0}])
        Metadata(dataset).repair([Rescale("alt", multiply=2)]).save(tmp_path / "m.dem")

        back = Metadata.load(tmp_path / "m.dem", dataset)
        back.unrepair()
        assert back.rows_at("sequence")["alt"].to_list() == [500, 1500]

    def test_it_carries_to_the_next_dataset(self):
        """Including values the first dataset never held, which the catch-all answers for."""
        md = _md([1.0, "N", 2.0]).repair([Remap("d", {None: -1, "N": 0})])
        following = md.new(_mot_dataset([[1]] * 2, [{"d": "N"}, {"d": 5.0}]))

        assert following.repairs == md.repairs
        assert following.rows_at("sequence")["d"].to_list() == [0, -1]

    def test_it_is_written_into_the_descriptor(self, tmp_path):
        import json

        md = _md([1.0, "N"]).repair([Remap("d", {None: -1, "N": 0})])
        md.export_encoding(tmp_path / "e.json")

        document = json.loads((tmp_path / "e.json").read_text())
        assert document["version"] == 2
        assert document["corrections"][0]["factor"] == "d"


@pytest.mark.required
class TestASuppliedFactorIsValidatedAtTheCall:
    """The caller is holding the array, so a raise is actionable — unlike a dataset's own
    metadata, which they cannot edit and which is held back for repair instead."""

    def test_add_factors_refuses_a_mixed_array(self):
        md = Metadata(_mot_dataset([[1]] * 3))
        md._structure()
        with pytest.raises(ValueError, match="no single value type"):
            md.add_factors({"d": np.array([1.0, "N", 2.0], dtype=object)}, level="sequence")

    def test_from_factors_refuses_a_mixed_array(self):
        with pytest.raises(ValueError, match="no single value type"):
            Metadata.from_factors(
                {"d": np.array([1.0, "N", 2.0], dtype=object)}, class_labels=np.zeros(3, dtype=np.intp)
            )

    def test_the_message_counts_both_kinds(self):
        with pytest.raises(ValueError, match=r"2 numeric, 1 text"):
            Metadata.from_factors(
                {"d": np.array([1.0, "N", 2.0], dtype=object)}, class_labels=np.zeros(3, dtype=np.intp)
            )

    @pytest.mark.parametrize(
        "values",
        [np.array(["a", "b", "c"], dtype=object), np.array([1.0, 2.0, 3.0]), np.array(["1", "2", "3"], dtype=object)],
    )
    def test_a_column_that_agrees_with_itself_is_accepted(self, values):
        assert Metadata.from_factors({"d": values}, class_labels=np.zeros(3, dtype=np.intp)).factor_names == ["d"]


@pytest.mark.required
class TestParseReadsTextAsAValue:
    """Removing what is not part of the number, so the number underneath can be read."""

    def test_a_thousands_separator_is_dropped(self):
        assert apply(["1,234", "2,000"], [ParseValue("d", drop=[","])]) == ["1234", "2000"]

    def test_a_unit_written_into_the_cell_is_dropped(self):
        assert apply(["12 kg", "7kg"], [ParseValue("d", drop=[" ", "kg"])]) == ["12", "7"]

    def test_the_drops_apply_in_the_order_given(self):
        assert apply(["a-b-c"], [ParseValue("d", drop=["a-", "b-"])]) == ["c"]

    def test_a_decimal_comma_is_read_as_a_point(self):
        assert apply(["3,14"], [ParseValue("d", decimal=",")]) == ["3.14"]

    def test_a_value_that_is_already_a_number_is_untouched(self):
        """Nothing about it was in doubt, so there is no decoration to strip."""
        assert apply([12.0, 7], [ParseValue("d", drop=["2"])]) == [12.0, 7]

    def test_text_the_rule_does_not_rescue_is_left_as_it_was(self):
        """A partial reading is a partial reading."""
        assert apply(["abc"], [ParseValue("d", drop=[","])]) == ["abc"]

    def test_an_absent_row_is_never_touched(self):
        assert apply([None], [ParseValue("d", drop=[","])]) == [None]

    def test_it_makes_a_held_back_column_a_factor(self):
        md = _md([1.0, "2,000", 3.0])
        assert "d" in md.unusable

        md.repair([ParseValue("d", drop=[","])])

        assert md.unusable == {}
        assert md.rows_at("sequence")["d"].to_list() == [1, 2000, 3]


@pytest.mark.required
class TestParseDateTimeReadsTextAsATime:
    """A column of timestamps agrees with itself perfectly; what it lacks is a vocabulary."""

    @pytest.mark.parametrize(
        ("every", "expected"),
        [
            ("year", "2020"),
            ("quarter", "2020-Q3"),
            ("month", "2020-08"),
            ("week", "2020-W35"),
            ("day", "2020-08-27"),
            ("hour", "2020-08-27T12"),
            ("month_of_year", "8"),
            ("day_of_week", "4"),
            ("hour_of_day", "12"),
        ],
    )
    def test_each_period_is_labelled(self, every, expected):
        """27 August 2020 was a Thursday, in ISO week 35."""
        assert apply(["2020-08-27T12:52:58"], [ParseDateTime("d", every=every)]) == [expected]

    def test_without_a_period_the_instant_is_kept(self):
        assert apply(["1970-01-01T00:01:00"], [ParseDateTime("d")]) == [60.0]

    def test_a_naive_timestamp_is_read_as_utc(self):
        """So the same declaration gives the same number wherever it is replayed."""
        naive = apply(["1970-01-01T00:00:00"], [ParseDateTime("d")])
        aware = apply(["1970-01-01T00:00:00+00:00"], [ParseDateTime("d")])
        assert naive == aware == [0.0]

    def test_a_trailing_z_is_read_as_utc(self):
        """Which the 3.10 floor's fromisoformat does not accept unaided."""
        assert apply(["1970-01-01T00:00:00Z"], [ParseDateTime("d")]) == [0.0]

    def test_a_declared_format_reads_a_column_no_standard_describes(self):
        assert apply(["27/08/2020 12:52"], [ParseDateTime("d", format="%d/%m/%Y %H:%M", every="day")]) == ["2020-08-27"]

    def test_a_value_the_format_does_not_read_is_left_as_it_was(self):
        assert apply(["not a time"], [ParseDateTime("d", every="day")]) == ["not a time"]

    def test_a_value_of_no_readable_kind_is_untouched(self):
        """Text, numbers and datetimes are all read; a mapping is none of those."""
        assert apply([{"a": 1}], [ParseDateTime("d", every="day")]) == [{"a": 1}]

    def test_an_absent_row_is_never_touched(self):
        assert apply([None], [ParseDateTime("d", every="day")]) == [None]


def _timestamps():
    """Distinct enough to name their rows: 48 values over 8 hours of one afternoon."""
    return [f"2020-08-27T{hour:02d}:{minute:02d}:00" for hour in range(8, 16) for minute in range(0, 42, 7)]


@pytest.mark.required
class TestRepairingAColumnThatNamesItsRows:
    """A timestamp is dropped for a reason no cleanup fixes, and repaired all the same."""

    def test_it_is_dropped_for_naming_its_rows(self):
        md = _md(_timestamps())
        assert md.dropped_factors == {"d": ["cardinality_over_budget"]}
        assert "d" not in md.factor_names

    def test_it_reports_its_values_and_says_it_is_repairable(self):
        """Nothing about its values disagreed, so they reached the store and are still there
        to be read again -- which is what makes the drop repairable, and what lets a caller
        see the spelling a format has to match."""
        held = _md(_timestamps()).unusable["d"]

        assert held.reasons == ("cardinality_over_budget",)
        assert held.repairable
        assert held.counts == {"text": 48}
        assert held.distinct["text"][0] == "2020-08-27T08:00:00"

    def test_reading_it_as_a_period_makes_it_a_factor(self):
        md = _md(_timestamps()).repair([ParseDateTime("d", every="hour_of_day")])

        assert md.dropped_factors == {}
        assert "d" in md.factor_names
        assert sorted(set(md.rows_at("sequence")["d"].to_list())) == list(range(8, 16))

    def test_a_reading_that_leaves_every_row_unique_leaves_it_dropped(self):
        """Rather than being admitted as a factor that still names its rows. A period too
        fine for the collection groups nothing: one frame a day for 48 days has as many
        days as rows."""
        daily = [f"2020-{month:02d}-{day:02d}T12:00:00" for month in (1, 2) for day in range(1, 25)]
        md = _md(daily).repair([ParseDateTime("d", every="day")])

        assert md.dropped_factors == {"d": ["cardinality_over_budget"]}
        assert "d" not in md.factor_names

    def test_the_instant_is_kept_and_cut_into_bins(self):
        """A numeric column never names its rows however distinct its values: it is cut
        into bins instead, which is the whole reason the identifier rule asks only about
        the columns that cannot be."""
        md = _md(_timestamps()).repair([ParseDateTime("d")])

        assert "d" in md.factor_names
        assert md.factor_info["d"].is_binned
        codes = md.factor_data[:, md.factor_names.index("d")]
        assert 1 < len(set(codes.tolist())) < len(_timestamps())

    def test_dropping_the_repair_restores_the_timestamps(self):
        md = _md(_timestamps()).repair([ParseDateTime("d", every="day")])
        assert "d" in md.factor_names

        md.unrepair()

        assert md.dropped_factors == {"d": ["cardinality_over_budget"]}
        assert "d" not in md.factor_names

    def test_a_correction_naming_nothing_at_all_is_still_refused(self):
        with pytest.raises(ValueError, match="Cannot repair"):
            _md(_timestamps()).repair([ParseDateTime("absent")])


@pytest.mark.required
class TestTheReadingsRefuseWhatNoDatasetIsNeededToReject:
    """Checked at the declaration, where the caller is holding the mistake."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"drop": "kg"}, "bare string is ambiguous"),
            ({"drop": [""]}, "non-empty string"),
            ({"drop": [","], "decimal": ","}, "Drop it or read it, not both"),
            ({"decimal": ",,"}, "single character"),
            ({}, "drops nothing"),
        ],
    )
    def test_a_parse_that_cannot_mean_anything_is_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ParseValue("d", **kwargs)

    def test_a_period_this_does_not_read_is_refused(self):
        """Checked at runtime as well as in the annotation: a period read out of a config
        file has never been through a type checker."""
        with pytest.raises(ValueError, match="not a period this reads"):
            ParseDateTime("d", every="fortnight")  # type: ignore[arg-type]

    @pytest.mark.parametrize("record", [ParseValue, ParseDateTime])
    def test_an_unnamed_factor_is_refused(self, record):
        with pytest.raises(ValueError, match="needs a factor name"):
            record("")

    @pytest.mark.parametrize(
        "record",
        [ParseValue("d", drop=[","]), ParseDateTime("d", every="day"), ParseDateTime("d")],
    )
    def test_a_declaration_can_be_put_in_a_set(self, record):
        """A record whose whole purpose is being stored and compared between runs."""
        assert len({record, record}) == 1


@pytest.mark.required
class TestParseDateTimeReadsEverySpellingOfAMoment:
    """A timestamp reaches a column as text, as a number, or as a datetime, and all three
    mean the same moment. Reading only the text left a declaration against either of the
    others recorded, replayed, and doing nothing to any row."""

    def test_a_datetime_is_already_a_moment(self):
        assert apply([datetime(2020, 8, 27, 12, 52, 58)], [ParseDateTime("d", every="day")]) == ["2020-08-27"]

    def test_a_bare_date_is_read_as_midnight(self):
        assert apply([date(2020, 8, 27)], [ParseDateTime("d", every="hour_of_day")]) == ["0"]

    def test_a_number_is_read_as_seconds_since_the_epoch(self):
        assert apply([1598532778], [ParseDateTime("d", every="day")]) == ["2020-08-27"]

    def test_the_unit_is_declared_rather_than_guessed(self):
        """The same integer is a plausible reading in every unit, so the record says which."""
        assert apply([1598532778000], [ParseDateTime("d", epoch="ms", every="day")]) == ["2020-08-27"]
        assert apply([1598532778000000], [ParseDateTime("d", epoch="us", every="day")]) == ["2020-08-27"]

    @pytest.mark.parametrize("value", [np.int64(1598532778), np.float64(1598532778.0)])
    def test_a_numpy_number_reads_like_a_python_one(self, value):
        assert apply([value], [ParseDateTime("d", every="day")]) == ["2020-08-27"]

    @pytest.mark.parametrize("value", [True, False, np.bool_(True)])
    def test_a_boolean_is_not_a_moment(self, value):
        """``bool`` is a subclass of ``int``, and True is not one second past the epoch."""
        assert apply([value], [ParseDateTime("d", every="day")]) == [value]

    def test_a_count_that_names_no_moment_is_left_as_it_was(self):
        """Milliseconds read as seconds land tens of thousands of years out, which is the
        usual sign the unit was misdeclared. The column stays mixed and says so."""
        assert apply([1598532778000], [ParseDateTime("d", every="day")]) == [1598532778000]

    def test_it_round_trips_through_its_own_output(self):
        """`every=None` emits seconds, which is why seconds is what a number is read as."""
        instant = apply(["2020-08-27T12:52:58"], [ParseDateTime("d")])
        assert apply(instant, [ParseDateTime("d", every="month")]) == ["2020-08"]

    def test_a_value_of_no_readable_kind_is_untouched(self):
        assert apply([{"a": 1}], [ParseDateTime("d", every="day")]) == [{"a": 1}]

    def test_a_unit_this_does_not_count_in_is_refused(self):
        with pytest.raises(ValueError, match="not a unit this counts in"):
            ParseDateTime("d", epoch="fortnights")  # type: ignore[arg-type]

    def test_it_repairs_a_column_of_epoch_numbers(self):
        """Numeric, so never dropped for naming its rows -- and previously a declaration
        against it was recorded, archived and silently inert."""
        md = _md([1598532778 + hour * 3600 for hour in range(6)])
        md.repair([ParseDateTime("d", every="hour_of_day")])
        assert sorted(set(md.rows_at("sequence")["d"].to_list())) == [12, 13, 14, 15, 16, 17]


@pytest.mark.required
class TestUnusableSamplesAColumnThatNamesItsRows:
    """The values of an identifier column are near-unique by definition, so reporting every
    one of them costs a set the size of the column for a report chosen from a handful."""

    def test_it_reports_a_bounded_sample_and_says_so(self):
        held = _md(_timestamps()).unusable["d"]

        assert held.sampled
        assert len(held.distinct["text"]) == 32
        assert held.counts == {"text": 48}, "the counts stay exact, whatever was kept"

    def test_a_column_under_the_cap_is_not_a_sample(self):
        """So the flag means "values are missing", not "this kind of drop". A column is an
        identifier by near-uniqueness, which 23 distinct values over 44 rows satisfies
        without reaching the cap."""
        distinct = [f"2020-08-27T12:{minute:02d}:00" for minute in range(23)]
        held = _md([*distinct, *([distinct[0]] * 21)]).unusable["d"]

        assert held.reasons == ("cardinality_over_budget",)
        assert not held.sampled
        assert len(held.distinct["text"]) == 23

    def test_a_mixed_column_still_reports_every_value(self):
        """There the values are what a repair has to name, and there are as many of them as
        the column has spellings rather than as it has rows."""
        held = _md([1.0, *[f"N{i}" for i in range(40)]]).unusable["d"]

        assert not held.sampled
        assert len(held.distinct["text"]) == 40

    def test_the_sample_keeps_the_values_the_full_report_would_show_first(self):
        """Capped by keeping the smallest, not by keeping whatever arrived first. A sample of
        arrivals is a sample of row order, so a value filling most of a column can sit
        outside it - which is exactly how SeaDrone's empty `date_time` went unreported."""
        crowded = [*[f"2020-08-27T{n // 60:02d}:{n % 60:02d}:00" for n in range(40)], *([""] * 40)]

        sampled, full = _kinds(crowded, limit=8), _kinds(crowded)

        assert sampled[1]["text"] == full[1]["text"][:8], "the sample is a prefix of the full list"
        assert sampled[1]["text"][0] == "", "including the value most of the column holds"
        assert sampled[0] == full[0], "and the counts are exact either way"

    def test_the_sample_flag_is_part_of_the_record(self):
        sampled = Unusable(("cardinality_over_budget",), "unit", True, {}, {}, True)
        assert "sampled=True" in repr(sampled)
        assert sampled != Unusable(("cardinality_over_budget",), "unit", True, {}, {})
        assert len({sampled, sampled}) == 1
