"""The encoding as an artifact: inspected, written out, and applied to the next dataset.

Recording how values became codes is only worth doing if the record can leave the object
and come back. These pin the loop: export what one dataset derived, read it in a pull
request, hand it to the next dataset, and get the same codes for the same values.
"""

import json
import re
import warnings
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._helpers import _code_names, _edge_format, factor_code_names
from dataeval._metadata._encoding import (
    DESCRIPTOR_VERSION,
    corrections_from_json,
    corrections_from_list,
    encoding_from_json,
    encoding_to_json,
)
from dataeval._metadata._metadata import _reconcile_encoding
from dataeval.types import BinSpec, LevelSpec, ParseDateTime, ParseValue, Remap, Rescale

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _md(factors, n=200, **kwargs):
    rng = np.random.default_rng(0)
    return Metadata.from_factors(factors, class_labels=rng.integers(0, 3, n), **kwargs)


def _winter(n=200):
    rng = np.random.default_rng(1)
    return {
        "temp_c": rng.normal(5.0, 12.0, n),
        "weather": np.array(["sun", "rain", "fog"])[rng.integers(0, 3, n)],
    }


@pytest.mark.required
class TestEncodingAccessor:
    def test_one_factor_and_every_factor(self):
        md = _md(_winter())
        every = md.encoding()

        assert set(every) == set(md.factor_names)
        assert md.encoding("temp_c") is every["temp_c"]

    def test_an_unknown_factor_is_an_error(self):
        with pytest.raises(KeyError, match="not among this metadata's factors"):
            _md(_winter()).encoding("nonexistent")

    def test_a_declared_cut_reports_where_and_who(self):
        """The question that had no answer before: where did you cut, and who chose it."""
        md = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        spec = md.encoding("temp_c")

        assert isinstance(spec, BinSpec)
        assert spec.edges == (-np.inf, 0.0, np.inf)
        assert spec.provenance == "edges"


@pytest.mark.required
class TestCodeNames:
    """Asking the record what a code means, without going through an evaluator.

    The lookup existed and had no public door: :attr:`ParityOutput.insufficient_data` and
    the ``label=`` axis groups are named through it, so anything rendering the same factor
    itself either got the same strings or approximated them.
    """

    def test_one_factor_and_every_factor(self):
        md = _md(_winter())
        every = md.code_names()

        assert set(every) == set(md.factor_names)
        assert md.code_names("weather") == every["weather"]

    def test_an_unknown_factor_is_an_error(self):
        with pytest.raises(KeyError, match="not among this metadata's factors"):
            _md(_winter()).code_names("nonexistent")

    def test_a_vocabulary_names_its_codes_by_value(self):
        md = _md(_winter())
        assert set(md.code_names("weather").values()) == {"sun", "rain", "fog"}

    def test_a_declared_cut_survives_into_its_own_labels(self):
        """Naming a bin after its contents hid the cutoff that gave it meaning."""
        md = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        assert md.code_names("temp_c") == {1: "< 0", 2: ">= 0"}

    def test_a_bin_nothing_reached_is_still_named(self):
        """An empty bin is a finding, and reporting one means naming it."""
        rng = np.random.default_rng(2)
        md = _md(
            {"temp_c": rng.normal(20.0, 3.0, 200)},
            continuous_factor_bins={"temp_c": [-np.inf, 0.0, 10.0, np.inf]},
        )
        names = md.code_names("temp_c")

        assert set(names) == {1, 2, 3}
        assert names[1] == "< 0"  # named although no row is below freezing
        assert names[2] == "[0, 10)"

    def test_names_agree_with_what_the_evaluators_report(self):
        """The whole reason this is public: one answer, not two that nearly match."""
        md = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        names = list(md.factor_names)
        through_evaluators = factor_code_names(md, md.factor_data, names)

        for position, name in enumerate(names):
            for code, label in through_evaluators[position].items():
                assert md.code_names(name)[code] == label

    def test_large_magnitudes_stay_distinguishable(self):
        """Six significant figures collapses epoch milliseconds onto one label."""
        rng = np.random.default_rng(3)
        base = 1787011240000000
        md = _md(
            {"capture_ms": (base + rng.integers(0, 1_200_000, 200)).astype(float)},
            continuous_factor_bins={"capture_ms": 6},
        )
        names = md.code_names("capture_ms")

        assert len(set(names.values())) == len(names)

    def test_a_factor_that_reached_neither_path_names_nothing(self):
        md = _md(_winter())
        md._factor_cache["temp_c"] = replace(
            md.factor_info["temp_c"], is_binned=False, is_digitized=False, encoding=None
        )
        assert md.code_names("temp_c") == {}


@pytest.mark.required
class TestDescriptorFormat:
    def test_the_descriptor_round_trips(self):
        """What ``encoding()`` emits has to be accepted back, or the loop cannot close."""
        md = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        assert encoding_from_json(encoding_to_json(md.encoding())) == md.encoding()

    def test_the_same_encoding_produces_the_same_bytes(self):
        """A descriptor is diffed in a pull request, so it has to be stable."""
        md = _md(_winter())
        assert encoding_to_json(md.encoding()) == encoding_to_json(md.encoding())

    def test_infinite_edges_survive_as_words(self):
        """JSON has no literal for an infinity, and ``Infinity`` is not portable JSON."""
        text = encoding_to_json({"f": BinSpec(edges=(-np.inf, 0.0, np.inf), provenance="edges")})
        document = json.loads(text)  # strict: would raise on a bare Infinity

        assert document["factors"]["f"]["edges"] == ["-inf", 0.0, "inf"]
        restored = encoding_from_json(text)["f"]
        assert isinstance(restored, BinSpec)
        assert restored.edges == (-np.inf, 0.0, np.inf)

    def test_a_descriptor_written_by_a_newer_dataeval_is_refused(self):
        text = json.dumps({"version": DESCRIPTOR_VERSION + 1, "factors": {}})
        with pytest.raises(ValueError, match="version"):
            encoding_from_json(text)

    def test_an_unknown_kind_names_the_factor(self):
        text = json.dumps({"version": DESCRIPTOR_VERSION, "factors": {"f": {"kind": "wat"}}})
        with pytest.raises(ValueError, match="'f'"):
            encoding_from_json(text)

    def test_export_writes_a_file_the_constructor_accepts(self, tmp_path):
        md = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        path = tmp_path / "bins.json"
        md.export_encoding(path)

        assert _md(_winter(), encoding=path).encoding("temp_c").edges == (-np.inf, 0.0, np.inf)


@pytest.mark.required
class TestApplyingARecord:
    """Stage 7: new data encoded against a locked descriptor rather than its own draw."""

    def test_two_metadata_sharing_a_record_agree_on_codes(self):
        """The property that makes train-vs-test factor comparison mean anything."""
        winter = _winter()
        first = _md(winter, continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        second = _md(winter, encoding=first.encoding())

        np.testing.assert_array_equal(first.factor_data, second.factor_data)

    def test_the_cut_is_reapplied_not_refitted(self):
        """A warmer second collection keeps the freezing boundary rather than moving it."""
        rng = np.random.default_rng(7)
        first = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        spring = {"temp_c": rng.normal(18.0, 6.0, 200), "weather": np.array(["sun"] * 200)}
        second = _md(spring, encoding=first.encoding())

        assert second.encoding("temp_c").edges == (-np.inf, 0.0, np.inf)

    def test_a_value_outside_the_range_lands_in_an_end_bin(self):
        """Infinite outer edges are what make the binned case stable: no new code appears."""
        first = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        extreme = {"temp_c": np.array([-500.0, 500.0] * 100), "weather": np.array(["sun"] * 200)}
        second = _md(extreme, encoding=first.encoding())

        codes = set(second.factor_data[:, list(second.factor_names).index("temp_c")].tolist())
        assert codes <= set(range(len(first.encoding("temp_c").edges) + 2))

    def test_a_new_category_appends_and_leaves_codes_untouched(self):
        """The core stability change, executing for the first time.

        ``np.unique`` would sort ``hail`` in ahead of ``rain`` and ``sun``, renumbering
        both. Appending is what keeps a code meaning what it meant.
        """
        first = _md(_winter())
        assert first.encoding("weather").levels == ("fog", "rain", "sun")

        rng = np.random.default_rng(3)
        later = {
            "temp_c": rng.normal(5.0, 12.0, 200),
            "weather": np.array(["sun", "rain", "fog", "hail"])[rng.integers(0, 4, 200)],
        }
        second = _md(later, encoding=first.encoding())
        grown = second.encoding("weather")

        assert isinstance(grown, LevelSpec)
        assert grown.levels == ("fog", "rain", "sun", "hail")
        # Out of sort order, deliberately: sorting is what renumbers.
        assert grown.levels[:3] == first.encoding("weather").levels

    def test_a_partial_descriptor_encodes_the_rest_normally(self):
        """What stage 4 produces: cutoffs declared for the factors that matter, and no more."""
        md = _md(_winter(), encoding={"temp_c": BinSpec(edges=(-np.inf, 0.0, np.inf), provenance="edges")})

        assert md.encoding("temp_c").provenance == "edges"
        assert md.encoding("weather").provenance == "derived"

    def test_encoding_and_continuous_factor_bins_together_is_an_error(self):
        """Two sources disagreeing about one factor has no good resolution."""
        with pytest.raises(ValueError, match="pass one"):
            _ = _md(
                _winter(),
                continuous_factor_bins={"temp_c": 5},
                encoding={"temp_c": BinSpec(edges=(-np.inf, 0.0, np.inf), provenance="edges")},
            )

    def test_a_record_that_is_not_a_spec_is_refused(self):
        with pytest.raises(TypeError, match="BinSpec or a LevelSpec"):
            _md(_winter(), encoding={"temp_c": [0.0, 1.0]})

    def test_a_derived_view_keeps_the_record(self):
        """`at()` and `where()` configure the next instance identically, record included."""
        md = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        assert md.at("unit").encoding("temp_c").edges == (-np.inf, 0.0, np.inf)


@pytest.mark.required
class TestAcceptingADerivedEncoding:
    """Stage 4: a person reads what was derived and ratifies it, changing no edges."""

    def test_accept_marks_the_record_without_moving_codes(self):
        """The distinction the policy argument turns on: nobody looked vs someone approved."""
        md = _md(_winter())
        before = md.factor_data.copy()
        assert md.encoding("temp_c").provenance == "derived"

        md.accept()

        assert md.encoding("temp_c").provenance == "accepted"
        np.testing.assert_array_equal(md.factor_data, before)

    def test_accept_covers_derived_vocabularies_too(self):
        """A vocabulary read off whatever values turned up is equally nobody's decision."""
        md = _md(_winter())
        md.accept()
        assert md.encoding("weather").provenance == "accepted"

    def test_accepting_one_factor_leaves_the_others_derived(self):
        md = _md(_winter())
        md.accept("temp_c")

        assert md.encoding("temp_c").provenance == "accepted"
        assert md.encoding("weather").provenance == "derived"

    def test_a_declared_cut_is_not_relabelled(self):
        """Accepting ratifies a derived placement; a declared one was never in question."""
        md = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        md.accept()
        assert md.encoding("temp_c").provenance == "edges"

    def test_an_unknown_factor_is_an_error(self):
        with pytest.raises(KeyError, match="not among this metadata's factors"):
            _md(_winter()).accept("nonexistent")

    def test_an_accepted_placement_stops_being_re_derived(self):
        """Accepting fixes the cut: a differently shaped sample gets the ratified edges."""
        first = _md(_winter())
        first.accept()
        rng = np.random.default_rng(9)
        other = {"temp_c": rng.normal(40.0, 2.0, 200), "weather": np.array(["sun"] * 200)}

        assert _md(other, encoding=first.encoding()).encoding("temp_c").edges == first.encoding("temp_c").edges


@pytest.mark.required
class TestDeclaredVocabularies:
    """The categorical counterpart to ``continuous_factor_bins``."""

    def test_a_declared_vocabulary_fixes_the_codes(self):
        """Codes are settled before any data is seen, which is what lets two datasets agree."""
        md = _md(_winter(), factor_levels={"weather": ["sun", "rain", "fog", "snow"]})
        spec = md.encoding("weather")

        assert isinstance(spec, LevelSpec)
        assert spec.provenance == "declared"
        assert spec.levels == ("sun", "rain", "fog", "snow")

    def test_an_unseen_value_appends_by_default(self):
        """Extension wants append; a category the declaration missed is not an error."""
        rng = np.random.default_rng(5)
        factors = {"temp_c": rng.normal(5.0, 12.0, 200), "weather": np.array(["sun", "hail"] * 100)}
        md = _md(factors, factor_levels={"weather": ["sun", "rain"]})

        assert md.encoding("weather").levels == ("sun", "rain", "hail")

    def test_strict_refuses_a_value_the_declaration_does_not_hold(self):
        """A closed taxonomy wants to hear that the data left it, not to be widened."""
        rng = np.random.default_rng(5)
        factors = {"temp_c": rng.normal(5.0, 12.0, 200), "weather": np.array(["sun", "hail"] * 100)}
        with pytest.raises(ValueError, match="declared vocabulary"):
            _ = _md(factors, factor_levels={"weather": ["sun", "rain"]}, strict=True).factor_data

    def test_declaring_a_factor_twice_is_an_error(self):
        with pytest.raises(ValueError, match="declare each factor once"):
            _ = _md(
                _winter(),
                encoding={"weather": LevelSpec(levels=("sun",), provenance="declared")},
                factor_levels={"weather": ["sun", "rain"]},
            )


@pytest.mark.required
class TestTheRecordSurvivesTheArchive:
    def test_accept_survives_a_save_and_load(self, tmp_path):
        """Without this the archive destroys the review the whole lifecycle produces."""
        md = _md(_winter())
        md.accept()
        path = tmp_path / "md.dem"
        md.save(path)

        back = Metadata.load(path)
        assert back.encoding("temp_c").provenance == "accepted"
        assert back.encoding("temp_c").edges == md.encoding("temp_c").edges

    def test_a_restored_instance_reproduces_its_codes(self, tmp_path):
        md = _md(_winter())
        path = tmp_path / "md.dem"
        md.save(path)

        np.testing.assert_array_equal(Metadata.load(path).factor_data, md.factor_data)


@pytest.mark.required
class TestFitnessIsReported:
    """R4: applying a locked descriptor reports its own fitness rather than re-fitting."""

    def test_a_stale_descriptor_names_the_bins_nothing_reached(self):
        rng = np.random.default_rng(4)
        warm = {"t": rng.normal(30.0, 3.0, 200)}
        with pytest.warns(UserWarning, match="left bins unused"):
            _ = _md(warm, continuous_factor_bins={"t": [-np.inf, -20.0, -10.0, 0.0, np.inf]}).factor_data

    def test_a_cut_finer_than_the_sample_supports_is_named(self):
        """M4: nothing checked a declared count against the sample before."""
        rng = np.random.default_rng(4)
        with pytest.warns(UserWarning, match="finer than the sample supports"):
            _ = _md({"t": rng.normal(0.0, 1.0, 200)}, continuous_factor_bins={"t": 500}).factor_data

    def test_a_declared_vocabulary_finer_than_the_sample_is_named_too(self):
        """M4 on the other declaration channel: a vocabulary fills a table the same way a cut does."""
        rng = np.random.default_rng(4)
        values = rng.integers(0, 500, 200)
        with pytest.warns(UserWarning, match="finer than the sample supports"):
            _ = _md({"t": values}, factor_levels={"t": list(range(500))}).factor_data

    def test_a_declared_vocabulary_is_not_asked_the_emptiness_question(self):
        """A closed taxonomy applied to a subset of itself is the intended use, not a misfit."""
        rng = np.random.default_rng(4)
        values = np.array(["sun", "rain"])[rng.integers(0, 2, 200)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = _md({"w": values}, factor_levels={"w": ["sun", "rain", "fog", "snow", "hail"]}).factor_data

        assert not [w for w in caught if "unused" in str(w.message) or "finer than" in str(w.message)]

    def test_the_fineness_report_offers_no_replacement_count(self):
        """M4a: the budget it fires against sits well past the resolution that reads best."""
        rng = np.random.default_rng(4)
        md = _md({"t": rng.normal(0.0, 1.0, 200)}, continuous_factor_bins={"t": 500})
        with pytest.warns(UserWarning, match="finer than the sample supports") as caught:
            _ = md.factor_data
        levels = len(np.unique(md.factor_data[:, 0]))

        message = str(next(w.message for w in caught if "finer than" in str(w.message)))
        assert "no count to recommend" in message
        # Every number in the message is a measurement of this factor -- the codes it landed
        # and the rows it landed them over. A third one would be read as a target, and the
        # obvious candidate is the budget this fires against: at 200 rows that is 20, while
        # a pair of true dependence 0.810 reports 0.507 cut at 20 against 0.729 cut at 8.
        assert set(re.findall(r"\d+", message)) == {str(levels), "200"}

    def test_a_cut_that_fits_is_silent(self):
        rng = np.random.default_rng(4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = _md({"t": rng.normal(0.0, 10.0, 200)}, continuous_factor_bins={"t": [-np.inf, 0.0, np.inf]}).factor_data

        assert not [w for w in caught if "bins unused" in str(w.message) or "finer than" in str(w.message)]

    def test_the_report_does_not_refit(self):
        """Notice that it no longer fits, not a new fit -- reencode() is the caller's call."""
        rng = np.random.default_rng(4)
        edges = [-np.inf, -20.0, -10.0, 0.0, np.inf]
        md = _md({"t": rng.normal(30.0, 3.0, 200)}, continuous_factor_bins={"t": edges})
        with pytest.warns(UserWarning, match="left bins unused"):
            _ = md.factor_data

        assert md.encoding("t").edges == tuple(edges)


@pytest.mark.required
class TestTheDigestAttributesAResult:
    """R5: a result should name the encoding it was computed under."""

    def test_the_digest_moves_when_the_encoding_does(self):
        derived = _md(_winter())
        declared = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        assert derived.encoding_digest != declared.encoding_digest

    def test_the_digest_holds_still_when_the_encoding_does(self):
        assert _md(_winter()).encoding_digest == _md(_winter()).encoding_digest

    def test_accepting_changes_the_digest(self):
        """Ratifying is a change to the policy, so a result computed after it is attributable."""
        md = _md(_winter())
        before = md.encoding_digest
        md.accept()
        assert md.encoding_digest != before

    def test_a_result_carries_the_digest_it_was_computed_under(self):
        from dataeval.bias import Balance

        md = _md(_winter())
        result = Balance().evaluate(md)
        assert result.meta().state["encoding_digest"] == md.encoding_digest


@pytest.mark.required
class TestReencode:
    def test_reencode_keeps_a_declared_cut_by_default(self):
        """Re-deriving over a declaration discards the semantic work the declaration is."""
        md = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        assert md.reencode().encoding("temp_c").edges == (-np.inf, 0.0, np.inf)

    def test_reencode_can_be_told_to_start_from_the_data(self):
        md = _md(_winter())
        md.accept()
        assert md.reencode(keep_declared=False).encoding("temp_c").provenance == "derived"

    def test_reencode_leaves_the_original_alone(self):
        """A result already computed under the old codes stays attributable to them."""
        md = _md(_winter())
        md.accept()
        _ = md.reencode(keep_declared=False).factor_data
        assert md.encoding("temp_c").provenance == "accepted"


@pytest.mark.required
class TestValuesThatDoNotBehaveLikeKeys:
    """Encoding reads values into a lookup, and not every value looks itself up."""

    def test_a_recorded_vocabulary_holding_a_missing_value_can_be_reapplied(self):
        """``NaN != NaN``, so a NaN cannot find its own slot in a dict keyed by value.

        The lookup finds the slot by hash and then rejects it on equality unless it is the
        very same object -- and ``tolist()`` builds a fresh float every time it is called,
        so it never is. Every recorded-encoding reapply over a digitized factor holding a
        missing value went down with a bare ``KeyError: nan``.
        """
        grade = np.array([0.0, 1.0, 2.0, np.nan] * 50)
        md = _md({"grade": grade}, encoding={"grade": LevelSpec(levels=(0.0, 1.0, 2.0, np.nan), provenance="declared")})
        codes = md.factor_data[:, 0]

        # The declared vocabulary is used as given: four levels, no fifth appended for a
        # NaN that failed to recognise itself, and every missing row on the one code.
        assert len(md.encoding("grade").levels) == 4
        assert list(np.unique(codes)) == [0, 1, 2, 3]
        assert codes[3] == 3
        assert int((codes == 3).sum()) == 50

    def test_a_missing_value_survives_the_archive(self, tmp_path):
        """The reviewer's route in: save, load, and the restored record is reapplied."""
        md = _md({"grade": np.array([0.0, 1.0, 2.0, np.nan] * 50)})
        _ = md.factor_data
        path = tmp_path / "md.dem"
        md.save(path)

        np.testing.assert_array_equal(Metadata.load(path).factor_data, md.factor_data)

    def test_a_vocabulary_declared_as_an_array_is_writable(self):
        """``factor_levels={"grade": np.arange(4)}`` is the natural spelling next to
        every other array-shaped argument, and a ``np.int64`` is not JSON-serializable.

        The digest reads through the same renderer, so an unwrapped level took every bias
        evaluation down with it, not merely ``export_encoding``.
        """
        md = _md({"grade": np.arange(4).repeat(50)}, factor_levels={"grade": np.arange(4)})

        assert md.encoding_digest
        assert json.loads(encoding_to_json(md.encoding()))["factors"]["grade"]["levels"] == [0, 1, 2, 3]


@pytest.mark.required
class TestDeclaringAFactorTwice:
    """One factor described twice has no good resolution; two arguments over disjoint
    factors are only a longhand for one record."""

    def test_one_factor_cut_two_ways_is_an_error(self):
        with pytest.raises(ValueError, match="pass one"):
            _md(
                _winter(),
                continuous_factor_bins={"temp_c": 4},
                encoding={"temp_c": BinSpec(edges=(-np.inf, 0.0, np.inf), provenance="edges")},
            )

    def test_a_vocabulary_and_a_cut_on_one_factor_is_an_error(self):
        """Left unchecked the vocabulary silently won: ``_encoding`` is consulted first,
        so a declared cutoff was discarded and a continuous factor level-encoded into one
        level per distinct value -- exactly the silent override the check exists to stop.
        """
        with pytest.raises(ValueError, match="declare each factor once"):
            _md(
                _winter(),
                continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]},
                factor_levels={"temp_c": [0.0, 1.0]},
            )

    def test_two_arguments_over_different_factors_are_fine(self):
        """Per factor, not per argument -- and this is a state the library itself produces."""
        md = _md(
            _winter(),
            continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]},
            factor_levels={"weather": ["sun", "rain", "fog"]},
        )

        assert md.encoding("temp_c").edges == (-np.inf, 0.0, np.inf)
        assert md.encoding("weather").provenance == "declared"

    def test_a_loaded_instance_can_still_be_reconfigured(self, tmp_path):
        """``load(path, continuous_factor_bins=...)`` leaves both populated, since the
        archive fills in the factors the reader said nothing about. A per-argument check
        then refused the instance the library had just handed back."""
        path = tmp_path / "md.dem"
        md = _md(_winter())
        _ = md.factor_data
        md.save(path)

        back = Metadata.load(path, continuous_factor_bins={"temp_c": 4})
        # Each factor still travels by exactly one route; the pair is the library's doing.
        assert back._continuous_factor_bins == {"temp_c": 4}
        assert set(back._encoding) == {"weather"}

        # Accepting it is the assertion. This is the call `new()` makes to configure the
        # next dataset the same way, and a per-argument check raised on a combination
        # nobody passed -- leaving a loaded instance unable to be reconfigured at all.
        assert _reconcile_encoding(back._continuous_factor_bins, back._encoding, None)


@pytest.mark.required
class TestARecordThatDoesNotFitItsFactor:
    def test_a_cut_recorded_for_a_non_numeric_factor_names_the_factor(self):
        """Left to the cast this surfaced as ``could not convert string to float: 'fog'``
        from inside NumPy, which names a value and neither the factor nor the record that
        sent it there."""
        with pytest.raises(TypeError, match="weather.*BinSpec"):
            _ = _md(
                _winter(),
                encoding={"weather": BinSpec(edges=(-np.inf, 0.0, np.inf), provenance="edges")},
            ).factor_data


@pytest.mark.required
class TestAGrownVocabularyIsWrittenBack:
    def test_growth_reaches_the_record_not_only_the_cache(self):
        """``new()`` hands the next dataset ``_encoding``, so a vocabulary grown here but
        cached there sends the next dataset an alphabet this one has already outgrown."""
        md = _md(_winter(), encoding={"weather": LevelSpec(levels=("sun", "rain"), provenance="declared")})
        _ = md.factor_data

        assert md.encoding("weather").levels == ("sun", "rain", "fog")
        # The durable half, which is what `new()` hands on -- not only the cache.
        assert md._encoding["weather"] == md.encoding("weather")

    def test_an_unseen_value_can_be_refused_instead(self):
        with pytest.raises(ValueError, match="declared vocabulary"):
            _ = _md(
                _winter(),
                encoding={"weather": LevelSpec(levels=("sun", "rain"), provenance="declared")},
                strict=True,
            ).factor_data


@pytest.mark.required
class TestDerivedInstancesOwnTheirRecords:
    """Every path that copies an instance copies what it can rewrite, or the copy is a view."""

    @pytest.mark.parametrize(
        "derive",
        [
            lambda md: md.at("unit"),
            lambda md: md.where(pl.col("class_label") >= 0),
            lambda md: md.reencode(),
        ],
        ids=["at", "where", "reencode"],
    )
    def test_accepting_on_a_derived_view_leaves_the_source_alone(self, derive):
        md = _md(_winter())
        _ = md.factor_data
        before = md.encoding("temp_c").provenance

        derive(md).accept()
        assert md.encoding("temp_c").provenance == before == "derived"

    def test_reencode_does_not_share_the_containers_it_promises_not_to_touch(self):
        """``reencode`` used a bare ``copy.copy``, so everything but the two containers it
        happened to rebuild was still the original's."""
        md = _md(_winter())
        _ = md.factor_data
        fresh = md.reencode()

        for name in ("_factors", "_factor_cache", "_factors_by_level", "_dropped_factors", "_aggregated_from"):
            assert getattr(fresh, name) is not getattr(md, name), name

    def test_reencode_from_the_data_drops_every_spelling_of_a_declaration(self):
        """``continuous_factor_bins`` is as much a cut somebody chose as a ``BinSpec`` is,
        and it is consulted on the re-derived pass."""
        md = _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        fresh = md.reencode(keep_declared=False)

        assert fresh.encoding("temp_c").provenance == "derived"
        assert fresh.encoding("temp_c").edges != (-np.inf, 0.0, np.inf)


@pytest.mark.required
class TestFitnessCountsTheBinsTheCutReaches:
    def test_missing_rows_are_not_an_occupied_bin(self):
        """The reserved missing code is not a bin the cut placed. Counting it inflated
        every tally by one, and where exactly one bin was empty it cancelled the shortfall
        out and said nothing at all."""
        rng = np.random.default_rng(4)
        values = rng.normal(30.0, 3.0, 200)
        values[:20] = np.nan
        with pytest.warns(UserWarning, match="left bins unused"):
            _ = _md({"t": values}, continuous_factor_bins={"t": [-np.inf, 0.0, np.inf]}).factor_data

    def test_an_empty_out_of_range_catchall_is_not_a_bin_that_went_unused(self):
        """``[0, 10, 20]`` declares two intervals, not four bins.

        ``np.digitize`` has to put an out-of-range value somewhere, so a finitely bounded
        list also yields a below-first and an above-last code. Nobody declared those, and
        their being *empty* is the cut working — every value fell inside the declared
        range. Counting them made a perfectly fitting cut warn on every read.
        """
        rng = np.random.default_rng(4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = _md({"t": rng.uniform(0.5, 19.5, 200)}, continuous_factor_bins={"t": [0.0, 10.0, 20.0]}).factor_data

        assert [w for w in caught if "bins unused" in str(w.message)] == []

    def test_a_finite_cut_still_reports_the_declared_intervals_it_missed(self):
        """The check survives the fix: it is the declared intervals that are counted."""
        rng = np.random.default_rng(4)
        with pytest.warns(UserWarning, match=r"1 of 3 bins hold rows"):
            _ = _md(
                {"t": rng.uniform(0.5, 9.5, 200)},
                continuous_factor_bins={"t": [0.0, 10.0, 20.0, 30.0]},
            ).factor_data


@pytest.mark.required
class TestASaveBeforeTheFirstReadKeepsTheDeclaration:
    """The manifest records what binning applied, and an unread instance has applied
    nothing -- so a declaration reached the archive only if something had read from the
    object first. The same object wrote two different files."""

    @pytest.mark.parametrize(
        ("kwargs", "check"),
        [
            (
                {"continuous_factor_bins": {"temp_c": [-np.inf, 0.0, np.inf]}},
                lambda s: s.edges == (-np.inf, 0.0, np.inf),
            ),
            (
                {"encoding": {"temp_c": BinSpec(edges=(-np.inf, 4.0, np.inf), provenance="edges")}},
                lambda s: s.edges == (-np.inf, 4.0, np.inf),
            ),
            ({"factor_levels": {"weather": ["sun", "rain", "fog"]}}, lambda s: s.provenance == "declared"),
        ],
        ids=["bins", "encoding", "levels"],
    )
    def test_a_declaration_survives_a_save_nothing_has_read_from(self, tmp_path, kwargs, check):
        name = "weather" if "factor_levels" in kwargs else "temp_c"
        path = tmp_path / "md.dem"
        _md(_winter(), **kwargs).save(path)

        assert check(Metadata.load(path).encoding(name))

    def test_a_declared_bin_count_restores_the_same_edges(self, tmp_path):
        """A count is not a cut -- it says how finely to divide, and where the edges land is
        still read off the values. Restoring the count re-derives the same edges, because
        the archive holds the same values."""
        path = tmp_path / "md.dem"
        _md(_winter(), continuous_factor_bins={"temp_c": 4}).save(path)

        assert (
            Metadata.load(path).encoding("temp_c").edges
            == _md(_winter(), continuous_factor_bins={"temp_c": 4}).encoding("temp_c").edges
        )

    def test_the_reader_still_outranks_the_archive(self, tmp_path):
        path = tmp_path / "md.dem"
        _md(_winter(), continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]}).save(path)

        back = Metadata.load(path, continuous_factor_bins={"temp_c": [-np.inf, 10.0, np.inf]})
        assert back.encoding("temp_c").edges == (-np.inf, 10.0, np.inf)


@pytest.mark.required
class TestTheRecordSurvivesEveryValueAColumnCanHold:
    """The descriptor is only reviewable if every level a factor can carry fits in JSON.

    The one value that never has to fit is a missing one: it is not a level on any path,
    so it never reaches the vocabulary to be written as the bare ``NaN`` token no other
    reader accepts. A binary column does reach it, and ``json`` refuses one outright --
    from ``encoding_digest``, which every bias evaluator reads *after* computing its result.
    """

    def _gapped(self, n=60):
        rng = np.random.default_rng(2)
        grade = rng.integers(0, 4, n).astype(float)
        grade[3] = np.nan
        return {"grade": grade}

    def test_a_missing_value_is_not_written_as_a_level(self, tmp_path):
        """It is not one of the values the factor takes, so the vocabulary does not name it.

        Recorded as a level it was a bare ``NaN`` token in the JSON, a level called "nan"
        beside real ones in every report, and observed occupancy to anything measuring how
        well a cut fits -- on the one encoding path of three that never reserved a code.
        """
        path = tmp_path / "encoding.json"
        _md(self._gapped(), n=60).export_encoding(path)

        def refuse(token):
            raise AssertionError(f"{token} is not JSON")

        document = json.loads(path.read_text(), parse_constant=refuse)
        assert document["factors"]["grade"]["levels"] == [0.0, 1.0, 2.0, 3.0]

    def test_taking_it_out_of_the_vocabulary_renumbers_nothing(self):
        """`np.unique` collapses NaN to one entry and sorts it last, so the code a missing
        row already held is exactly the one `missing_code` names."""
        md = _md(self._gapped(), n=60)
        codes = md.factor_data[:, list(md.factor_names).index("grade")]
        assert codes[3] == md.encoding("grade").missing_code
        assert md.factor_info["grade"].missing == 1

    def test_the_missing_value_comes_back_as_a_missing_value(self, tmp_path):
        """It has to read back as a gap, or reapplying the record renumbers the factor."""
        path = tmp_path / "encoding.json"
        md = _md(self._gapped(), n=60)
        md.export_encoding(path)

        again = _md(self._gapped(), n=60, encoding=path)
        assert np.array_equal(md.factor_data, again.factor_data)

    def test_a_binary_column_does_not_take_the_digest_down(self):
        """`bytes` is not JSON-serializable, and the digest is read on every bias result."""
        rng = np.random.default_rng(3)
        md = _md({"blob": np.array([b"a", b"b", b"c"] * 20), "num": rng.normal(0, 1, 60)}, n=60)

        assert len(md.encoding_digest) == 16

    def test_a_binary_column_keeps_its_codes_across_an_archive(self, tmp_path):
        """Rendered as text, and read back as text -- so the codes still line up."""
        rng = np.random.default_rng(3)
        path = tmp_path / "md.dem"
        md = _md({"blob": np.array([b"a", b"b", b"c"] * 20), "num": rng.normal(0, 1, 60)}, n=60)
        md.save(path)

        assert np.array_equal(Metadata.load(path).factor_data, md.factor_data)


@pytest.mark.required
class TestARestoredInstanceIsStillConfigurable:
    """An archive carries a declared cut twice -- as the count and as the edges it resolved
    to -- and both come back. Naming one factor in both is the pair the constructor refuses,
    so the instance a caller has just loaded has to stay usable anyway."""

    def _saved(self, tmp_path):
        path = tmp_path / "md.dem"
        md = _md(_winter(), continuous_factor_bins={"temp_c": 5})
        _ = md.factor_data
        md.save(path)
        return md, Metadata.load(path)

    def test_new_can_reconfigure_the_instance_load_just_built(self, tmp_path):
        """This raised "cut by both `continuous_factor_bins` and `encoding`" on every
        declared cut that had been through a save."""
        _, back = self._saved(tmp_path)

        # Each factor travels by exactly one route, so the pair the constructor refuses
        # cannot be built. Asserted through `_reconcile_encoding` — the call `new()` makes,
        # and the one that raised — so it does not need a second dataset to hand.
        assert set(back._continuous_factor_bins) & set(back._encoding) == set()
        assert _reconcile_encoding(back._continuous_factor_bins, back._encoding, None)

    def test_the_archive_records_each_factor_in_exactly_one_member(self, tmp_path):
        """The root cause. A read resolves the count into a `BinSpec`, and writing both the
        count and the spec put one factor in two members that are restored independently.
        """
        md, back = self._saved(tmp_path)

        # The resolved record wins, and it remembers a count was what was asked for.
        assert back.encoding("temp_c").provenance == "count"
        assert back.encoding("temp_c").edges == md.encoding("temp_c").edges

    def test_an_unread_instance_still_writes_its_declared_count(self, tmp_path):
        """The member exists for this case, so the deduplication must not swallow it."""
        path = tmp_path / "unread.dem"
        _md(_winter(), continuous_factor_bins={"temp_c": 5}).save(path)
        back = Metadata.load(path)

        assert back._continuous_factor_bins == {"temp_c": 5}
        reference = _md(_winter(), continuous_factor_bins={"temp_c": 5})
        assert back.encoding("temp_c").edges == reference.encoding("temp_c").edges

    def test_setting_the_bins_re_cuts_the_factor(self, tmp_path):
        """The record is consulted before this mapping, so leaving it in place made the
        assignment silently do nothing on exactly the instances most likely to be re-cut."""
        _, back = self._saved(tmp_path)
        back.continuous_factor_bins = {"temp_c": 2}

        assert len(back.encoding("temp_c").edges) == 3


@pytest.mark.required
class TestALargeMagnitudeCutIsStillReadable:
    """The labels key ``insufficient_data`` and name a ``label=`` axis's groups.

    Six significant figures is right for the factors most people have, and at 1.8e12 it
    rendered every bound of every bin identically — so the output that exists to say which
    subset to collect more of said nothing.
    """

    @staticmethod
    def _names(md):
        return factor_code_names(md, md.factor_data, list(md.factor_names))[0]

    def test_a_timestamp_cut_names_its_bins_distinctly(self):
        stamps = (np.datetime64("2026-08-19", "ms") + np.arange(200).astype("timedelta64[ms]") * 600_000).astype(
            "datetime64[ms]",
        )
        md = Metadata.from_factors({"captured_at": stamps}, class_labels=np.zeros(200, dtype=int))
        names = self._names(md)

        assert len(set(names.values())) == len(names)
        # Written out rather than exponentiated: the digits an exponent hides are the ones
        # that tell two capture times apart.
        assert not any("e+" in name for name in names.values())

    def test_an_ordinary_float_cut_is_unchanged(self):
        """The common case keeps the short labels it had."""
        rng = np.random.default_rng(1)
        md = Metadata.from_factors({"t": rng.normal(5.0, 12.0, 200)}, class_labels=np.zeros(200, dtype=int))

        assert set(self._names(md).values()) == {"< -5.20401", "[-5.20401, 17.1274)", ">= 17.1274"}

    def test_precision_rises_only_as_far_as_distinctness_needs(self):
        """Raising it unconditionally is the opposite failure: ``0.1`` as ``0.10000000000000001``."""
        assert _edge_format([-np.inf, 0.1, 0.2, 0.3, np.inf]) == ".6g"
        assert _edge_format([-np.inf, 1000000.4, 1000000.6, np.inf]) == ".0f"
        # Beyond float64's integral range the value really is scientific.
        assert _edge_format([-np.inf, 1.5e20, 2.5e20, np.inf]) == ".6g"


@pytest.mark.required
class TestADeclarationThatNamesNothingSaysSo:
    """A descriptor naming no factor is not a cheap no-op.

    Every factor it was meant to pin falls back to a cut derived from this draw — the drift
    it exists to prevent — and the failure looks identical to never having passed one.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"continuous_factor_bins": {"citty": 5}},
            {"encoding": {"citty": BinSpec(edges=(-np.inf, 0.0, np.inf), provenance="edges")}},
            {"factor_levels": {"citty": ["a", "b"]}},
        ],
        ids=["bins", "encoding", "levels"],
    )
    def test_every_declaration_channel_names_the_key_it_could_not_place(self, kwargs):
        with pytest.warns(UserWarning, match="citty"):
            _ = _md(_winter(), **kwargs).factor_data

    def test_a_declaration_that_lands_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = _md(_winter(), continuous_factor_bins={"temp_c": 5}).factor_data

        assert [w for w in caught if "given an encoding" in str(w.message)] == []


@pytest.mark.required
class TestStrictSurvivesTheArchive:
    """A closed vocabulary that silently reopens is the one failure ``strict`` prevents.

    And it fails *open*: the next dataset's unseen value is appended rather than refused,
    and nothing says the taxonomy widened.
    """

    CLOSED = ["clear", "cloudy", "rainy"]

    def _closed(self, n=120):
        return _md(
            {"weather": np.array(self.CLOSED)[np.arange(n) % 3]},
            n=n,
            factor_levels={"weather": self.CLOSED},
            strict=True,
        )

    def test_strict_is_restored(self, tmp_path):
        path = tmp_path / "md.dem"
        md = self._closed()
        _ = md.factor_data
        md.save(path)

        assert Metadata.load(path)._strict is True

    def test_the_restored_taxonomy_still_refuses_an_unseen_value(self, tmp_path):
        path = tmp_path / "md.dem"
        md = self._closed()
        _ = md.factor_data
        md.save(path)
        back = Metadata.load(path)

        widened = np.array([*self.CLOSED, "fog"])[np.arange(120) % 4]
        with pytest.raises(ValueError, match="fog"):
            _ = _md({"weather": widened}, n=120, encoding=back.encoding(), strict=back._strict).factor_data

    def test_the_reader_can_close_a_vocabulary_the_archive_left_open(self, tmp_path):
        path = tmp_path / "md.dem"
        _md({"weather": np.array(self.CLOSED)[np.arange(120) % 3]}, n=120).save(path)

        assert Metadata.load(path, strict=True)._strict is True

    def test_a_file_from_before_this_restores_permissive(self, tmp_path):
        """The member is optional, so an older archive is not read as closed."""
        path = tmp_path / "md.dem"
        _md(_winter()).save(path)

        assert Metadata.load(path)._strict is False


@pytest.mark.required
class TestARecordedVocabularyAcceptsAPartlyDeclaredFactor:
    """The derived path splits missing values out; the replay path has to do the same."""

    @staticmethod
    def _partly_declared():
        from tests.metadata.test_structurers import _mot_dataset

        return _mot_dataset([[1], [1], [1]], [{"w": "sun"}, {"w": "rain"}, {}])

    def test_applying_a_declared_vocabulary_does_not_raise(self):
        """`np.unique` cannot sort None against a string — it raised rather than answering,
        so a partly recorded factor had no way through the recorded-encoding path at all."""
        metadata = Metadata(self._partly_declared(), partial_factors=True, factor_levels={"w": ["sun", "rain"]})
        assert metadata.rows_at("sequence")["w"].to_list() == ["sun", "rain", None]

    def test_the_unrecorded_row_takes_the_reserved_code_rather_than_a_level(self):
        """A value nobody recorded is not one of the values the factor takes."""
        metadata = Metadata(self._partly_declared(), partial_factors=True, factor_levels={"w": ["sun", "rain"]})
        spec = metadata.encoding("w")
        assert spec.levels == ("sun", "rain")
        codes = metadata.factor_data[:, list(metadata.factor_names).index("w")]
        assert codes.tolist()[2] == spec.missing_code

    def test_the_reserved_code_is_named_rather_than_numbered(self):
        """It rendered as a bare '2' beside real category names, reading as a category."""
        spec = LevelSpec(levels=("fog", "rain"), provenance="derived")
        assert _code_names(np.array([0, 1, 2]), spec)[2] == "missing"


@pytest.mark.required
class TestCorrectionsAreDeclarationsAndRecordsAtOnce:
    """What is declared and what is written are one object, so a repair can be reviewed in
    a diff and reapplied to the next dataset without being re-decided."""

    @staticmethod
    def _corrections():
        return [
            Remap("direction", {None: -1, "N": 0, "NE": 45, (-1000, 0): -99}),
            Rescale("altitude", over=(0, 1000), multiply=0.3048),
            Rescale("altitude", over=(1000, None), multiply=0.001),
        ]

    def _round_trip(self, corrections):
        return corrections_from_json(encoding_to_json({}, corrections))

    def test_they_round_trip_exactly(self):
        assert self._round_trip(self._corrections()) == self._corrections()

    def test_the_order_they_apply_in_is_kept(self):
        """An array, not an object: one factor may take several, and order decides."""
        back = self._round_trip(self._corrections())
        assert [c.factor for c in back] == ["direction", "altitude", "altitude"]
        assert [c.over for c in back if isinstance(c, Rescale)] == [(0, 1000), (1000, None)]

    def test_a_numeric_key_does_not_become_text(self):
        """JSON object keys are strings, so a mapping keyed on 1 written as an object would
        come back keyed on '1' — the exact confusion corrections exist to resolve."""
        mapping = [Remap("grade", {1: "low", 2.5: "mid", None: "other"})]
        (back,) = self._round_trip(mapping)
        assert isinstance(back, Remap)
        assert dict(back.mapping) == {1: "low", 2.5: "mid", None: "other"}
        assert [type(key).__name__ for key in back.mapping] == ["int", "float", "NoneType"]

    def test_the_catch_all_comes_back_as_the_catch_all(self):
        """``null`` reads as a missing value inside a vocabulary; here it is the key that
        matches everything unnamed, and has to stay None."""
        (back,) = self._round_trip([Remap("d", {None: -1})])
        assert isinstance(back, Remap)
        assert None in back.mapping

    def test_a_range_key_comes_back_as_a_range(self):
        (back,) = self._round_trip([Remap("depth", {(-1000, 0): -1})])
        assert isinstance(back, Remap)
        assert dict(back.mapping) == {(-1000, 0): -1}

    def test_the_readings_round_trip_exactly(self):
        """A reading is config: it is written into a descriptor, committed, and replayed."""
        readings = [
            ParseValue("count", drop=[",", " kg"], decimal="."),
            ParseValue("span", decimal=","),
            ParseDateTime("date_time", every="hour_of_day"),
            ParseDateTime("logged", format="%d/%m/%Y %H:%M", every="day"),
            ParseDateTime("stamp"),
        ]
        assert self._round_trip(readings) == readings

    def test_a_drop_keeps_its_spelling(self):
        """An array of substrings, so a multi-character drop does not become characters."""
        (back,) = self._round_trip([ParseValue("d", drop=["kg"])])
        assert isinstance(back, ParseValue)
        assert back.drop == ("kg",)

    def test_the_epoch_unit_round_trips(self):
        """Which unit a number counts in is a declaration, so it has to survive the file."""
        (back,) = self._round_trip([ParseDateTime("t", epoch="ms", every="day")])
        assert isinstance(back, ParseDateTime)
        assert back.epoch == "ms"

    def test_a_descriptor_written_before_numeric_timestamps_reads_as_seconds(self):
        """Which is what a reading that only ever touched text would have emitted."""
        written = [{"kind": "parse_datetime", "factor": "t", "format": None, "every": "day"}]
        (back,) = corrections_from_list(written)
        assert isinstance(back, ParseDateTime)
        assert back.epoch == "s"

    def test_a_kind_this_reader_does_not_have_is_refused_by_name(self):
        """Which is what a descriptor written by a newer DataEval says to an older one."""
        document = json.dumps({"version": 2, "factors": {}, "corrections": [{"kind": "parse_ipv6", "factor": "d"}]})
        with pytest.raises(ValueError, match="kind 'parse_ipv6'"):
            corrections_from_json(document)

    def test_the_document_carries_both_sections(self):
        document = json.loads(encoding_to_json({"w": LevelSpec(levels=("rain",), provenance="declared")}, []))
        assert document["version"] == DESCRIPTOR_VERSION
        assert set(document) == {"version", "corrections", "factors"}


@pytest.mark.required
class TestADescriptorWrittenBeforeCorrectionsExisted:
    def test_it_loads_with_no_corrections(self):
        """Which is what it meant."""
        v1 = json.dumps({"version": 1, "factors": {"w": {"kind": "levels", "levels": ["rain", "sun"]}}})
        assert corrections_from_json(v1) == []
        recorded = encoding_from_json(v1)["w"]
        assert isinstance(recorded, LevelSpec)
        assert recorded.levels == ("rain", "sun")

    def test_a_version_from_the_future_is_refused_by_number(self):
        with pytest.raises(ValueError, match="version 99"):
            encoding_from_json(json.dumps({"version": 99, "factors": {}}))


@pytest.mark.required
class TestACorrectionRefusesWhatNoDatasetIsNeededToReject:
    """The shapes that are wrong on their face, caught at construction rather than at use."""

    def test_a_remap_that_names_nothing(self):
        with pytest.raises(ValueError, match="names nothing to replace"):
            Remap("d", {})

    def test_a_range_that_runs_backwards(self):
        with pytest.raises(ValueError, match="runs backwards"):
            Rescale("d", over=(10, 1))

    def test_a_range_that_is_not_a_pair(self):
        with pytest.raises(ValueError, match=r"must be a \(low, high\) pair"):
            Remap("d", {(1, 2, 3): 0})

    def test_a_rescale_that_discards_the_readings(self):
        """``multiply=0`` gives every value in range the same answer, which is a Remap."""
        with pytest.raises(ValueError, match="same answer"):
            Rescale("d", multiply=0)

    @pytest.mark.parametrize("build", [lambda: Remap("", {"a": 1}), lambda: Rescale("")])
    def test_a_correction_with_no_factor(self, build):
        with pytest.raises(ValueError, match="needs a factor name"):
            build()

    def test_a_correction_can_be_put_in_a_set(self):
        """A frozen dataclass wrapping a mapping is not hashable, and says so at the call."""
        assert len({Remap("d", {"a": 1}), Remap("d", {"a": 1})}) == 1


@pytest.mark.required
class TestTheDescriptorVersionTracksItsVocabulary:
    def test_the_stamp_moved_when_the_correction_kinds_grew(self):
        """Version 2 was set when a correction was ``remap | rescale``. The vocabulary then
        grew to four kinds, so a v2 reader rejects a document a v2 writer produces — which
        is the exact confusion the field exists to prevent."""
        assert DESCRIPTOR_VERSION == 3

    def test_every_readable_version_is_named_in_the_refusal(self):
        text = json.dumps({"version": 99, "factors": {}})
        with pytest.raises(ValueError, match="reads 1, 2 and 3"):
            encoding_from_json(text)

    def test_a_version_two_document_still_reads(self):
        """Written on this branch before the stamp moved, and readable: this reader has
        every kind either version can name."""
        text = json.dumps({
            "version": 2,
            "factors": {},
            "corrections": [
                {
                    "kind": "rescale",
                    "factor": "d",
                    "over": [None, None],
                    "multiply": 2.0,
                    "add": 0.0,
                    "provenance": "declared",
                }
            ],
        })
        assert corrections_from_json(text) == [Rescale("d", multiply=2.0)]
