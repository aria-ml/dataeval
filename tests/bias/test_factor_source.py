"""Which representation of a factor gets scored, and what says so in the output.

A cut is a claim about the world, so an evaluator that reads past it answers a question
nobody asked. These pin the rule that decides: a declared, counted or ratified cut keeps
its codes; one nobody chose gives way to the values it was cut from -- but only where
those values carry something the cut threw away.
"""

import warnings

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._helpers import resolve_factor_channel
from dataeval.bias import Balance

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

N = 400


def _pair(seed=1, n=N):
    """Two correlated continuous factors, so the regimes have something to disagree about."""
    rng = np.random.default_rng(seed)
    temp = rng.normal(5.0, 12.0, n)
    return {"temp_c": temp, "humid": 0.7 * temp + rng.normal(0.0, 8.0, n)}


def _md(factors=None, n=N, **kwargs):
    rng = np.random.default_rng(0)
    given = _pair() if factors is None else factors
    return Metadata.from_factors(given, class_labels=rng.integers(0, 3, n), **kwargs)


def _regimes(result) -> set[str]:
    return set(result.factors["scored_as"].to_list())


def _score(result, a="humid", b="temp_c") -> float:
    row = result.factors.filter((pl.col("factor1") == a) & (pl.col("factor2") == b))
    return float(row["mi_value"][0])


@pytest.mark.required
class TestTheDefaultReadsWhatNobodyClaimed:
    def test_an_underived_pair_is_read_as_measured(self):
        """Nobody cut these on purpose, so there is no claim to honour."""
        assert _regimes(Balance().evaluate(_md())) == {"estimator"}

    def test_a_declared_cut_keeps_its_codes(self):
        """The whole point: a cutoff somebody chose is not read past."""
        md = _md(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf], "humid": [-np.inf, 0.0, np.inf]})
        assert _regimes(Balance().evaluate(md)) == {"linfoot"}

    def test_a_declared_count_keeps_its_codes(self):
        """A count is a claim about resolution, which is still a claim."""
        md = _md(continuous_factor_bins={"temp_c": 4, "humid": 4})
        assert _regimes(Balance().evaluate(md)) == {"linfoot"}

    def test_accepting_a_derived_cut_makes_it_a_claim(self):
        """Ratifying is what turns a placement nobody chose into one somebody did."""
        md = _md()
        _ = md.factor_data
        before = _regimes(Balance().evaluate(md))
        md.accept()

        assert before == {"estimator"}
        assert _regimes(Balance().evaluate(md)) == {"linfoot"}

    def test_one_declared_one_not_is_a_mixed_pair(self):
        md = _md(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        assert _regimes(Balance().evaluate(md)) == {"estimator"}

    def test_a_category_is_always_read_as_codes(self):
        """There is no measurement behind a category to prefer."""
        rng = np.random.default_rng(3)
        factors = {
            "weather": np.array(["sun", "rain", "fog"])[rng.integers(0, 3, N)],
            "site": np.array(["north", "south"])[rng.integers(0, 2, N)],
        }
        assert _regimes(Balance().evaluate(_md(factors))) == {"table"}


@pytest.mark.required
class TestAnIntegerColumnIsAlreadyCoded:
    """Reading integral values natively recovers nothing, and costs the cardinality cap."""

    def test_an_identifier_does_not_correlate_with_everything(self):
        """A per-entity id determines every factor measured beside it.

        Reading its values natively tabulates it at full cardinality, where it reports 1.0
        against everything -- arithmetically correct and useless, and exactly what capping
        the level count exists to prevent. ``uid`` stands in for the reserved ``id`` key:
        the same near-unique integer column under a name the layout does not own.
        """
        rng = np.random.default_rng(4)
        factors = {"uid": np.arange(N), "weather": np.array(["sun", "rain", "fog"])[rng.integers(0, 3, N)]}
        result = Balance().evaluate(_md(factors))

        assert _regimes(result) == {"table"}
        assert _score(result, "uid", "weather") < 0.5

    def test_a_count_keeps_the_bins_it_was_given(self):
        """Same argument without the pathology: a count is coded whichever array it is in."""
        rng = np.random.default_rng(5)
        factors = {"detections": rng.integers(0, 400, N), "weather": np.array(["sun", "rain"])[rng.integers(0, 2, N)]}
        assert _regimes(Balance().evaluate(_md(factors))) == {"table"}


@pytest.mark.required
class TestTheSelectorIsExplicit:
    def test_coded_reproduces_the_pre_1_1_read(self):
        assert _regimes(Balance(factor_source="coded").evaluate(_md())) == {"linfoot"}

    def test_values_ignores_a_declared_cut(self):
        """The escape hatch in the other direction, and it says so by ignoring the claim."""
        md = _md(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf], "humid": [-np.inf, 0.0, np.inf]})
        assert _regimes(Balance(factor_source="values").evaluate(md)) == {"estimator"}

    def test_values_is_refused_by_a_codes_only_container(self):
        """Refused rather than quietly answered from the codes, which are other numbers."""

        class Coded:
            factor_names = ["a", "b"]
            factor_data = np.tile(np.arange(4), (2, 100)).T[:400]
            class_labels = np.zeros(400, dtype=np.intp)
            is_binned = [True, True]

        with pytest.raises(ValueError, match="factor_values"):
            Balance(factor_source="values").evaluate(Coded())

    def test_a_coarse_cut_reports_less_than_the_values_it_came_from(self):
        """The cost the selector is choosing between, on one pair of factors.

        Two bins genuinely share less than the values they were cut from, so reporting less
        is right -- and it is why one threshold reads differently across the three regimes.
        """
        md = _md(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf], "humid": [-np.inf, 0.0, np.inf]})
        declared = _score(Balance().evaluate(md))
        measured = _score(Balance(factor_source="values").evaluate(md))

        assert declared < measured

    def test_the_selector_is_recorded_on_the_result(self):
        result = Balance(factor_source="coded").evaluate(_md())
        assert result.meta().state["factor_source"] == "coded"


@pytest.mark.required
class TestAValuesOnlyContainer:
    """The valued protocol is an alternative representation, not an extension."""

    class Measured:
        def __init__(self, values, labels, names):
            self.factor_values = values
            self.class_labels = labels
            self.factor_names = names

    def _measured(self):
        factors = _pair()
        return self.Measured(
            np.column_stack([factors["temp_c"], factors["humid"]]),
            np.random.default_rng(0).integers(0, 3, N).astype(np.intp),
            ["temp_c", "humid"],
        )

    def test_it_is_not_mistaken_for_a_dataset(self):
        """Answering False here sends it to Metadata's constructor as though it were one."""
        assert _regimes(Balance().evaluate(self._measured())) == {"estimator"}

    def test_it_agrees_with_a_metadata_over_the_same_values(self):
        assert _score(Balance().evaluate(self._measured())) == pytest.approx(
            _score(Balance(factor_source="values").evaluate(_md())), rel=1e-9
        )

    def test_conditioning_on_a_factor_says_why_it_cannot(self):
        """A label axis groups rows, which measured values cannot do."""
        with pytest.raises(ValueError, match="groups rows"):
            Balance(label="temp_c").evaluate(self._measured())


@pytest.mark.required
class TestScoredAsNamesTheBranch:
    def test_every_pair_carries_a_regime(self):
        result = Balance().evaluate(_md())
        assert result.factors["scored_as"].null_count() == 0

    def test_the_column_is_symmetric_with_the_pair(self):
        """factors holds both orderings of each pair; a regime is a property of the pair."""
        result = Balance().evaluate(_md(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]}))
        rows = {(a, b): s for a, b, s in result.factors.select(["factor1", "factor2", "scored_as"]).iter_rows()}

        assert all(rows[(a, b)] == rows[(b, a)] for a, b in rows)

    def test_the_three_regimes_are_reachable(self):
        rng = np.random.default_rng(6)
        factors = {
            **_pair(),
            "declared": rng.normal(0.0, 1.0, N),
            "weather": np.array(["sun", "rain", "fog"])[rng.integers(0, 3, N)],
        }
        md = _md(factors, continuous_factor_bins={"declared": 4, "temp_c": 4})
        # temp_c and declared are both cut on purpose -> linfoot; weather has its own
        # alphabet -> table; humid is nobody's cut and is measured -> estimator.
        assert _regimes(Balance().evaluate(md)) == {"linfoot", "table", "estimator"}


@pytest.mark.required
class TestNumNeighborsApplies:
    """It was deprecated on the grounds that nothing reached the estimator. Now something does."""

    def test_it_moves_a_measured_pair(self):
        md = _md()
        assert _score(Balance(num_neighbors=3).evaluate(md)) != _score(Balance(num_neighbors=20).evaluate(md))

    def test_it_does_not_move_a_coded_pair(self):
        md = _md(continuous_factor_bins={"temp_c": 4, "humid": 4})
        assert _score(Balance(num_neighbors=3).evaluate(md)) == _score(Balance(num_neighbors=20).evaluate(md))

    def test_setting_it_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Balance(num_neighbors=3)

        assert [w for w in caught if "deprecated" in str(w.message)] == []


@pytest.mark.required
class TestFactorValues:
    def test_the_two_representations_line_up_column_for_column(self):
        md = _md()
        assert md.factor_values.shape == md.factor_data.shape

    def test_a_numeric_factor_reports_what_it_measured(self):
        factors = _pair()
        md = _md(factors)
        column = list(md.factor_names).index("temp_c")
        np.testing.assert_allclose(md.factor_values[:, column], factors["temp_c"])

    def test_a_category_reports_its_codes(self):
        """There is no native form to report, and a category's codes are its own alphabet."""
        rng = np.random.default_rng(7)
        md = _md({"weather": np.array(["sun", "rain", "fog"])[rng.integers(0, 3, N)]})
        np.testing.assert_array_equal(md.factor_values[:, 0], md.factor_data[:, 0])

    def test_a_declared_cut_does_not_change_what_was_measured(self):
        """`factor_values` reports the values; the cut lives in `factor_data`."""
        plain = _md().factor_values
        cut = _md(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]}).factor_values
        np.testing.assert_allclose(plain, cut)


@pytest.mark.required
class TestANamedChannelTheContainerDoesNotHave:
    """Both directions refuse symmetrically, and ``"auto"`` reads whatever is there."""

    class Continuous:
        factor_names = ["temp_c", "humid"]
        class_labels = np.random.default_rng(0).integers(0, 3, N).astype(np.intp)
        factor_values = np.column_stack([(t := np.random.default_rng(1).normal(5.0, 12.0, N)), 0.7 * t])

    class Counts:
        """Measured, but integral — so "read the codes instead" has nothing to read."""

        factor_names = ["detections", "revisits"]
        class_labels = np.random.default_rng(0).integers(0, 3, N).astype(np.intp)
        factor_values = np.column_stack([
            np.random.default_rng(2).integers(0, 9, N),
            np.random.default_rng(3).integers(0, 5, N),
        ]).astype(float)

    class Coded:
        factor_names = ["a", "b"]
        class_labels = np.zeros(N, dtype=np.intp)
        factor_data = np.tile(np.arange(4), (2, N // 4)).T
        is_binned = [True, True]

    @pytest.mark.parametrize("container", ["Continuous", "Counts"])
    def test_coded_is_refused_by_a_values_only_container(self, container):
        with pytest.raises(ValueError, match="factor_data"):
            Balance(factor_source="coded").evaluate(getattr(self, container)())

    def test_values_is_refused_by_a_codes_only_container(self):
        with pytest.raises(ValueError, match="factor_values"):
            Balance(factor_source="values").evaluate(self.Coded())

    def test_auto_reads_integral_values_when_there_are_no_codes(self):
        """The default path, and the one that crashed.

        Integral values normally resolve to "keep the codes" — reading them natively
        recovers nothing. That answer is only available where codes exist; a values-only
        container of counts has none, so every column resolved to a channel it has not got.
        """
        assert _regimes(Balance().evaluate(self.Counts())) == {"table"}

    def test_auto_reads_measured_values_when_there_are_no_codes(self):
        assert _regimes(Balance().evaluate(self.Continuous())) == {"estimator"}

    def test_auto_reads_codes_when_there_are_no_values(self):
        assert _regimes(Balance().evaluate(self.Coded())) == {"linfoot"}


@pytest.mark.required
class TestCodedAndBinnedAreDifferentQuestions:
    """Two axes, not two words for one.

    ``coded`` asks whether a column holds integers rather than measurements, and is read
    off the array. ``binned`` asks whether those integers came from cutting a range, and
    is read off the record. Binned implies coded and not the reverse, which is why
    ``factor_source`` is named for the first axis and consults the second.
    """

    def test_a_category_is_coded_but_not_binned(self):
        rng = np.random.default_rng(8)
        md = _md({
            "site": np.array(["ridge", "valley", "coast"])[rng.integers(0, 3, N)],
            "weather": np.array(["sun", "rain"])[rng.integers(0, 2, N)],
        })

        assert md.factor_info["site"].is_binned is False
        # ...and `factor_source="coded"` reads it all the same, which is the whole reason
        # the argument is not called "binned".
        assert _regimes(Balance(factor_source="coded").evaluate(md)) == {"table"}

    def test_a_declared_cut_is_both(self):
        md = _md(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf], "humid": [-np.inf, 0.0, np.inf]})

        assert md.factor_info["temp_c"].is_binned is True
        assert _regimes(Balance(factor_source="coded").evaluate(md)) == {"linfoot"}

    def test_reading_values_makes_a_binned_factor_neither(self):
        """The record still says it was binned; this read simply did not use the cut."""
        md = _md(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf], "humid": [-np.inf, 0.0, np.inf]})

        assert md.factor_info["temp_c"].is_binned is True
        assert _regimes(Balance(factor_source="values").evaluate(md)) == {"estimator"}

    def test_the_two_axes_disagree_on_the_pair_that_motivated_them(self):
        """A binned continuous factor: coded, but its alphabet is not its own.

        The combination a single flag could not express, and the reason ``is_binned``
        replaced ``is_discrete``.
        """
        md = _md(continuous_factor_bins={"temp_c": 6, "humid": 6})
        channel = resolve_factor_channel(md, "coded", list(md.factor_names), range(len(md.factor_names)))

        assert channel.coded == [True, True]
        assert channel.own_alphabet == [False, False]


@pytest.mark.required
class TestAMissingMeasurementHasNoNativeReading:
    """A row with no value for a factor has nothing to read natively.

    ``factor_data`` reserves a code for it -- that is what ``BinSpec.missing_code`` is --
    and the neighbor estimator refuses a NaN outright rather than skipping the row. The
    default therefore keeps the codes for such a column, whoever chose the cut.
    """

    def _gapped(self):
        factors = _pair()
        factors["temp_c"] = factors["temp_c"].copy()
        factors["temp_c"][5] = np.nan
        return factors

    def test_the_default_reads_the_codes_rather_than_failing(self):
        """Left to the values this raised `Input X contains NaN` out of sklearn."""
        md = _md(self._gapped())

        # Still `estimator`: the partner carries no gap and is read as measured, which is
        # enough to put the pair on the estimator. What matters is that it returns at all.
        assert _regimes(Balance().evaluate(md)) == {"estimator"}

    def test_the_gap_does_not_move_the_factor_it_shares_a_matrix_with(self):
        """Only the column carrying the gap falls back; the rest are read as before."""
        md = _md(self._gapped())
        channel = resolve_factor_channel(md, "auto", list(md.factor_names), range(len(md.factor_names)))

        assert channel.coded == [False, True]  # humid measured, temp_c coded

    def test_naming_the_channel_says_which_factor_cannot_be_read(self):
        """`values` was asked for outright, so there is no code to fall back to -- say why."""
        md = _md(self._gapped())

        with pytest.raises(ValueError, match=r"\['temp_c'\] hold missing measurements"):
            Balance(factor_source="values").evaluate(md)
