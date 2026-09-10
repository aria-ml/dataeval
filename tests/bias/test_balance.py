import copy

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.bias._balance import Balance, BalanceOutput
from tests.conftest import MockMetadata, to_metadata


@pytest.fixture(scope="module")
def metadata_results():
    str_vals = ["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"]
    cnt_vals = [
        -0.54425898,
        -0.31630016,
        0.41163054,
        1.04251337,
        -0.12853466,
        1.36646347,
        -0.66519467,
        0.35151007,
        0.90347018,
        0.0940123,
        -0.74349925,
        -0.92172538,
    ]
    cat_vals = [1.1, 1.1, 0.1, 0.1, 1.1, 0.1, 1.1, 0.1, 0.1, 1.1, 1.1, 0.1]
    class_labels = ["dog", "dog", "dog", "cat", "dog", "cat", "dog", "dog", "dog", "cat", "cat", "cat"]
    md = {"var_cat": str_vals, "var_cnt": cnt_vals, "var_float_cat": cat_vals}
    return to_metadata(md, class_labels, {"var_cnt": 3, "var_float_cat": 2})


@pytest.fixture(scope="module")
def mismatch_metadata():
    raw_metadata = {"factor1": list(range(10)), "factor2": list(range(10)), "factor3": list(range(10))}
    class_labels = [1] * 10
    continuous_bins = {"factor1": 5, "factor2": 5, "factor3": 5}
    return to_metadata(raw_metadata, class_labels, continuous_bins)


@pytest.fixture(scope="module")
def simple_metadata():
    raw_metadata = {"factor1": [1] * 100 + [2] * 100, "factor2": [1] * 100 + [2] * 100}
    class_labels = [1] * 100 + [2] * 100
    return to_metadata(raw_metadata, class_labels)


@pytest.mark.required
class TestBalanceUnit:
    """Test the Balance class interface."""

    def test_initialization_defaults(self):
        balance_obj = Balance()
        assert balance_obj.num_neighbors == 5
        assert balance_obj.class_imbalance_threshold == 0.3
        assert balance_obj.factor_correlation_threshold == 0.5

    def test_initialization_custom(self):
        balance_obj = Balance(num_neighbors=10, class_imbalance_threshold=0.4, factor_correlation_threshold=0.6)
        assert balance_obj.num_neighbors == 10
        assert balance_obj.class_imbalance_threshold == 0.4
        assert balance_obj.factor_correlation_threshold == 0.6

    def test_empty_metadata(self):
        mock_metadata = MockMetadata(
            class_labels=np.array([], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_binned=[],
            index2label={},
        )
        balance_obj = Balance()
        with pytest.raises(ValueError, match="No factors found"):
            balance_obj.evaluate(mock_metadata)

    def test_metadata_stored(self, metadata_results):
        balance_obj = Balance()
        balance_obj.evaluate(metadata_results)
        assert balance_obj.metadata is not None
        assert isinstance(balance_obj.metadata, Metadata)
        assert balance_obj.metadata.factor_names == metadata_results.factor_names

    def test_threshold_parameters(self, simple_metadata):
        """Test that custom thresholds affect the output."""
        balance_obj1 = Balance(class_imbalance_threshold=0.1, factor_correlation_threshold=0.1)
        result1 = balance_obj1.evaluate(simple_metadata)

        balance_obj2 = Balance(class_imbalance_threshold=0.9, factor_correlation_threshold=0.9)
        result2 = balance_obj2.evaluate(simple_metadata)

        # Lower thresholds should detect more issues (or equal)
        imbalanced_1 = result1.classwise.filter(pl.col("is_imbalanced")).height
        imbalanced_2 = result2.classwise.filter(pl.col("is_imbalanced")).height
        assert imbalanced_1 >= imbalanced_2

        correlated_1 = result1.factors.filter(pl.col("is_correlated")).height
        correlated_2 = result2.factors.filter(pl.col("is_correlated")).height
        assert correlated_1 >= correlated_2

    def test_classwise_labels_with_missing_classes(self):
        """Regression: classwise labels should match actual class values, not positional indices."""
        # Classes 2 and 3 are present, but 0 and 1 are not
        # index2label maps: {0: 'cat', 1: 'dog', 2: 'bird', 3: 'fish'}
        mock_metadata = MockMetadata(
            class_labels=np.array([2, 2, 2, 3, 3, 3] * 10, dtype=np.intp),
            factor_data=np.array([[0, 0, 0, 1, 1, 1] * 10], dtype=np.int64).T,
            factor_names=["factor1"],
            is_binned=[False],
            index2label={0: "cat", 1: "dog", 2: "bird", 3: "fish"},
        )
        result = Balance().evaluate(mock_metadata)
        class_names = set(result.classwise["class_name"].to_list())
        assert "bird" in class_names, f"Expected 'bird' in classwise labels, got {class_names}"
        assert "fish" in class_names, f"Expected 'fish' in classwise labels, got {class_names}"
        assert "cat" not in class_names, f"'cat' should not appear (no data), got {class_names}"
        assert "dog" not in class_names, f"'dog' should not appear (no data), got {class_names}"

    def test_correct_dataframe_shapes(self, metadata_results):
        metadata = copy.deepcopy(metadata_results)
        metadata.exclude = []
        num_factors = len(metadata.factor_names)
        num_classes = len(np.unique(metadata.class_labels))

        balance_obj = Balance()
        result = balance_obj.evaluate(metadata)

        # Check balance DataFrame
        assert isinstance(result.balance, pl.DataFrame)
        # balance includes class_label + metadata factors
        assert result.balance.height == num_factors + 1

        # Check balance DataFrame schema
        assert set(result.balance.schema.keys()) == {
            "factor_name",
            "mi_value",
        }
        assert result.balance.schema["factor_name"].base_type() == pl.Categorical
        assert result.balance.schema["mi_value"] == pl.Float64

        # First entry should be class_label
        assert result.balance["factor_name"][0] == "class_label"

        # Check classwise DataFrame
        assert isinstance(result.classwise, pl.DataFrame)
        # classwise covers metadata factors only - class_label is 1.0 by construction
        assert result.classwise.height == num_classes * num_factors
        assert "class_label" not in result.classwise["factor_name"].to_list()

        # Check classwise DataFrame schema
        assert set(result.classwise.schema.keys()) == {
            "class_name",
            "factor_name",
            "mi_value",
            "is_imbalanced",
        }
        assert result.classwise.schema["class_name"].base_type() == pl.Categorical
        assert result.classwise.schema["factor_name"].base_type() == pl.Categorical
        assert result.classwise.schema["mi_value"] == pl.Float64
        assert result.classwise.schema["is_imbalanced"] == pl.Boolean

        # Check factors DataFrame
        assert isinstance(result.factors, pl.DataFrame)
        # Number of ordered pairs = n*(n-1) (includes both A->B and B->A)
        expected_pairs = num_factors * (num_factors - 1)
        assert result.factors.height == expected_pairs

        # Check factors DataFrame schema
        assert set(result.factors.schema.keys()) == {
            "factor1",
            "factor2",
            "mi_value",
            "is_correlated",
            "scored_as",
        }
        assert result.factors.schema["factor1"].base_type() == pl.Categorical
        assert result.factors.schema["factor2"].base_type() == pl.Categorical
        assert result.factors.schema["mi_value"] == pl.Float64
        assert result.factors.schema["is_correlated"] == pl.Boolean


@pytest.mark.required
def test_plot_type_identifies_the_output_for_plotting():
    """`plot_type` is the discriminator dataeval-plots dispatches on."""
    empty = pl.DataFrame()
    output = BalanceOutput(balance=empty, factors=empty, classwise=empty)
    assert output.plot_type == "balance"


def _sequences(count, per_sequence):
    """A tracking dataset of `count` videos, each `per_sequence` frames of one detection."""
    from tests.metadata.test_structurers import _mot_dataset

    return _mot_dataset([[[i] for i in range(per_sequence)] for _ in range(count)])


@pytest.mark.required
class TestBalanceAcrossLevels:
    """A factor read below the level it was measured at is not one observation per row."""

    @staticmethod
    def _pair(md):
        factors = Balance().evaluate(md).factors
        row = factors.filter((pl.col("factor1") == "weather") & (pl.col("factor2") == "terrain"))
        return float(row["mi_value"][0]), bool(row["is_correlated"][0])

    @staticmethod
    def _built(seed, sequences=40, per_sequence=25):
        """A tracking metadata whose only factors are two independent per-sequence ones."""
        rng = np.random.default_rng(seed)
        values = {
            # Integral, so both are read as categories rather than binned: the pair is
            # scored off its contingency table either way, but the entropy branch is the
            # one whose ceiling does not move with where a cut happened to fall.
            "weather": rng.integers(0, 3, sequences).astype(np.int64),
            "terrain": rng.integers(0, 4, sequences).astype(np.int64),
        }
        md = Metadata(_sequences(sequences, per_sequence))
        md._structure()
        md.add_factors(values, level="sequence")
        return md, values

    def test_independent_per_sequence_factors_are_not_reported_as_correlated(self):
        """Forty sequences over a thousand detections, with nothing shared between the factors.

        The values repeat twenty-five times each and never vary within a sequence, so there
        are forty observations of this pair however many rows carry them. Read against the
        rows instead, this pair averaged 0.06 and reached 0.15 -- and at ten sequences over
        the same thousand rows it crossed the 0.5 threshold one run in three.
        """
        scored = [self._pair(self._built(seed)[0]) for seed in range(12)]
        assert not any(flagged for _, flagged in scored)
        assert max(score for score, _ in scored) < 0.1

    def test_the_score_is_what_reading_one_row_per_sequence_gives(self):
        """The propagated view is an account of the sequence-level one, and now agrees with it.

        Stronger than asserting the number fell: it fixes *which* number is right, so a
        correction that overshot fails here just as a missing one does. The comparison is
        built rather than taken from `at("sequence")`, because the class labels this
        evaluator conditions on live at `instance` and do not roll up.
        """
        for seed in range(6):
            md, values = self._built(seed)
            propagated, _ = self._pair(md)

            rng = np.random.default_rng(seed)
            isolated_md = Metadata.from_factors(values, class_labels=rng.integers(0, 2, len(values["weather"])))
            isolated, _ = self._pair(isolated_md)
            assert propagated == pytest.approx(isolated, abs=1e-9), f"seed {seed}"

    def test_a_single_level_dataset_is_untouched(self):
        """Every factor at the level being read is the case that never needed correcting."""
        rng = np.random.default_rng(0)
        factors = {
            "weather": rng.integers(0, 3, 200).astype(np.int64),
            "terrain": rng.integers(0, 4, 200).astype(np.int64),
        }
        md = Metadata.from_factors(factors, class_labels=rng.integers(0, 2, 200))
        assert md.levels == ("unit",)
        # Reaches the statistic with no entity counts at all, so it is scored as it always was.
        assert self._pair(md)[0] >= 0.0
