import logging

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.bias._parity import Parity
from tests.conftest import MockICDataset, MockMetadata, to_metadata

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")


@pytest.mark.required
class TestParityUnit:
    """Test the Parity class interface."""

    def test_initialization_defaults(self):
        parity_obj = Parity()
        assert parity_obj.score_threshold == 0.3
        assert parity_obj.p_value_threshold == 0.05

    def test_initialization_custom(self):
        parity_obj = Parity(score_threshold=0.4, p_value_threshold=0.01)
        assert parity_obj.score_threshold == 0.4
        assert parity_obj.p_value_threshold == 0.01

    def test_warns_with_not_enough_frequency(self, caplog):
        labels = [0, 1]
        factors = {"factor1": [10, 20]}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        with caplog.at_level(logging.WARNING):
            parity_obj.evaluate(metadata)
        assert len(caplog.text) > 0

    def test_passes_with_enough_frequency(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)
        assert isinstance(result.factors, pl.DataFrame)

    def test_output_is_dataframe(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)

        # Check that output is a DataFrame
        assert isinstance(result.factors, pl.DataFrame)

        # Check schema
        assert set(result.factors.schema.keys()) == {
            "factor_name",
            "score",
            "p_value",
            "is_significant",
            "has_insufficient_data",
        }
        assert result.factors.schema["factor_name"].base_type() == pl.Categorical
        assert result.factors.schema["score"] == pl.Float64
        assert result.factors.schema["p_value"] == pl.Float64
        assert result.factors.schema["is_significant"] == pl.Boolean
        assert result.factors.schema["has_insufficient_data"] == pl.Boolean

    def test_empty_metadata(self):
        mock_metadata = MockMetadata(
            class_labels=np.array([], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_binned=[],
            index2label={},
        )
        parity_obj = Parity()
        with pytest.raises(ValueError, match="No factors found"):
            parity_obj.evaluate(mock_metadata)

    def test_metadata_stored(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        parity_obj.evaluate(metadata)
        assert parity_obj.metadata is not None
        assert isinstance(parity_obj.metadata, Metadata)
        assert parity_obj.metadata.factor_names == metadata.factor_names

    def test_evaluate_with_raw_dataset(self):
        """Passing a raw AnnotatedDataset wraps it in Metadata (else branch, line 195)."""
        labels = [0] * 5 + [1] * 5
        factors = [{"factor1": "foo"}] * 10
        images = np.zeros((len(labels), 1, 16, 16))
        raw_dataset = MockICDataset(images, labels, metadata=factors)

        parity_obj = Parity()
        result = parity_obj.evaluate(raw_dataset)  # type: ignore

        assert isinstance(result.factors, pl.DataFrame)
        assert isinstance(parity_obj.metadata, Metadata)

    def test_threshold_parameters(self):
        """Test that custom thresholds affect correlation detection."""
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        metadata = to_metadata(factors, labels)

        parity_obj1 = Parity(score_threshold=0.1, p_value_threshold=0.5)
        result1 = parity_obj1.evaluate(metadata)

        parity_obj2 = Parity(score_threshold=0.9, p_value_threshold=0.01)
        result2 = parity_obj2.evaluate(metadata)

        # Lower thresholds should detect more correlated factors (or equal)
        correlated_1 = result1.factors.filter(pl.col("is_significant")).height
        correlated_2 = result2.factors.filter(pl.col("is_significant")).height
        assert correlated_1 >= correlated_2


class TestParityFunctional:
    """Test functional behavior of Parity class."""

    def test_correlated_factors(self):
        """
        In this dataset, class and factor1 are perfectly correlated.
        This tests that the p-value is less than 0.05, which
        corresponds to class and factor1 being highly correlated.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)

        # Checks that factor1 is highly correlated with class
        p_value = result.factors.filter(pl.col("factor_name") == "factor1")["p_value"][0]
        assert p_value < 0.05

    def test_uncorrelated_factors(self):
        """
        This verifies that if the factor is homogeneous for the whole dataset,
        that chi2 and p correspond to factor1 being uncorrelated with class.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)

        # Checks that factor1 is uncorrelated with class
        factor_row = result.factors.filter(pl.col("factor_name") == "factor1")
        score = factor_row["score"][0]
        p_value = factor_row["p_value"][0]

        assert np.isclose(score, 0)
        assert np.isclose(p_value, 1)

    def test_quantized_factors(self):
        """
        This discretizes 'factor1' into having two values.
        This verifies that the '11' and '10' values get grouped together.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 2}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        parity_obj = Parity()
        result1 = parity_obj.evaluate(metadata)

        discrete_dataset = {"factor2": [10] * 5 + [20] * 5}
        metadata = to_metadata(discrete_dataset, labels)
        result2 = parity_obj.evaluate(metadata)

        # Checks that the test on the quantization continuous_dataset is
        # equivalent to the test on the discrete dataset discrete_dataset
        score1 = result1.factors["score"][0]
        p_value1 = result1.factors["p_value"][0]
        score2 = result2.factors["score"][0]
        p_value2 = result2.factors["p_value"][0]

        assert score1 == score2
        assert p_value1 == p_value2

    def test_overquantized_factors(self):
        """
        This quantizes factor1 to have only one value, so that the discretized
        factor1 is the same over the entire dataset.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 1}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)

        # Checks if factor1 and class are perfectly uncorrelated
        score = result.factors["score"][0]
        p_value = result.factors["p_value"][0]

        assert np.isclose(score, 0)
        assert np.isclose(p_value, 1)

    def test_underquantized_has_low_freqs(self, caplog):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        parity_obj = Parity()

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with caplog.at_level(logging.WARNING):
            result = parity_obj.evaluate(metadata)
        assert len(caplog.text) > 0

        # Check that has_insufficient_data flag is set
        has_insuff = result.factors["has_insufficient_data"][0]
        assert has_insuff is True

    def test_underquantized_has_repeated_low_freqs(self, caplog):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        labels = [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10)) + list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        parity_obj = Parity()

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with caplog.at_level(logging.WARNING):
            result = parity_obj.evaluate(metadata)
        assert len(caplog.text) > 0

        # Check that has_insufficient_data flag is set
        has_insuff = result.factors["has_insufficient_data"][0]
        assert has_insuff is True


@pytest.mark.required
class TestInsufficientDataNamesItsLevels:
    """The only output that used to hand a user a bare factor code."""

    def _metadata(self, n=120, **kwargs):
        rng = np.random.default_rng(3)
        return Metadata.from_factors(
            {
                "illum_lux": rng.normal(50, 30, n),
                "weather": np.array(["sun", "rain", "fog"])[rng.integers(0, 3, n)],
            },
            class_labels=rng.integers(0, 3, n),
            **kwargs,
        )

    def test_a_binned_factors_level_is_named_by_its_interval(self):
        """`{"illum_lux": {3: ...}}` said nothing about which lighting to go and collect."""
        result = Parity().evaluate(self._metadata())
        for levels in result.insufficient_data.values():
            for name in levels:
                assert not name.lstrip("-").isdigit(), f"{name!r} is still a bare code"

    def test_a_declared_cutoff_names_the_level_it_declared(self):
        """The level a user asked about, spelled the way they asked for it."""
        result = Parity().evaluate(
            self._metadata(continuous_factor_bins={"illum_lux": [-np.inf, 10.0, np.inf]}),
        )
        for levels in result.insufficient_data.get("illum_lux", {}):
            assert levels in ("< 10", ">= 10")

    def test_a_categorical_factors_level_is_named_by_its_value(self):
        rng = np.random.default_rng(11)
        # One rare category, so it is guaranteed to land in insufficient_data.
        weather = np.array(["sun"] * 100 + ["eclipse"] * 3)
        md = Metadata.from_factors(
            {"weather": weather},
            class_labels=rng.integers(0, 3, len(weather)),
        )
        levels = Parity().evaluate(md).insufficient_data.get("weather", {})
        assert "eclipse" in levels

    def test_a_container_with_no_record_still_reports_something(self):
        """A bare MetadataLike keeps the code, stringified -- what it always got.

        The record lives on Metadata; the protocol stays four members, so a third-party
        container degrades to naming a code after itself rather than failing.
        """
        rng = np.random.default_rng(5)
        n = 60
        codes = np.column_stack([rng.integers(0, 8, n), rng.integers(0, 3, n)])
        bare = MockMetadata(
            class_labels=rng.integers(0, 3, n),
            factor_data=codes,
            factor_names=["a", "b"],
            is_binned=[True, False],
            index2label={0: "x", 1: "y", 2: "z"},
        )
        result = Parity().evaluate(bare)
        for levels in result.insufficient_data.values():
            assert all(isinstance(name, str) for name in levels)


def _detections(images, per_image):
    """An object detection dataset of `images` images, each holding `per_image` detections."""
    from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget

    rng = np.random.default_rng(0)
    boxes = np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (per_image, 1))
    targets = [
        ObjectDetectionTarget(boxes, rng.integers(0, 2, per_image), np.full(per_image, 0.5)) for _ in range(images)
    ]
    return MockDataset(list(range(images)), targets)


@pytest.mark.required
class TestParityAcrossLevels:
    """A factor read below the level it was measured at is not one observation per row."""

    @staticmethod
    def _scored(md):
        factors = Parity(label="weather").evaluate(md).factors
        row = factors.filter(pl.col("factor_name") == "brightness")
        return float(row["score"][0]), float(row["p_value"][0])

    @staticmethod
    def _built(seed, images=60, per_image=40):
        """Two independent per-image factors, read on detection rows."""
        rng = np.random.default_rng(seed)
        md = Metadata(_detections(images, per_image))
        md._structure()
        md.add_factors(
            {
                "brightness": rng.integers(0, 3, images).astype(np.int64),
                "weather": rng.integers(0, 3, images).astype(np.int64),
            },
            level="unit",
        )
        return md

    def test_independence_is_not_rejected_on_replication_alone(self):
        """Sixty images over twenty-four hundred detections, with nothing shared.

        The G-test statistic is linear in the table's total, so the fan-out multiplies the
        evidence for a difference that is not there. Read against the detections instead,
        every one of these seeds rejected independence, the strongest at p=4e-90.
        """
        rejected = [p for _, p in (self._scored(self._built(seed)) for seed in range(8)) if p < 0.05]
        assert len(rejected) <= 1, f"rejected {len(rejected)} of 8 independent pairs"

    def test_both_statistics_are_what_reading_one_row_per_image_gives(self):
        """The propagated view is an account of the per-image one, and now agrees with it.

        Stronger than asserting the p-value rose: it fixes *which* number is right, so a
        correction that overshot fails here just as a missing one does. Cramér's V is
        included because it moves too -- it divides by ``n`` and survives the scaling, but
        its Bergsma correction subtracts a term in ``1/(n-1)`` that an inflated ``n``
        shrinks away.
        """
        for seed in range(6):
            md = self._built(seed)
            assert self._scored(md) == pytest.approx(self._scored(md.at("unit")), abs=1e-9), f"seed {seed}"

    def test_the_insufficient_data_flag_is_not_suppressed_by_replication(self):
        """The flag compares counts against 5, and replication is what hides a thin cell.

        Two images in a category read as eighty observations at a fan-out of forty, so the
        warning this exists to raise is exactly the one that goes quiet. Counts are
        fractional once scaled, because what they count is images rather than detections.
        """
        md = self._built(0)
        flagged = Parity(label="weather").evaluate(md).insufficient_data
        assert flagged == Parity(label="weather").evaluate(md.at("unit")).insufficient_data
        assert any(
            isinstance(count, float)
            for classes in flagged.values()
            for counts in classes.values()
            for count in counts.values()
        )

    def test_a_single_level_dataset_is_untouched(self):
        """Every factor at the level being read is the case that never needed correcting."""
        rng = np.random.default_rng(0)
        factors = {
            "brightness": rng.integers(0, 3, 200).astype(np.int64),
            "weather": rng.integers(0, 3, 200).astype(np.int64),
        }
        md = Metadata.from_factors(factors, class_labels=rng.integers(0, 2, 200))
        assert md.levels == ("unit",)
        # Reaches the statistic with no entity counts at all, so it is scored as it always was.
        assert 0.0 <= self._scored(md)[1] <= 1.0
