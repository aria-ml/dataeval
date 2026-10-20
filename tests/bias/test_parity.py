import logging

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.bias._parity import Parity
from tests.conftest import MockICDataset, MockMetadata, to_metadata
from tests.core.test_rao_scott import seq_gap

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

_FACTOR_STATUS = ["associated", "weak", "inconclusive", "not_detected", "untestable"]
_CELL_STATUS = ["underrepresented", "overrepresented", "too_thin", "no_finding", "untestable"]


def bound(replicates, alpha=0.05):
    """The largest rejection rate a calibrated test shows over `replicates`, within three standard errors."""
    return alpha + 3 * np.sqrt(alpha * (1 - alpha) / replicates)


def _expand(table):
    """Rows (level, class) reproducing a contingency table."""
    table = np.asarray(table)
    levels, classes = np.indices(table.shape)
    counts = table.ravel()
    return np.repeat(levels.ravel(), counts), np.repeat(classes.ravel(), counts)


def table_metadata(table):
    """One factor, `site`, whose rows are the levels "rare" then "common", against classes "A" and "B"."""
    levels, classes = _expand(table)
    return Metadata.from_factors(
        {"site": np.array(["rare", "common"])[levels]},
        class_labels=classes,
        index2label={0: "A", 1: "B"},
    )


def rare_cells(out):
    return out.cells.filter(pl.col("level") == "rare")


def _bare(factor_data, class_labels, item_indices=None, names=None):
    factor_data = np.asarray(factor_data).reshape(len(class_labels), -1)
    return MockMetadata(
        class_labels=np.asarray(class_labels, dtype=np.intp),
        factor_data=factor_data,
        factor_names=names or [f"f{i}" for i in range(factor_data.shape[1])],
        is_binned=[False] * factor_data.shape[1],
        index2label={int(c): str(c) for c in np.unique(class_labels)},
        item_indices=None if item_indices is None else np.asarray(item_indices),
    )


@pytest.mark.required
class TestParityUnit:
    """Test the Parity class interface."""

    def test_initialization_defaults(self):
        parity_obj = Parity()
        assert parity_obj.score_threshold == 0.3
        assert parity_obj.p_value_threshold == 0.05
        assert parity_obj.lift_threshold == 1.25
        assert parity_obj.assume_independent is False

    def test_initialization_custom(self):
        parity_obj = Parity(score_threshold=0.4, p_value_threshold=0.01, lift_threshold=1.5, assume_independent=True)
        assert parity_obj.score_threshold == 0.4
        assert parity_obj.p_value_threshold == 0.01
        assert parity_obj.lift_threshold == 1.5
        assert parity_obj.assume_independent is True

    def test_config_declaration_is_not_overridden_by_the_default(self):
        assert Parity(config=Parity.Config(assume_independent=True)).assume_independent is True

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
        factors = {"factor1": ["a", "b"] * 5}
        metadata = to_metadata(factors, labels)
        result = Parity().evaluate(metadata)

        assert isinstance(result.factors, pl.DataFrame)
        assert result.factors.columns == [
            "factor_name",
            "status",
            "score",
            "p_value",
            "p_value_adjusted",
            "is_associated",
            "has_insufficient_data",
            "design_effect",
            "effective_groups",
            "df_num",
            "df_den",
        ]
        schema = result.factors.schema
        assert schema["factor_name"].base_type() == pl.Categorical
        assert schema["status"] == pl.Enum(_FACTOR_STATUS)
        for column in ("score", "p_value", "p_value_adjusted", "design_effect", "effective_groups", "df_num", "df_den"):
            assert schema[column] == pl.Float64
        assert schema["is_associated"] == pl.Boolean
        assert schema["has_insufficient_data"] == pl.Boolean

        assert isinstance(result.cells, pl.DataFrame)
        assert result.cells.columns == [
            "factor_name",
            "level",
            "class_name",
            "status",
            "is_actionable",
            "observed",
            "expected",
            "deficit",
            "lift",
            "p_value",
            "p_value_adjusted",
            "is_deviant",
            "is_sparse",
        ]
        schema = result.cells.schema
        for column in ("factor_name", "level", "class_name"):
            assert schema[column].base_type() == pl.Categorical
        assert schema["status"] == pl.Enum(_CELL_STATUS)
        assert schema["observed"] == pl.Int64
        for column in ("expected", "deficit", "lift", "p_value", "p_value_adjusted"):
            assert schema[column] == pl.Float64
        for column in ("is_actionable", "is_deviant", "is_sparse"):
            assert schema[column] == pl.Boolean
        assert result.cells.height == 4

        assert (result.test, result.group_source, result.group_level) == ("pearson", "rows", None)
        assert (result.n_rows, result.n_groups) == (10, 10)
        assert not hasattr(result, "insufficient_data")

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
        """Passing a raw AnnotatedDataset wraps it in Metadata."""
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

        result1 = Parity(score_threshold=0.1, p_value_threshold=0.5).evaluate(metadata)
        result2 = Parity(score_threshold=0.9, p_value_threshold=0.01).evaluate(metadata)

        correlated_1 = result1.factors.filter(pl.col("is_associated")).height
        correlated_2 = result2.factors.filter(pl.col("is_associated")).height
        assert correlated_1 >= correlated_2


@pytest.mark.required
class TestParityFunctional:
    """Test functional behavior of Parity class."""

    def test_correlated_factors(self):
        """Class and factor1 are perfectly correlated, so the factor is associated."""
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        metadata = to_metadata(factors, labels)
        result = Parity().evaluate(metadata)

        row = result.factors.filter(pl.col("factor_name") == "factor1")
        assert row["p_value"][0] < 0.05
        assert row["status"][0] == "associated"

    def test_a_constant_factor_is_untestable_not_independent(self):
        """A factor with one level cannot differ across levels; that is no evidence of parity."""
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        result = Parity().evaluate(metadata)

        row = result.factors.filter(pl.col("factor_name") == "factor1")
        assert row["status"][0] == "untestable"
        assert np.isnan(row["score"][0])
        assert np.isnan(row["p_value"][0])
        assert not row["is_associated"][0]
        assert (result.cells["status"] == "untestable").all()

    def test_quantized_factors(self):
        """Binning 'factor1' into two values gives the same test as the discrete equivalent."""
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 2}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        parity_obj = Parity()
        result1 = parity_obj.evaluate(metadata)

        discrete_dataset = {"factor2": [10] * 5 + [20] * 5}
        metadata = to_metadata(discrete_dataset, labels)
        result2 = parity_obj.evaluate(metadata)

        assert result1.factors["score"][0] == result2.factors["score"][0]
        assert result1.factors["p_value"][0] == result2.factors["p_value"][0]

    def test_overquantized_factors(self):
        """One bin leaves the factor constant over the dataset, which is untestable."""
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 1}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        result = Parity().evaluate(metadata)

        assert result.factors["status"][0] == "untestable"
        assert np.isnan(result.factors["score"][0])

    def test_underquantized_has_low_freqs(self, caplog):
        """Ten bins over ten points leave every cell too thin to trust."""
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = to_metadata(factors, labels, continuous_bincounts)

        with caplog.at_level(logging.WARNING):
            result = Parity().evaluate(metadata)
        assert len(caplog.text) > 0
        assert result.factors["has_insufficient_data"][0] is True
        assert result.cells["is_sparse"].all()

    def test_underquantized_has_repeated_low_freqs(self, caplog):
        labels = [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10)) + list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = to_metadata(factors, labels, continuous_bincounts)

        with caplog.at_level(logging.WARNING):
            result = Parity().evaluate(metadata)
        assert len(caplog.text) > 0
        assert result.factors["has_insufficient_data"][0] is True


@pytest.mark.required
class TestCellsNameTheirLevels:
    """A cell names its level and class, not a bare code."""

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
        """`illum_lux = 3` said nothing about which lighting to go and collect."""
        levels = Parity().evaluate(self._metadata()).cells.filter(pl.col("factor_name") == "illum_lux")["level"]
        assert levels.len() > 0
        for name in levels:
            assert not name.lstrip("-").isdigit(), f"{name!r} is still a bare code"

    def test_a_declared_cutoff_names_the_level_it_declared(self):
        """The level a user asked about, spelled the way they asked for it."""
        result = Parity().evaluate(
            self._metadata(continuous_factor_bins={"illum_lux": [-np.inf, 10.0, np.inf]}),
        )
        levels = result.cells.filter(pl.col("factor_name") == "illum_lux")["level"].unique().to_list()
        assert set(levels) == {"< 10", ">= 10"}

    def test_a_categorical_factors_level_is_named_by_its_value(self):
        rng = np.random.default_rng(11)
        weather = np.array(["sun"] * 100 + ["eclipse"] * 3)
        md = Metadata.from_factors({"weather": weather}, class_labels=rng.integers(0, 3, len(weather)))
        cells = Parity().evaluate(md).cells
        assert set(cells["level"].to_list()) == {"sun", "eclipse"}
        assert (cells.filter(pl.col("level") == "eclipse")["status"] == "too_thin").all()

    def test_classes_are_named_by_the_index2label(self):
        cells = Parity().evaluate(table_metadata([[10, 20], [30, 40]])).cells
        assert set(cells["class_name"].to_list()) == {"A", "B"}

    def test_a_container_with_no_record_still_reports_something(self):
        """A bare MetadataLike keeps the code, stringified -- what it always got."""
        rng = np.random.default_rng(5)
        n = 60
        bare = _bare(np.column_stack([rng.integers(0, 8, n), rng.integers(0, 3, n)]), rng.integers(0, 3, n))
        cells = Parity(assume_independent=True).evaluate(bare).cells
        assert set(cells.filter(pl.col("factor_name") == "f1")["level"].to_list()) == {"0", "1", "2"}


def _detections(images, per_image):
    """An object detection dataset of `images` images, each holding `per_image` detections."""
    from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget

    rng = np.random.default_rng(0)
    boxes = np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (per_image, 1))
    targets = [
        ObjectDetectionTarget(boxes, rng.integers(0, 2, per_image), np.full(per_image, 0.5)) for _ in range(images)
    ]
    return MockDataset(list(range(images)), targets)


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


# The reviewer's dependent-cell design: two planted deviations in a 4 x 5 table, the rest compensating.
_PLANTED = np.ones((4, 5))
_PLANTED[3, 0], _PLANTED[1, 2] = 0.2, 3.0
_PLANTED /= _PLANTED.sum()
_TRUE_LIFT = _PLANTED / (_PLANTED.sum(1, keepdims=True) * _PLANTED.sum(0, keepdims=True))


@pytest.mark.required
class TestCellsAndStatuses:
    def test_reviewer_low_v_deficit_is_underrepresented(self):
        """D1, validation row 6: no cell judgment depends on the factor's V."""
        out = Parity().from_metadata(table_metadata([[0, 60], [1500, 1440]]))
        row = out.cells.filter((pl.col("level") == "rare") & (pl.col("class_name") == "A"))
        assert row["status"][0] == "underrepresented"
        assert row["is_actionable"][0]
        assert row["observed"][0] == 0
        assert row["expected"][0] == pytest.approx(30)
        assert out.factors["score"][0] < 0.3

    def test_reviewer_sparse_deviation_is_too_thin(self):
        """Validation row 7: significant, and still never presented as a finding."""
        row = rare_cells(Parity().from_metadata(table_metadata([[4, 0], [0, 36]])))
        assert (row["status"] == "too_thin").all()
        assert row["is_deviant"].all()
        assert not row["is_actionable"].any()

    def test_significant_cell_inside_the_band_is_kept_not_actionable(self):
        """Lift 0.9 on 20000 rows is significant and inside the 1.25 band."""
        out = Parity().from_metadata(table_metadata([[4500, 5500], [5500, 4500]]))
        row = out.cells.filter((pl.col("level") == "rare") & (pl.col("class_name") == "A"))
        assert row["lift"][0] == pytest.approx(0.9)
        assert row["is_deviant"][0]
        assert row["status"][0] == "no_finding"
        assert not row["is_actionable"][0]
        assert out.factors["status"][0] == "weak"

        # Narrowing the band to 1.1 makes the same cell actionable.
        narrow = Parity(lift_threshold=1.1).from_metadata(table_metadata([[4500, 5500], [5500, 4500]]))
        assert narrow.cells["is_actionable"].all()

    def test_factor_status_inconclusive_vs_not_detected(self):
        """Not significant on 4 groups per level is not a negative; on 100 per level it is."""

        def status(groups_per_level, seed):
            rng = np.random.default_rng(seed)
            groups = np.repeat(np.arange(2 * groups_per_level), 50)
            out = Parity().from_metadata(_bare(groups % 2, rng.integers(0, 2, groups.size), groups))
            assert out.group_source == "item"
            assert out.test == "rao_scott_f"
            return out.factors.row(0, named=True)

        few, ample = status(4, 0), status(100, 0)
        assert few["p_value_adjusted"] > 0.05
        assert few["has_insufficient_data"]
        assert few["status"] == "inconclusive"
        assert ample["p_value_adjusted"] > 0.05
        assert not ample["has_insufficient_data"]
        assert ample["status"] == "not_detected"

    def test_cells_sort_by_factor_status_then_deficit(self):
        rng = np.random.default_rng(7)
        n = 600
        labels = rng.integers(0, 3, n)
        md = Metadata.from_factors(
            {
                "weather": np.array(["sun", "rain", "fog", "snow"])[(labels + (rng.random(n) < 0.4)) % 4],
                "angle": np.array(["low", "high"])[rng.integers(0, 2, n)],
                "site": np.array(["a", "b", "c", "rare"])[np.minimum(rng.integers(0, 40, n), 3)],
            },
            class_labels=labels,
        )
        cells = Parity().from_metadata(md).cells
        assert len(set(cells["status"].to_list())) >= 3

        names = cells["factor_name"].cast(pl.String).to_list()
        assert names == sorted(names)
        for (_,), frame in cells.group_by(["factor_name"], maintain_order=True):
            codes = frame["status"].to_physical().to_list()
            assert codes == sorted(codes)
            for (_,), same in frame.group_by(["status"], maintain_order=True):
                deficits = same["deficit"].to_list()
                assert deficits == sorted(deficits, reverse=True)

    def test_composite_label_axis_decodes_cells(self):
        rng = np.random.default_rng(0)
        n = 300
        md = Metadata.from_factors(
            {
                "weather": np.array(["sun", "rain"])[rng.integers(0, 2, n)],
                "time_of_day": np.array(["day", "night"])[rng.integers(0, 2, n)],
                "angle": np.array(["low", "high"])[rng.integers(0, 2, n)],
            },
            class_labels=rng.integers(0, 2, n),
        )
        cells = Parity(label=["weather", "time_of_day"]).from_metadata(md).cells
        assert set(cells["factor_name"].to_list()) == {"angle"}
        assert all(" × " in name for name in cells["class_name"].to_list())

    def test_false_actionable_rate_on_dependent_cells(self):
        """Validation row 8: among actionable cells, the share whose true lift is inside the band."""
        rng = np.random.default_rng(21)
        inside = (_TRUE_LIFT > 1 / 1.25) & (_TRUE_LIFT < 1.25)
        shares = []
        for _ in range(200):
            levels, classes = _expand(rng.multinomial(4000, _PLANTED.ravel()).reshape(_PLANTED.shape))
            cells = Parity(assume_independent=True).from_metadata(_bare(levels, classes)).cells
            flagged = cells.filter(pl.col("is_actionable"))
            level = flagged["level"].cast(pl.String).cast(pl.Int64).to_numpy()
            cls = flagged["class_name"].cast(pl.String).cast(pl.Int64).to_numpy()
            shares.append(inside[level, cls].sum() / max(flagged.height, 1))
        assert np.mean(shares) <= 0.05

    def test_planted_sequence_gap_is_underrepresented(self):
        """Validation row 5 (amended): the grouped cell path can produce a finding.

        The CI generator of tests/core/test_rao_scott.py at 200 sequences: one class at 0.15 of its share at
        one of three levels, 50 frames per sequence. R = 100 seeded runs. The sequences are the groups through
        ``item_indices`` (``group_source="item"``), which reaches the same table, variance floor and statuses
        as a tracking Metadata without building 100 of them.
        """
        rng = np.random.default_rng(7)
        reps, found, factor_p = 100, 0, []
        for _ in range(reps):
            level, classes, groups = seq_gap(rng, n_seq=200)
            out = Parity().from_metadata(_bare(level, classes, groups))
            assert out.group_source == "item"
            gap = out.cells.filter((pl.col("level") == "2") & (pl.col("class_name") == "0"))
            found += gap["status"][0] == "underrepresented"
            factor_p.append(out.factors["p_value_adjusted"][0])
        assert found / reps >= 0.9
        assert max(factor_p) <= 0.05


@pytest.mark.required
class TestSamplingUnit:
    def test_detection_rows_group_by_image(self):
        out = Parity(label="weather").from_metadata(_built(0))
        assert (out.group_source, out.group_level, out.n_groups, out.test) == ("level", "unit", 60, "rao_scott_f")
        assert out.n_rows == 60 * 40
        assert np.isfinite(out.factors["df_den"][0])

    def test_grouped_detection_null_is_calibrated(self):
        """Independent per-image factors read on detection rows reject at most at the nominal rate."""
        replicates = 40
        rejected = 0
        for seed in range(replicates):
            factors = Parity(label="weather").from_metadata(_built(seed)).factors
            rejected += factors.filter(pl.col("factor_name") == "brightness")["p_value"][0] < 0.05
        assert rejected / replicates <= bound(replicates)

    @pytest.mark.parametrize(("r", "n_groups", "replicates"), [(20, 2000, 100), (40, 4000, 30)])
    def test_bursty_null_has_no_actionable_cells(self, r, n_groups, replicates):
        """A per-group factor against classes that come in per-group bursts: no cell is actionable beyond the band."""
        rng = np.random.default_rng(14)
        groups = np.repeat(np.arange(n_groups), 10)
        tables, cells = 0, []
        for _ in range(replicates):
            own = rng.integers(0, r, n_groups)[groups]
            labels = np.where(rng.random(groups.size) < 0.5, own, rng.integers(0, r, groups.size))
            out = Parity().from_metadata(_bare(rng.integers(0, r, n_groups)[groups], labels, groups))
            assert out.test == "rao_scott_f"
            actionable = out.cells["is_actionable"].to_numpy()
            tables += actionable.any()
            cells.append(actionable)
        assert tables / replicates <= bound(replicates)
        assert np.mean(np.concatenate(cells)) <= bound(replicates)

    def test_image_view_is_rows(self):
        out = Parity(label="weather").from_metadata(_built(0).at("unit"))
        assert (out.group_source, out.group_level, out.test) == ("rows", None, "pearson")
        assert out.n_rows == out.n_groups == 60

    def test_from_factors_with_repeated_item_ids_is_grouped(self):
        """Rows sharing an item id are one sampling unit, not independent rows."""
        rng = np.random.default_rng(0)
        items = np.repeat(np.arange(200), 5)
        md = Metadata.from_factors(
            {"site": (items % 2).astype(np.intp)},
            class_labels=rng.integers(0, 2, items.size),
            index2label={0: "A", 1: "B"},
            item_indices=items,
        )
        out = Parity().from_metadata(md)
        assert (out.group_source, out.group_level, out.test) == ("level", md.item_level, "rao_scott_f")
        assert (out.n_rows, out.n_groups) == (1000, 200)

    def test_bare_container_requires_declaration(self):
        rng = np.random.default_rng(0)
        bare = _bare(rng.integers(0, 2, 40), rng.integers(0, 2, 40))
        with pytest.raises(ValueError, match="Cannot establish independent sampling units"):
            Parity().from_metadata(bare)
        out = Parity(assume_independent=True).from_metadata(bare)
        assert (out.group_source, out.test) == ("declared", "pearson")

    def test_state_records_group_source_and_lift(self):
        out = Parity(label="weather").from_metadata(_built(0))
        state = out.meta().state
        assert state["group_source"] == "level"
        assert state["lift_threshold"] == 1.25
        assert state["assume_independent"] is False
        assert state["test"] == "rao_scott_f"
        assert Parity(label="weather").from_metadata(_built(0).at("unit")).meta().state["test"] == "pearson"
