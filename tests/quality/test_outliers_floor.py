"""The per-population floor of Outliers and the populations table."""

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.flags import ImageStats
from dataeval.quality import Outliers
from dataeval.utils.thresholds import ConstantThreshold


def _column(n, outlier=1000.0):
    values = np.random.default_rng(0).normal(10, 1, n)
    values[-1] = outlier
    return Metadata.from_factors({"x": values})


@pytest.mark.required
class TestFloor:
    def test_nineteen_rows_are_not_judged(self):
        result = Outliers().evaluate(_column(19))
        assert len(result) == 0
        row = result.populations.row(0, named=True)
        assert (row["metric_name"], row["n"], row["judged"], row["reason"]) == ("x", 19, False, "below 20 rows")

    def test_twenty_rows_are_judged(self):
        result = Outliers().evaluate(_column(20))
        assert result.data()["item_index"].to_list() == [19]
        assert result.populations.row(0, named=True)["judged"] is True

    def test_a_constant_threshold_is_not_floored(self):
        result = Outliers(outlier_threshold=ConstantThreshold(upper=100.0)).evaluate(_column(15))
        assert result.data()["item_index"].to_list() == [14]

    def test_no_spread_is_reported(self):
        result = Outliers().evaluate(Metadata.from_factors({"x": np.ones(30)}))
        assert result.populations.row(0, named=True)["reason"] == "no spread"

    def test_classwise_buckets_are_floored_and_listed(self):
        values = np.random.default_rng(0).normal(0, 1, 60)
        md = Metadata.from_factors({"x": values}, class_labels=np.repeat([0, 1, 2], [40, 15, 5]))
        pops = Outliers().evaluate(md).classwise(md).populations
        judged = dict(zip(pops["population"].cast(pl.Utf8), pops["judged"], strict=True))
        assert judged == {"class_label=0": True, "class_label=1": False, "class_label=2": False}

    def test_a_metric_with_no_real_value_is_not_listed(self):
        md = Metadata.from_factors({"x": np.random.default_rng(0).normal(10, 1, 30), "y": np.full(30, np.nan)})
        result = Outliers().evaluate(md)
        assert result.populations["metric_name"].cast(pl.Utf8).to_list() == ["x"]

    def test_an_output_without_populations_still_loads(self):
        result = Outliers().evaluate(_column(30))
        vars(result).pop("_populations", None)
        assert result.populations.columns == ["level", "metric_name", "method", "population", "n", "judged", "reason"]

    def test_a_level_view_narrows_the_populations(self, get_od_dataset):
        ds = get_od_dataset(list(np.random.default_rng(0).random((30, 3, 16, 16))), 2, True)
        result = Outliers(flags=ImageStats.PIXEL_MEAN).evaluate(ds, per_target=True)
        assert set(result.populations["level"].to_list()) == {None, "instance"}
        assert set(result.targets.populations["level"].to_list()) == {"instance"}
        assert set(result.items.populations["level"].to_list()) == {None}
