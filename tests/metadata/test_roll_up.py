import numpy as np
import polars as pl
import pytest

from dataeval._metadata._aggregate import roll_up
from dataeval._metadata._reductions import expressions, identity_of
from dataeval.types import Aggregator

pytestmark = pytest.mark.required


def test_a_childless_target_gets_the_identity():
    rolled = roll_up(
        pl.DataFrame({"x": [1.0, -2.0, 3.0]}),
        np.array([0, 0, 2]),
        3,
        [pl.col("x").abs().sum().alias("s")],
        empty=0.0,
    )
    assert rolled.columns[0].to_list() == [3.0, 0.0, 3.0]
    assert rolled.childless == 1


def test_nan_is_missing_and_low_coverage_nulls():
    rolled = roll_up(
        pl.DataFrame({"x": [1.0, np.nan, np.nan, 4.0]}),
        np.array([0, 0, 0, 1]),
        2,
        [pl.col("x").mean().alias("m")],
        min_coverage=0.5,
    )
    assert rolled.columns[0].to_list() == [None, 4.0]
    assert rolled.uncovered == (1,)


def test_rows_without_a_target_take_no_part():
    rolled = roll_up(pl.DataFrame({"x": [5.0, 1.0]}), np.array([-1, 0]), 1, [pl.col("x").sum().alias("s")])
    assert rolled.columns[0].to_list() == [1.0]
    assert rolled.no_ancestor == 1


def test_a_temporal_reduction_reads_the_order_column():
    agg = Aggregator("variability", "unit", "sequence", ("x",), order_by="t")
    frame = pl.DataFrame({"x": [0.0, 2.0, 6.0], "t": [0.0, 1.0, 2.0]})
    rolled = roll_up(frame, np.zeros(3, dtype=np.intp), 1, expressions(agg), order="t")
    assert rolled.columns[0].to_list() == [pytest.approx(3.0)]  # mean(|2/1|, |4/1|)


def test_unique_keys_count_once_per_group():
    frame = pl.DataFrame({"x": [1.0, 1.0, 2.0]})
    rolled = roll_up(frame, np.array([0, 0, 0]), 1, [pl.col("x").sum().alias("s")], unique=pl.Series([7, 7, 8]))
    assert rolled.columns[0].to_list() == [3.0]


@pytest.mark.parametrize(
    ("how", "timed", "untimed"), [("variability", 2.0, None), ("changes", 2, 0), ("longest_run", 2, None)]
)
def test_a_group_with_no_ordering_beside_a_timed_one(how, timed, untimed):
    """The untimed group reads as an empty series, and does not take the timed one down with it.

    ``changes`` counts no transitions in an empty series, so it answers 0 rather than null.
    """
    agg = Aggregator(how, "unit", "sequence", ("x",), order_by="t")
    frame = pl.DataFrame({"x": [0.0, 2.0, 6.0, 6.0, 1.0, 5.0], "t": [0.0, 1.0, 2.0, 3.0, None, None]})
    rolled = roll_up(frame, np.array([0, 0, 0, 0, 1, 1]), 2, expressions(agg), order="t")
    assert rolled.columns[0].to_list() == [timed, untimed]


@pytest.mark.parametrize("how", ["variability", "changes", "longest_run"])
def test_a_temporal_reduction_over_no_rows_answers_the_identity(how):
    agg = Aggregator(how, "unit", "sequence", ("x",), order_by="t")
    frame = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "t": pl.Series([], dtype=pl.Float64)})
    rolled = roll_up(frame, np.zeros(0, dtype=np.intp), 2, expressions(agg), empty=identity_of(agg), order="t")
    assert rolled.columns[0].to_list() == [identity_of(agg)] * 2
