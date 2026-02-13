import math
from typing import Any

import numpy as np
import polars as pl
import pytest
from typing_extensions import Self

from dataeval.shift._drift._chunk import (
    Chunk,
    Chunker,
    CountBasedChunker,
    IndexChunk,
    PeriodBasedChunker,
    PeriodChunk,
    SizeBasedChunker,
)

rng = np.random.default_rng()


class MockChunk(Chunk):
    def __lt__(self, other: Self) -> bool:
        return True

    def __add__(self, other: Self) -> Self:
        return other

    def dict(self) -> dict[str, Any]:
        return {}


@pytest.fixture
def sample_index_chunk() -> IndexChunk:
    df = pl.DataFrame(rng.uniform(0, 100, size=(100, 4)), schema=list("ABCD"))
    chunk = IndexChunk(data=df, start_index=0, end_index=100)
    return chunk


@pytest.fixture
def sample_period_chunk() -> PeriodChunk:
    import datetime

    df = pl.DataFrame(rng.uniform(0, 100, size=(100, 4)), schema=list("ABCD"))
    # Create a simple period-like object for testing
    period = type(
        "Period",
        (),
        {
            "start_time": datetime.datetime(2020, 1, 1),
            "end_time": datetime.datetime(2020, 12, 31, 23, 59, 59, 999999),
            "__str__": lambda self: "2020",
        },
    )()
    chunk = PeriodChunk(data=df, period=period, chunk_size=100)
    return chunk


@pytest.fixture
def sample_chunk_data() -> pl.DataFrame:
    n_rows = 20 * 1008
    # Create date range using polars
    # Calculate end time: start + (n_rows - 1) * interval
    import datetime

    start_dt = datetime.datetime(2020, 1, 6)
    end_dt = start_dt + datetime.timedelta(minutes=10 * (n_rows - 1))

    data = pl.DataFrame({"ordered_at": pl.datetime_range(start=start_dt, end=end_dt, interval="10m", eager=True)})

    # Add week column
    data = data.with_columns((pl.col("ordered_at").dt.week() - 1).alias("week"))

    # Add period column
    data = data.with_columns(pl.lit("reference").alias("period"))
    data = data.with_columns(
        pl.when(pl.col("week") >= 11).then(pl.lit("analysis")).otherwise(pl.col("period")).alias("period"),
    )

    # Add random columns
    np.random.seed(13)
    data = data.with_columns(
        [
            pl.Series("f1", np.random.randn(n_rows)),
            pl.Series("f2", np.random.rand(n_rows)),
            pl.Series("f3", np.random.randint(4, size=n_rows)),
            pl.Series("f4", np.random.randint(20, size=n_rows), dtype=pl.Int64),
            pl.Series("y_pred", np.random.randint(2, size=n_rows)),
            pl.Series("y_true", np.random.randint(2, size=n_rows)),
        ],
    )
    data = data.with_columns(pl.col("ordered_at").alias("timestamp"))

    # Rule 1b is the shifted feature, 75% 0 instead of 50%
    rule1a = {2: 0, 3: 1}
    rule1b = {2: 0, 3: 0}
    data = data.with_columns(
        pl.when(pl.col("week") < 16)
        .then(pl.col("f3").replace(rule1a))
        .otherwise(pl.col("f3").replace(rule1b))
        .alias("f3"),
    )

    # Rule 2b is the shifted feature
    c1 = "white"
    c2 = "red"
    c3 = "green"
    c4 = "blue"

    rule2a = {
        0: c1, 1: c1, 2: c1, 3: c1, 4: c1, 5: c2, 6: c2, 7: c2, 8: c2, 9: c2,
        10: c3, 11: c3, 12: c3, 13: c3, 14: c3, 15: c4, 16: c4, 17: c4, 18: c4, 19: c4,
    }  # fmt: skip

    rule2b = {
        0: c1, 1: c1, 2: c1, 3: c1, 4: c1, 5: c2, 6: c2, 7: c2, 8: c2, 9: c2,
        10: c3, 11: c3, 12: c3, 13: c1, 14: c1, 15: c4, 16: c4, 17: c4, 18: c1, 19: c2,
    }  # fmt: skip

    data = data.with_columns(
        pl.when(pl.col("week") < 16)
        .then(pl.col("f4").replace_strict(rule2a, return_dtype=pl.Utf8))
        .otherwise(pl.col("f4").replace_strict(rule2b, return_dtype=pl.Utf8))
        .alias("f4"),
    )

    data = data.with_columns(pl.when(pl.col("week") >= 16).then(pl.col("f1") + 0.6).otherwise(pl.col("f1")).alias("f1"))

    data = data.with_columns(
        pl.when(pl.col("week") >= 16).then(pl.col("f2").sqrt()).otherwise(pl.col("f2")).alias("f2"),
    )

    return data


@pytest.mark.parametrize(
    "text",
    [
        "key=[0:100]",
        "data=pl.DataFrame(shape=(100, 4))",
        "start_index=0",
        "end_index=100",
    ],
)
def test_index_chunk_repr_should_contain_attribute(sample_index_chunk, text):
    sut = str(sample_index_chunk)
    assert text in sut


@pytest.mark.parametrize(
    "text",
    [
        "key=2020",
        "data=pl.DataFrame(shape=(100, 4))",
        "start_date=2020-01-01 00:00:00",
        "end_date=2020-12-31",
        "chunk_size=100",
    ],
)
def test_period_chunk_repr_should_contain_attribute(sample_period_chunk, text):
    sut = str(sample_period_chunk)
    assert text in sut


def test_chunk_len_should_return_data_length(sample_index_chunk):
    sut = len(sample_index_chunk)
    assert sut == len(sample_index_chunk.data)


def test_chunk_len_should_return_0_for_empty_chunk():
    sut = len(MockChunk(data=pl.DataFrame()))
    assert sut == 0


def test_chunker_should_log_warning_when_less_than_6_chunks(sample_chunk_data, caplog):
    import logging

    class SimpleChunker(Chunker[MockChunk]):
        def _split(self, data: pl.DataFrame) -> list[MockChunk]:
            return [MockChunk(data=data)]

    c = SimpleChunker()
    with caplog.at_level(logging.WARNING):
        _ = c.split(sample_chunk_data)

    assert "The resulting number of chunks is too low." in caplog.text


def test_chunker_should_set_index_boundaries(sample_chunk_data):
    class SimpleChunker(Chunker[IndexChunk]):
        def _split(self, data: pl.DataFrame) -> list[IndexChunk]:
            return [
                IndexChunk(data.slice(0, 6666), 0, 6665),
                IndexChunk(data.slice(6666, 6666), 6666, 13331),
                IndexChunk(data.slice(13332), 13332, 20159),
            ]

    chunker = SimpleChunker()
    sut = chunker.split(data=sample_chunk_data)
    assert sut[0].start_index == 0
    assert sut[0].end_index == 6665
    assert sut[1].start_index == 6666
    assert sut[1].end_index == 13331
    assert sut[2].start_index == 13332
    assert sut[2].end_index == 20159


def test_chunker_should_include_all_data_columns_by_default(sample_chunk_data):
    class SimpleChunker(Chunker[MockChunk]):
        def _split(self, data: pl.DataFrame) -> list[MockChunk]:
            return [MockChunk(data=data)]

    c = SimpleChunker()
    sut = c.split(sample_chunk_data)[0].data.columns
    assert sorted(sut) == sorted(sample_chunk_data.columns)


def test_chunker_should_fail_when_timestamp_column_is_not_provided(sample_chunk_data):
    c = PeriodBasedChunker(timestamp_column_name=None)  # type: ignore
    with pytest.raises(ValueError, match="timestamp_column_name must be provided"):
        c.split(sample_chunk_data)


def test_chunker_should_fail_when_timestamp_column_is_not_present(sample_chunk_data):
    c = PeriodBasedChunker(timestamp_column_name="foo")
    with pytest.raises(ValueError, match="timestamp column 'foo' not in columns"):
        c.split(sample_chunk_data)


def test_size_based_chunker_raises_exception_when_passed_nan_size(sample_chunk_data):
    with pytest.raises(ValueError, match="chunk_size=.* is invalid"):
        SizeBasedChunker(chunk_size="size?")  # type: ignore


def test_size_based_chunker_raises_exception_when_passed_negative_size(sample_chunk_data):
    with pytest.raises(ValueError, match="chunk_size=.* is invalid"):
        SizeBasedChunker(chunk_size=-1)


def test_size_based_chunker_raises_exception_when_passed_zero_size(sample_chunk_data):
    with pytest.raises(ValueError, match="chunk_size=.* is invalid"):
        SizeBasedChunker(chunk_size=0)


def test_size_based_chunker_works_with_empty_dataset():
    chunker = SizeBasedChunker(chunk_size=100)
    sut = chunker.split(pl.DataFrame(schema=["date", "timestamp", "f1", "f2", "f3", "f4"]))
    assert len(sut) == 0


def test_size_based_chunker_returns_chunks_of_required_size(sample_chunk_data):
    chunk_size = 1500
    chunker = SizeBasedChunker(chunk_size=chunk_size)
    sut = chunker.split(sample_chunk_data)
    assert len(sut[0]) == chunk_size
    assert len(sut) == math.ceil(sample_chunk_data.shape[0] / chunk_size)


def test_size_based_chunker_returns_last_chunk_that_is_partially_filled(sample_chunk_data):
    chunk_size = 3333
    expected_last_chunk_size = sample_chunk_data.shape[0] % chunk_size
    chunker = SizeBasedChunker(chunk_size)
    sut = chunker.split(sample_chunk_data)
    assert len(sut[-1]) == expected_last_chunk_size


def test_size_based_chunker_works_when_data_set_is_multiple_of_chunk_size(sample_chunk_data):
    chunk_size = 1000
    data = sample_chunk_data.slice(0, 20000)
    chunker = SizeBasedChunker(chunk_size)
    sut = []
    try:
        sut = chunker.split(data)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")

    assert len(sut[-1]) == chunk_size


def test_size_based_chunker_drops_last_incomplete_chunk_when_incomplete_set_to_drop(
    sample_chunk_data,
):
    chunk_size = 3333
    chunker = SizeBasedChunker(chunk_size, incomplete="drop")
    sut = chunker.split(sample_chunk_data)
    assert len(sut[-1]) == chunk_size
    assert len(sut) == 6


def test_size_based_chunker_keeps_last_incomplete_chunk_when_incomplete_set_to_keep(
    sample_chunk_data,
):
    chunk_size = 3333
    chunker = SizeBasedChunker(chunk_size, incomplete="keep")
    sut = chunker.split(sample_chunk_data)
    assert len(sut[-1]) == len(sample_chunk_data) % chunk_size
    assert len(sut) == 7


def test_size_based_chunker_appends_to_last_chunk_when_incomplete_set_to_append(
    sample_chunk_data,
):
    chunk_size = 3333
    chunker = SizeBasedChunker(chunk_size, incomplete="append")
    sut = chunker.split(sample_chunk_data)
    assert len(sut[-1]) == chunk_size + (len(sample_chunk_data) % chunk_size)
    assert len(sut) == 6


def test_size_based_chunker_assigns_observation_range_to_chunk_keys(sample_chunk_data):
    chunk_size = 1500
    last_chunk_start = (sample_chunk_data.shape[0] // chunk_size) * chunk_size
    last_chunk_end = sample_chunk_data.shape[0] - 1

    chunker = SizeBasedChunker(chunk_size=chunk_size)
    sut = chunker.split(sample_chunk_data)
    assert sut[0].key == "[0:1499]"
    assert sut[1].key == "[1500:2999]"
    assert sut[-1].key == f"[{last_chunk_start}:{last_chunk_end}]"


def test_count_based_chunker_raises_exception_when_passed_nan_size(sample_chunk_data):
    with pytest.raises(ValueError, match="given chunk_number .* is invalid"):
        _ = CountBasedChunker(chunk_number="size?")  # type: ignore


def test_count_based_chunker_raises_exception_when_passed_negative_size(sample_chunk_data):
    with pytest.raises(ValueError, match="given chunk_number .* is invalid"):
        _ = CountBasedChunker(chunk_number=-1)


def test_count_based_chunker_raises_exception_when_passed_zero_size(sample_chunk_data):
    with pytest.raises(ValueError, match="given chunk_number .* is invalid"):
        _ = CountBasedChunker(chunk_number=0)


def test_count_based_chunker_works_with_empty_dataset():
    chunker = CountBasedChunker(chunk_number=5)
    sut = chunker.split(pl.DataFrame(schema=["date", "timestamp", "f1", "f2", "f3", "f4"]))
    assert len(sut) == 0


def test_count_based_chunker_returns_chunks_of_required_size(sample_chunk_data):
    chunk_count = 5
    chunker = CountBasedChunker(chunk_number=chunk_count)
    sut = chunker.split(sample_chunk_data)
    assert len(sut[0]) == sample_chunk_data.shape[0] // chunk_count
    assert len(sut) == chunk_count


def test_count_based_chunker_assigns_observation_range_to_chunk_keys(sample_chunk_data):
    chunk_number = 5

    chunker = CountBasedChunker(chunk_number=chunk_number)
    sut = chunker.split(sample_chunk_data)
    assert sut[0].key == "[0:4031]"
    assert sut[1].key == "[4032:8063]"
    assert sut[-1].key == "[16128:20159]"


def test_period_based_chunker_raises_ValueError(sample_chunk_data):
    chunker = PeriodBasedChunker(offset="W", timestamp_column_name="timestamp")
    bad_dates = sample_chunk_data.clone()
    bad_dates = bad_dates.with_columns(pl.lit("foo").alias("timestamp"))
    # Polars raises InvalidOperationError instead of ValueError for invalid datetime operations
    with pytest.raises((ValueError, pl.exceptions.InvalidOperationError)):
        chunker.split(bad_dates)


@pytest.mark.parametrize(
    "chunker",
    [
        SizeBasedChunker(chunk_size=5000),
        CountBasedChunker(chunk_number=10),
        PeriodBasedChunker(offset="W", timestamp_column_name="timestamp"),
    ],
)
class TestChunkOperations:
    def test_size_based_chunker_sets_chunk_index(self, sample_chunk_data, chunker):
        sut = chunker.split(sample_chunk_data)
        assert all(chunk.index == index for index, chunk in enumerate(sut))

    def test_chunk_compare(self, sample_chunk_data, chunker):
        sut = chunker.split(sample_chunk_data)
        assert sut[0] < sut[1]
        assert sut[1] > sut[0]

    def test_chunk_add(self, sample_chunk_data, chunker):
        sut = chunker.split(sample_chunk_data)
        combined = sut[0] + sut[1]
        assert len(combined) == len(sut[0]) + len(sut[1])
        assert combined.start_index == sut[0].start_index
        assert combined.end_index == sut[1].end_index
        if isinstance(combined, PeriodChunk):
            assert combined.start_datetime == sut[0].start_datetime
            assert combined.end_datetime == sut[1].end_datetime


@pytest.mark.parametrize("chunker_cls", [SizeBasedChunker, CountBasedChunker])
def test_invalid_incomplete_param(chunker_cls):
    with pytest.raises(ValueError, match="incomplete=foo is invalid"):
        chunker_cls(5, incomplete="foo")
