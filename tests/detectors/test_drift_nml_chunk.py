import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import Period
from typing_extensions import Self

from dataeval.detectors.drift._nml._chunk import (
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

    def _merge(self, other: Self) -> Self:
        return other

    def dict(self) -> dict[str, Any]:
        return {}


@pytest.fixture
def sample_index_chunk() -> IndexChunk:
    df = pd.DataFrame(rng.uniform(0, 100, size=(100, 4)), columns=pd.Series(list("ABCD")))
    chunk = IndexChunk(data=df, start_index=0, end_index=100)
    return chunk


@pytest.fixture
def sample_period_chunk() -> PeriodChunk:
    df = pd.DataFrame(rng.uniform(0, 100, size=(100, 4)), columns=pd.Series(list("ABCD")))
    chunk = PeriodChunk(data=df, period=Period("1/1/2020", "Y"), chunk_size=100)  # type: ignore - Period might be NaTType?
    return chunk


@pytest.fixture
def sample_chunk_data() -> pd.DataFrame:
    data = pd.DataFrame(
        pd.date_range(start="1/6/2020", freq="10min", periods=20 * 1008), columns=pd.Series(["ordered_at"])
    )
    data["week"] = data.ordered_at.dt.isocalendar().week - 1
    data["period"] = "reference"
    data.loc[data.week >= 11, ["period"]] = "analysis"
    np.random.seed(13)
    data["f1"] = np.random.randn(data.shape[0])
    data["f2"] = np.random.rand(data.shape[0])
    data["f3"] = np.random.randint(4, size=data.shape[0])
    data["f4"] = np.random.randint(20, size=data.shape[0])
    data["y_pred"] = np.random.randint(2, size=data.shape[0])
    data["y_true"] = np.random.randint(2, size=data.shape[0])
    data["timestamp"] = data["ordered_at"]

    # Rule 1b is the shifted feature, 75% 0 instead of 50%
    rule1a = {2: 0, 3: 1}
    rule1b = {2: 0, 3: 0}
    data.loc[data.week < 16, ["f3"]] = data.loc[data.week < 16, ["f3"]].replace(rule1a)
    data.loc[data.week >= 16, ["f3"]] = data.loc[data.week >= 16, ["f3"]].replace(rule1b)

    # Rule 2b is the shifted feature
    c1 = "white"
    c2 = "red"
    c3 = "green"
    c4 = "blue"

    rule2a = {
        0: c1,
        1: c1,
        2: c1,
        3: c1,
        4: c1,
        5: c2,
        6: c2,
        7: c2,
        8: c2,
        9: c2,
        10: c3,
        11: c3,
        12: c3,
        13: c3,
        14: c3,
        15: c4,
        16: c4,
        17: c4,
        18: c4,
        19: c4,
    }

    rule2b = {
        0: c1,
        1: c1,
        2: c1,
        3: c1,
        4: c1,
        5: c2,
        6: c2,
        7: c2,
        8: c2,
        9: c2,
        10: c3,
        11: c3,
        12: c3,
        13: c1,
        14: c1,
        15: c4,
        16: c4,
        17: c4,
        18: c1,
        19: c2,
    }

    data.loc[data.week < 16, ["f4"]] = data.loc[data.week < 16, ["f4"]].replace(rule2a)
    data.loc[data.week >= 16, ["f4"]] = data.loc[data.week >= 16, ["f4"]].replace(rule2b)

    data.loc[data.week >= 16, ["f1"]] = data.loc[data.week >= 16, ["f1"]] + 0.6
    data.loc[data.week >= 16, ["f2"]] = np.sqrt(data.loc[data.week >= 16, ["f2"]])

    return data


@pytest.mark.parametrize(
    "text",
    [
        "key=[0:100]",
        "data=pd.DataFrame(shape=(100, 4))",
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
        "data=pd.DataFrame(shape=(100, 4))",
        "start_date=2020-01-01 00:00:00",
        "end_date=2020-12-31 23:59:59.999999999",
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
    sut = len(MockChunk(data=pd.DataFrame()))
    assert sut == 0


def test_chunker_should_log_warning_when_less_than_6_chunks(sample_chunk_data, caplog):
    class SimpleChunker(Chunker[MockChunk]):
        def _split(self, data: pd.DataFrame) -> list[MockChunk]:
            return [MockChunk(data=data)]

    c = SimpleChunker()
    with pytest.warns(UserWarning, match="The resulting number of chunks is too low."):
        _ = c.split(sample_chunk_data)


def test_chunker_should_set_index_boundaries(sample_chunk_data):
    class SimpleChunker(Chunker[IndexChunk]):
        def _split(self, data: pd.DataFrame) -> list[IndexChunk]:
            return [
                IndexChunk(data.iloc[0:6666, :], 0, 6666),
                IndexChunk(data.iloc[6666:13332, :], 6666, 13332),
                IndexChunk(data.iloc[13332:, :], 13332, 20159),
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
        def _split(self, data: pd.DataFrame) -> list[MockChunk]:
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
    with pytest.raises(ValueError):
        SizeBasedChunker(chunk_size="size?")  # type: ignore


def test_size_based_chunker_raises_exception_when_passed_negative_size(sample_chunk_data):
    with pytest.raises(ValueError):
        SizeBasedChunker(chunk_size=-1)


def test_size_based_chunker_raises_exception_when_passed_zero_size(sample_chunk_data):
    with pytest.raises(ValueError):
        SizeBasedChunker(chunk_size=0)


def test_size_based_chunker_works_with_empty_dataset():
    chunker = SizeBasedChunker(chunk_size=100)
    sut = chunker.split(pd.DataFrame(columns=pd.Series(["date", "timestamp", "f1", "f2", "f3", "f4"])))
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
    data = sample_chunk_data.loc[0:19999, :]
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
    with pytest.raises(ValueError):
        _ = CountBasedChunker(chunk_number="size?")  # type: ignore


def test_count_based_chunker_raises_exception_when_passed_negative_size(sample_chunk_data):
    with pytest.raises(ValueError):
        _ = CountBasedChunker(chunk_number=-1)


def test_count_based_chunker_raises_exception_when_passed_zero_size(sample_chunk_data):
    with pytest.raises(ValueError):
        _ = CountBasedChunker(chunk_number=0)


def test_count_based_chunker_works_with_empty_dataset():
    chunker = CountBasedChunker(chunk_number=5)
    sut = chunker.split(pd.DataFrame(columns=pd.Series(["date", "timestamp", "f1", "f2", "f3", "f4"])))
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


@pytest.mark.parametrize(
    "chunker",
    [
        SizeBasedChunker(chunk_size=5000),
        CountBasedChunker(chunk_number=10),
        PeriodBasedChunker(offset="W", timestamp_column_name="timestamp"),
    ],
)
def test_size_based_chunker_sets_chunk_index(sample_chunk_data, chunker):
    sut = chunker.split(sample_chunk_data)
    assert all(chunk.chunk_index == chunk_index for chunk_index, chunk in enumerate(sut))
