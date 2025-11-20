"""
NannyML module providing intelligent splitting of data into chunks.

Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/chunk.py

Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

import copy
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeVar, cast

import pandas as pd
from pandas import Index, Period
from typing_extensions import Self

logger = logging.getLogger(__name__)


class Chunk(ABC):
    """A subset of data that acts as a logical unit during calculations."""

    KEYS: Sequence[str]

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        self.key: str
        self.data = data

        self.start_index: int = -1
        self.end_index: int = -1
        self.chunk_index: int = -1

    def __repr__(self) -> str:
        attr_str = ", ".join([f"{k}={v}" for k, v in self.dict().items()])
        return f"{self.__class__.__name__}(data=pd.DataFrame(shape={self.data.shape}), {attr_str})"

    def __len__(self) -> int:
        return self.data.shape[0]

    @abstractmethod
    def __add__(self, other: Any) -> Any: ...

    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...

    @abstractmethod
    def dict(self) -> dict[str, Any]: ...


class IndexChunk(Chunk):
    """Creates a new chunk.

    Parameters
    ----------
    data : DataFrame, required
        The data to be contained within the chunk.
    start_datetime: datetime
        The starting point in time for this chunk.
    end_datetime: datetime
        The end point in time for this chunk.
    """

    KEYS = ("key", "chunk_index", "start_index", "end_index")

    def __init__(
        self,
        data: pd.DataFrame,
        start_index: int,
        end_index: int,
    ) -> None:
        super().__init__(data)
        self.key = f"[{start_index}:{end_index}]"
        self.start_index: int = start_index
        self.end_index: int = end_index

    def __lt__(self, other: Self) -> bool:
        return self.end_index < other.start_index

    def __add__(self, other: Self) -> Self:
        a, b = (self, other) if self < other else (other, self)
        result = copy.deepcopy(a)
        result.data = pd.concat([a.data, b.data])
        result.end_index = b.end_index
        return result

    def dict(self) -> dict[str, Any]:
        return dict(zip(self.KEYS, (self.key, self.chunk_index, self.start_index, self.end_index)))


class PeriodChunk(Chunk):
    """Creates a new chunk.

    Parameters
    ----------
    data : DataFrame, required
        The data to be contained within the chunk.
    start_datetime: datetime
        The starting point in time for this chunk.
    end_datetime: datetime
        The end point in time for this chunk.
    chunk_size : int
        The size of the chunk.
    """

    KEYS = ("key", "chunk_index", "start_date", "end_date", "chunk_size")

    def __init__(self, data: pd.DataFrame, period: Period, chunk_size: int) -> None:
        super().__init__(data)
        self.key = str(period)
        self.start_datetime = period.start_time
        self.end_datetime = period.end_time
        self.chunk_size = chunk_size

    def __lt__(self, other: Self) -> bool:
        return self.end_datetime < other.start_datetime

    def __add__(self, other: Self) -> Self:
        a, b = (self, other) if self < other else (other, self)
        result = copy.deepcopy(a)
        result.data = pd.concat([a.data, b.data])
        result.end_index = b.end_index
        result.end_datetime = b.end_datetime
        result.chunk_size += b.chunk_size
        return result

    def dict(self) -> dict[str, Any]:
        return dict(
            zip(self.KEYS, (self.key, self.chunk_index, self.start_datetime, self.end_datetime, self.chunk_size))
        )


TChunk = TypeVar("TChunk", bound=Chunk)


class Chunker(Generic[TChunk]):
    """Base class for Chunker implementations.

    Inheriting classes will split a DataFrame into a list of Chunks.
    They will do this based on several constraints, e.g. observation timestamps, number of observations per Chunk
    or a preferred number of Chunks.
    """

    def split(self, data: pd.DataFrame) -> list[TChunk]:
        """Splits a given data frame into a list of chunks.

        This method provides a uniform interface across Chunker implementations to keep them interchangeable.

        After performing the implementation-specific `_split` method, there are some checks on the resulting chunk list.

        If the total number of chunks is low a warning will be written out to the logs.

        We dynamically determine the optimal minimum number of observations per chunk and then check if the resulting
        chunks contain at least as many. If there are any underpopulated chunks a warning will be written out in
        the logs.

        Parameters
        ----------
        data: DataFrame
            The data to be split into chunks

        Returns
        -------
        chunks: List[Chunk]
            The list of chunks

        """
        if data.shape[0] == 0:
            return []

        chunks = self._split(data)
        for chunk_index, chunk in enumerate(chunks):
            chunk.start_index = cast(int, chunk.data.index.min())
            chunk.end_index = cast(int, chunk.data.index.max())
            chunk.chunk_index = chunk_index

        if len(chunks) < 6:
            # TODO wording
            warnings.warn(
                "The resulting number of chunks is too low. "
                "Please consider splitting your data in a different way or continue at your own risk."
            )

        return chunks

    @abstractmethod
    def _split(self, data: pd.DataFrame) -> list[TChunk]: ...


class PeriodBasedChunker(Chunker[PeriodChunk]):
    """A Chunker that will split data into Chunks based on a date column in the data.

    Examples
    --------
    Chunk using monthly periods and providing a column name

    >>> from nannyml.chunk import PeriodBasedChunker
    >>> df = pd.read_parquet("/path/to/my/data.pq")
    >>> chunker = PeriodBasedChunker(timestamp_column_name="observation_date", offset="M")
    >>> chunks = chunker.split(data=df)

    Or chunk using weekly periods

    >>> from nannyml.chunk import PeriodBasedChunker
    >>> df = pd.read_parquet("/path/to/my/data.pq")
    >>> chunker = PeriodBasedChunker(timestamp_column_name=df["observation_date"], offset="W", minimum_chunk_size=50)
    >>> chunks = chunker.split(data=df)

    """

    def __init__(self, timestamp_column_name: str, offset: str = "W") -> None:
        """Creates a new PeriodBasedChunker.

        Parameters
        ----------
        timestamp_column_name : str
            The column name containing the timestamp to chunk on
        offset : str
            A frequency string representing a pandas.tseries.offsets.DateOffset.
            The offset determines how the time-based grouping will occur. A list of possible values
            can be found at <https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases>.
        """
        self.timestamp_column_name = timestamp_column_name
        self.offset = offset

    def _split(self, data: pd.DataFrame) -> list[PeriodChunk]:
        chunks = []
        if self.timestamp_column_name is None:
            raise ValueError("timestamp_column_name must be provided")
        if self.timestamp_column_name not in data:
            raise ValueError(f"timestamp column '{self.timestamp_column_name}' not in columns")

        grouped = data.groupby(pd.to_datetime(data[self.timestamp_column_name]).dt.to_period(self.offset))

        for k, v in grouped.groups.items():
            period, index = cast(Period, k), cast(Index, v)
            chunk = PeriodChunk(
                data=grouped.get_group(period),  # type: ignore | dataframe
                period=period,
                chunk_size=len(index),
            )
            chunks.append(chunk)

        return chunks


class SizeBasedChunker(Chunker[IndexChunk]):
    """A Chunker that will split data into Chunks based on the preferred number of observations per Chunk.

    Notes
    -----
    - Chunks are adjacent, not overlapping
    - There may be "incomplete" chunks, as the remainder of observations after dividing by `chunk_size`
      will form a chunk of their own.

    Examples
    --------
    Chunk using monthly periods and providing a column name

    >>> from nannyml.chunk import SizeBasedChunker
    >>> df = pd.read_parquet("/path/to/my/data.pq")
    >>> chunker = SizeBasedChunker(chunk_size=2000, incomplete="drop")
    >>> chunks = chunker.split(data=df)

    """

    def __init__(
        self,
        chunk_size: int,
        incomplete: Literal["append", "drop", "keep"] = "keep",
    ) -> None:
        """Create a new SizeBasedChunker.

        Parameters
        ----------
        chunk_size: int
            The preferred size of the resulting Chunks, i.e. the number of observations in each Chunk.
        incomplete: str, default='keep'
            Choose how to handle any leftover observations that don't make up a full Chunk.
            The following options are available:

            - ``'drop'``: drop the leftover observations
            - ``'keep'``: keep the incomplete Chunk (containing less than ``chunk_size`` observations)
            - ``'append'``: append leftover observations to the last complete Chunk (overfilling it)

            Defaults to ``'keep'``.

        Returns
        -------
        chunker: a size-based instance used to split data into Chunks of a constant size.

        """
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size={chunk_size} is invalid - provide an integer greater than 0")
        if incomplete not in ("append", "drop", "keep"):
            raise ValueError(f"incomplete={incomplete} is invalid - must be one of ['append', 'drop', 'keep']")

        self.chunk_size = chunk_size
        self.incomplete = incomplete

    def _split(self, data: pd.DataFrame) -> list[IndexChunk]:
        def _create_chunk(index: int, data: pd.DataFrame, chunk_size: int) -> IndexChunk:
            chunk_data = data.iloc[index : index + chunk_size]
            return IndexChunk(
                data=chunk_data,
                start_index=index,
                end_index=index + chunk_size - 1,
            )

        chunks = [
            _create_chunk(index=i, data=data, chunk_size=self.chunk_size)
            for i in range(0, data.shape[0], self.chunk_size)
            if i + self.chunk_size - 1 < len(data)
        ]

        # deal with unassigned observations
        if data.shape[0] % self.chunk_size != 0 and self.incomplete != "drop":
            incomplete_chunk = _create_chunk(
                index=self.chunk_size * (data.shape[0] // self.chunk_size),
                data=data,
                chunk_size=(data.shape[0] % self.chunk_size),
            )
            if self.incomplete == "append":
                chunks[-1] += incomplete_chunk
            else:
                chunks += [incomplete_chunk]

        return chunks


class CountBasedChunker(Chunker[IndexChunk]):
    """A Chunker that will split data into chunks based on the preferred number of total chunks.

    Notes
    -----
    - Chunks are adjacent, not overlapping
    - There may be "incomplete" chunks, as the remainder of observations after dividing by `chunk_size`
      will form a chunk of their own.

    Examples
    --------
    >>> from nannyml.chunk import CountBasedChunker
    >>> df = pd.read_parquet("/path/to/my/data.pq")
    >>> chunker = CountBasedChunker(chunk_number=100)
    >>> chunks = chunker.split(data=df)

    """

    def __init__(
        self,
        chunk_number: int,
        incomplete: Literal["append", "drop", "keep"] = "keep",
    ) -> None:
        """Creates a new CountBasedChunker.

        It will calculate the amount of observations per chunk based on the given chunk count.
        It then continues to split the data into chunks just like a SizeBasedChunker does.

        Parameters
        ----------
        chunk_number: int
            The amount of chunks to split the data in.
        incomplete: str, default='keep'
            Choose how to handle any leftover observations that don't make up a full Chunk.
            The following options are available:

            - ``'drop'``: drop the leftover observations
            - ``'keep'``: keep the incomplete Chunk (containing less than ``chunk_size`` observations)
            - ``'append'``: append leftover observations to the last complete Chunk (overfilling it)

            Defaults to ``'keep'``.

        Returns
        -------
        chunker: CountBasedChunker

        """
        if not isinstance(chunk_number, int) or chunk_number <= 0:
            raise ValueError(f"given chunk_number {chunk_number} is invalid - provide an integer greater than 0")
        if incomplete not in ("append", "drop", "keep"):
            raise ValueError(f"incomplete={incomplete} is invalid - must be one of ['append', 'drop', 'keep']")

        self.chunk_number = chunk_number
        self.incomplete: Literal["append", "drop", "keep"] = incomplete

    def _split(self, data: pd.DataFrame) -> list[IndexChunk]:
        chunk_size = data.shape[0] // self.chunk_number
        chunker = SizeBasedChunker(chunk_size, self.incomplete)
        return chunker.split(data=data)
