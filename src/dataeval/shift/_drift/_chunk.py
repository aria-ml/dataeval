"""
NannyML module providing intelligent splitting of data into chunks.

Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/chunk.py

Licensed under Apache Software License (Apache 2.0)
"""

import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeVar

import polars as pl
from typing_extensions import Self

_logger = logging.getLogger(__name__)


class Chunk(ABC):
    """A subset of data that acts as a logical unit during calculations."""

    KEYS: Sequence[str]

    def __init__(
        self,
        data: pl.DataFrame,
    ) -> None:
        self.key: str
        self.data = data

        self.index: int = -1
        self.start_index: int = -1
        self.end_index: int = -1

    def __repr__(self) -> str:
        attr_str = ", ".join([f"{k}={v}" for k, v in self.dict().items()])
        return f"{self.__class__.__name__}(data=pl.DataFrame(shape={self.data.shape}), {attr_str})"

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

    KEYS = ("key", "index", "start_index", "end_index")

    def __init__(
        self,
        data: pl.DataFrame,
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
        result.data = pl.concat([a.data, b.data])
        result.end_index = b.end_index
        return result

    def dict(self) -> dict[str, Any]:
        return dict(zip(self.KEYS, (self.key, self.index, self.start_index, self.end_index)))


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

    KEYS = ("key", "index", "start_date", "end_date", "chunk_size")

    def __init__(self, data: pl.DataFrame, period: Any, chunk_size: int) -> None:
        super().__init__(data)
        self.key = str(period)
        self.start_datetime = period.start_time
        self.end_datetime = period.end_time
        self.chunk_size = chunk_size

    def __lt__(self, other: Self) -> bool:
        return self.start_datetime < other.start_datetime

    def __add__(self, other: Self) -> Self:
        a, b = (self, other) if self < other else (other, self)
        result = copy.deepcopy(a)
        result.data = pl.concat([a.data, b.data])
        result.end_index = b.end_index
        result.end_datetime = b.end_datetime
        result.chunk_size += b.chunk_size
        return result

    def dict(self) -> dict[str, Any]:
        return dict(zip(self.KEYS, (self.key, self.index, self.start_datetime, self.end_datetime, self.chunk_size)))


TChunk = TypeVar("TChunk", bound=Chunk)


class Chunker(Generic[TChunk]):
    """Base class for Chunker implementations.

    Inheriting classes will split a DataFrame into a list of Chunks.
    They will do this based on several constraints, e.g. observation timestamps, number of observations per Chunk
    or a preferred number of Chunks.
    """

    def split(self, data: pl.DataFrame) -> list[TChunk]:
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
        for index, chunk in enumerate(chunks):
            chunk.index = index
            # Polars doesn't have an index attribute - start_index and end_index are set in chunk creation
            # Keep the values already set during chunk creation

        if len(chunks) < 6:
            # TODO wording
            _logger.warning(
                "The resulting number of chunks is too low. "
                "Please consider splitting your data in a different way or continue at your own risk."
            )

        return chunks

    @abstractmethod
    def _split(self, data: pl.DataFrame) -> list[TChunk]: ...


class PeriodBasedChunker(Chunker[PeriodChunk]):
    """
    A Chunker that will split data into Chunks based on a date column in the data.

    Parameters
    ----------
    timestamp_column_name : str
        The column name containing the timestamp to chunk on
    offset : str
        A frequency string representing a relative time offset.

    Notes
    -----
    The offset determines how the time-based grouping will occur. A list of possible values
    can be found at <https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.offset_by.html>.
    """

    def __init__(self, timestamp_column_name: str, offset: str = "W") -> None:
        """Creates a new PeriodBasedChunker."""
        self.timestamp_column_name = timestamp_column_name
        self.offset = offset

    def _split(self, data: pl.DataFrame) -> list[PeriodChunk]:
        chunks = []
        if self.timestamp_column_name is None:
            raise ValueError("timestamp_column_name must be provided")
        if self.timestamp_column_name not in data.columns:
            raise ValueError(f"timestamp column '{self.timestamp_column_name}' not in columns")

        # Use polars datetime operations with offset_by for period calculations
        # Convert offset to polars duration (W->1w, M->1mo, etc)
        offset_map = {"W": "1w", "M": "1mo", "D": "1d", "H": "1h", "Y": "1y"}
        polars_offset = offset_map.get(self.offset, self.offset)

        # Group by truncated timestamp and calculate period boundaries
        data_with_period = data.with_columns(
            [
                pl.col(self.timestamp_column_name).dt.truncate(polars_offset).alias("_period_start"),
                pl.col(self.timestamp_column_name)
                .dt.truncate(polars_offset)
                .dt.offset_by(polars_offset)
                .alias("_period_end"),
            ]
        )

        for (period_start, period_end), group_df in data_with_period.group_by(["_period_start", "_period_end"]):
            # Remove the temporary period columns
            group_data = group_df.drop("_period_start", "_period_end")

            # Create a period-like object with proper start and end times
            period = type(
                "Period",
                (),
                {
                    "start_time": period_start,
                    "end_time": period_end,
                    "__str__": lambda self, s=period_start: str(s),  # Capture period_start in closure
                },
            )()

            chunk = PeriodChunk(
                data=group_data,
                period=period,
                chunk_size=len(group_df),
            )
            chunks.append(chunk)

        # Sort chunks by start_datetime to ensure chronological order
        chunks.sort(key=lambda c: c.start_datetime)

        return chunks


class SizeBasedChunker(Chunker[IndexChunk]):
    """
    A Chunker that will split data into Chunks based on the preferred number of observations per Chunk.

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

    Notes
    -----
    - Chunks are adjacent, not overlapping
    - There may be "incomplete" chunks, as the remainder of observations after dividing by `chunk_size`
      will form a chunk of their own.
    """

    def __init__(
        self,
        chunk_size: int,
        incomplete: Literal["append", "drop", "keep"] = "keep",
    ) -> None:
        """Create a new SizeBasedChunker."""
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size={chunk_size} is invalid - provide an integer greater than 0")
        if incomplete not in ("append", "drop", "keep"):
            raise ValueError(f"incomplete={incomplete} is invalid - must be one of ['append', 'drop', 'keep']")

        self.chunk_size = chunk_size
        self.incomplete = incomplete

    def _split(self, data: pl.DataFrame) -> list[IndexChunk]:
        def _create_chunk(index: int, data: pl.DataFrame, chunk_size: int) -> IndexChunk:
            chunk_data = data.slice(index, chunk_size)
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
    """
    A Chunker that will split data into chunks based on the preferred number of total chunks.

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

    Notes
    -----
    - Chunks are adjacent, not overlapping
    - There may be "incomplete" chunks, as the remainder of observations after dividing by `chunk_size`
      will form a chunk of their own.
    """

    def __init__(
        self,
        chunk_number: int,
        incomplete: Literal["append", "drop", "keep"] = "keep",
    ) -> None:
        """Creates a new CountBasedChunker."""
        if not isinstance(chunk_number, int) or chunk_number <= 0:
            raise ValueError(f"given chunk_number {chunk_number} is invalid - provide an integer greater than 0")
        if incomplete not in ("append", "drop", "keep"):
            raise ValueError(f"incomplete={incomplete} is invalid - must be one of ['append', 'drop', 'keep']")

        self.chunk_number = chunk_number
        self.incomplete: Literal["append", "drop", "keep"] = incomplete

    def _split(self, data: pl.DataFrame) -> list[IndexChunk]:
        chunk_size = data.shape[0] // self.chunk_number
        chunker = SizeBasedChunker(chunk_size, self.incomplete)
        return chunker.split(data=data)
