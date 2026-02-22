"""Chunking infrastructure for drift detection.

Supports fixed-size chunks, count-based chunks, and index-based groupings.
Chunkers return lists of index arrays rather than data slices, keeping the
protocol lightweight and decoupled from the data itself.

During ``fit()``, drift detectors normalize any chunker to a
:class:`SizeChunker` using the chunk size computed from the reference
data. This ensures that ``predict()`` uses the same chunk size regardless
of how many test samples are provided, keeping per-chunk statistics
comparable to the baseline established during fitting.
"""

__all__ = [
    "CountChunker",
    "IndexChunker",
    "SizeChunker",
    "resolve_chunker",
]

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import Chunker


class BaseChunker(Chunker, ABC):
    """Base class for numpy array chunkers."""

    @abstractmethod
    def split(self, n: int) -> list[NDArray[np.intp]]:
        """Split a range of sample indices into chunks.

        Parameters
        ----------
        n : int
            Number of samples in the array to be chunked.

        Returns
        -------
        list[NDArray[np.intp]]
            List of index arrays, each containing indices for one chunk.
        """

    def __call__(self, n: int) -> list[NDArray[np.intp]]:
        return self.split(n)


class CountChunker(BaseChunker):
    """Split array into a fixed number of chunks.

    Uses ``np.array_split`` to distribute samples as evenly as possible
    across exactly ``chunk_count`` chunks. When samples don't divide evenly,
    the first chunks each receive one extra sample rather than creating a
    tiny leftover chunk.

    Parameters
    ----------
    chunk_count : int
        Number of chunks to create.
    """

    def __init__(self, chunk_count: int) -> None:
        if not isinstance(chunk_count, int) or chunk_count <= 0:
            raise ValueError(f"chunk_count={chunk_count} is invalid - provide a positive integer")
        self.chunk_count = chunk_count

    def split(self, n: int) -> list[NDArray[np.intp]]:
        return [arr.astype(np.intp) for arr in np.array_split(np.arange(n), self.chunk_count)]


class SizeChunker(BaseChunker):
    """Split array into chunks of fixed size.

    Parameters
    ----------
    chunk_size : int
        Number of samples per chunk.
    incomplete : {"keep", "drop", "append"}, default "keep"
        How to handle leftover samples that don't fill a complete chunk.
        "keep" creates an incomplete final chunk, "drop" discards them,
        "append" adds them to the last complete chunk.
    """

    def __init__(
        self,
        chunk_size: int,
        incomplete: Literal["keep", "drop", "append"] = "keep",
    ) -> None:
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size={chunk_size} is invalid - provide a positive integer")
        if incomplete not in ("keep", "drop", "append"):
            raise ValueError(f"incomplete={incomplete} is invalid - must be 'keep', 'drop', or 'append'")
        self.chunk_size = chunk_size
        self.incomplete = incomplete

    def split(self, n: int) -> list[NDArray[np.intp]]:
        # Create complete chunks
        chunks = [
            np.arange(i, i + self.chunk_size, dtype=np.intp)
            for i in range(0, n, self.chunk_size)
            if i + self.chunk_size <= n
        ]

        # Handle leftover
        remainder = n % self.chunk_size
        if remainder > 0:
            leftover_start = self.chunk_size * (n // self.chunk_size)
            if self.incomplete == "keep":
                chunks.append(np.arange(leftover_start, n, dtype=np.intp))
            elif self.incomplete == "append" and chunks:
                # Extend the last chunk to include leftover samples
                chunks[-1] = np.arange(int(chunks[-1][0]), n, dtype=np.intp)
            # "drop": do nothing

        return chunks


class IndexChunker(BaseChunker):
    """Split array using user-provided index groupings.

    Parameters
    ----------
    indices : list[list[int]]
        Each inner list contains sample indices for one chunk.
        Example: ``[[0, 2, 4], [1, 3, 5]]`` creates 2 chunks.
    """

    def __init__(self, indices: list[list[int]]) -> None:
        if not indices:
            raise ValueError("indices must be a non-empty list of index lists")
        self._indices = indices

    def split(self, n: int) -> list[NDArray[np.intp]]:  # noqa: ARG002
        """Return stored index groups. The ``n`` argument is unused."""
        return [np.asarray(idx_list, dtype=np.intp) for idx_list in self._indices]


def resolve_chunker(
    chunker: BaseChunker | None = None,
    chunk_size: int | None = None,
    chunk_count: int | None = None,
    chunk_indices: list[list[int]] | None = None,
) -> BaseChunker | None:
    """Resolve various chunking specifications into a BaseChunker.

    Only one of the parameters should be provided. If multiple are given,
    priority is: chunker > chunk_indices > chunk_size > chunk_count.

    Parameters
    ----------
    chunker : BaseChunker or None
        An explicit chunker instance.
    chunk_size : int or None
        Create a SizeChunker with this size.
    chunk_count : int or None
        Create a CountChunker with this count.
    chunk_indices : list[list[int]] or None
        Create an IndexChunker with these index groups.

    Returns
    -------
    BaseChunker or None
        Resolved chunker, or None if no chunking was requested.
    """
    if chunker is not None:
        return chunker
    if chunk_indices is not None:
        return IndexChunker(chunk_indices)
    if chunk_size is not None:
        return SizeChunker(chunk_size)
    if chunk_count is not None:
        return CountChunker(chunk_count)
    return None
