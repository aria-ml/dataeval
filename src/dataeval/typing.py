"""
Common type hints used for interoperability with DataEval.
"""

__all__ = ["Array", "ArrayLike"]

from typing import Any, Iterator, Protocol, Sequence, TypeVar, Union, runtime_checkable


@runtime_checkable
class Array(Protocol):
    """
    Protocol for array objects providing interoperability with DataEval.

    Supports common array representations with popular libraries like
    PyTorch, Tensorflow and JAX, as well as NumPy arrays.

    Example
    -------
    >>> import numpy as np
    >>> import torch
    >>> from dataeval.typing import Array

    Create array objects

    >>> ndarray = np.random.random((10, 10))
    >>> tensor = torch.tensor([1, 2, 3])

    Check type at runtime

    >>> isinstance(ndarray, Array)
    True

    >>> isinstance(tensor, Array)
    True
    """

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __array__(self) -> Any: ...
    def __getitem__(self, key: Any, /) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...


TArray = TypeVar("TArray", bound=Array)

ArrayLike = Union[Sequence[Any], Array]
"""
Type alias for array-like objects used for interoperability with DataEval.

This includes native Python sequences, as well as objects that conform to
the `Array` protocol.
"""
