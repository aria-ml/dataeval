__all__ = []

from collections.abc import Sequence

from numpy.typing import NDArray


def _compare_keys(keys1: Sequence[str], keys2: Sequence[str]) -> None:
    """
    Raises error when two lists are not equivalent including ordering

    Parameters
    ----------
    keys1 : list of strings
        List of strings to compare
    keys2 : list of strings
        List of strings to compare

    Raises
    ------
    ValueError
        If lists do not have the same values, value counts, or ordering
    """

    if keys1 != keys2:
        raise ValueError(f"Metadata keys must be identical, got {keys1} and {keys2}")


def _validate_factors_and_data(factors: Sequence[str], data: NDArray) -> None:
    """
    Raises error when the number of factors and number of rows do not match

    Parameters
    ----------
    factors : list of strings
        List of factor names of size N
    data : NDArray
        Array of values with shape (M, N)

    Raises
    ------
    ValueError
        If the length of factors does not equal the length of the transposed data
    """
    if len(factors) != len(data.T):
        raise ValueError(f"Factors and data have mismatched lengths. Got {len(factors)} and {len(data.T)}")
