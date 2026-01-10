__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Flag
from typing import Any, Generic, TypeVar

TFlag = TypeVar("TFlag", bound=Flag)


class Calculator(Generic[TFlag], ABC):
    """
    Abstract base class for stateful statistics calculators.

    Calculators are responsible for computing specific categories of statistics
    on data. They are stateful and can cache intermediate results for efficiency.

    Each calculator:
    - Declares which flags it can handle via get_applicable_flags()
    - Receives raw datum + DatumProcessor instance during initialization
    - Computes statistics via compute() method
    - Can cache intermediate results as instance attributes or cached_properties

    Parameters
    ----------
    datum : Any
        The raw data element to compute statistics on.
    calculator : Calculator
        A calculator instance that provides preprocessed/transformed views of the datum.
    per_channel : bool, default False
        Whether to compute statistics per-channel (where applicable).
    """

    @abstractmethod
    def get_applicable_flags(self) -> TFlag:
        """
        Return which flags this calculator can handle.

        Returns
        -------
        Flag
            A flag enum value representing all flags this calculator can process.
            Typically a group flag like ImageStats.PIXEL or TextStats.SENTIMENT.
        """

    @abstractmethod
    def get_handlers(self) -> dict[TFlag, tuple[str, Callable[[], list[Any]]]]:
        """
        Return a mapping of flags to their corresponding stat names and handler functions.

        Each handler function should compute and return the statistic as a list of values.

        Returns
        -------
        dict[Flag, tuple[str, Callable[[], list[Any]]]]
            A dictionary mapping each flag this calculator can handle to a tuple:
            - stat name (str): The name of the statistic (e.g., "mean", "std").
            - handler (Callable): A function that computes and returns the statistic as a list.
        """

    def get_empty_values(self) -> dict[str, Any]:
        """
        Return empty values for statistics when they don't apply to certain channels.

        By default, all statistics use np.nan as the empty value. Override this method
        to provide custom empty values for specific statistics (e.g., arrays, strings).

        Returns
        -------
        dict[str, Any]
            A dictionary mapping stat names to their empty values.
            If a stat is not in this dict, np.nan is used as the default.

        Examples
        --------
        For a calculator with array-valued statistics:

        >>> def get_empty_values(self) -> dict[str, Any]:
        ...     return {
        ...         "center": [np.nan, np.nan],
        ...         "histogram": [np.nan] * 256,
        ...     }
        """
        return {}

    def compute(self, flags: TFlag) -> dict[str, list[Any]]:
        """
        Compute statistics for the requested flags.

        Parameters
        ----------
        flags : Flag
            The specific flags to compute. This will be a subset of the flags
            returned by get_applicable_flags(), representing what the user requested.

        Returns
        -------
        dict[str, list[Any]]
            Dictionary mapping stat names to lists of values. Each stat should return
            a list, where:
            - Single value per datum: list of length 1, e.g., [42.0]
            - Per-channel values: list of length N (number of channels), e.g., [41.0, 42.0, 43.0]

            The processor framework will reconcile outputs from multiple calculators.
        """
        stats: dict[str, list[Any]] = {}
        handlers = self.get_handlers()

        for flag, (stat_name, handler) in handlers.items():
            if flag in flags:
                stats[stat_name] = handler()

        return stats
