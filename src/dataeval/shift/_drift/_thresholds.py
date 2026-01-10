"""
Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/thresholds.py

Licensed under Apache Software License (Apache 2.0)
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from typing_extensions import Self


class Threshold(ABC):
    """A base class used to calculate lower and upper threshold values given one or multiple arrays.

    Any subclass should implement the abstract `thresholds` method.
    It takes an array or list of arrays and converts them into lower and upper threshold values, represented
    as a tuple of optional floats.

    A `None` threshold value is interpreted as if there is no upper or lower threshold.
    One or both values might be `None`.
    """

    _registry: ClassVar[dict[str, type[Self]]] = {}
    """Class registry lookup to get threshold subclass from threshold_type string"""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({str(vars(self))})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and other.__dict__ == self.__dict__

    def __init_subclass__(cls, threshold_type: str) -> None:
        Threshold._registry[threshold_type] = cls

    @abstractmethod
    def thresholds(self, data: np.ndarray) -> tuple[float | None, float | None]:
        """Returns lower and upper threshold values when given one or more np.ndarray instances.

        Parameters:
            data: np.ndarray
                An array of values used to calculate the thresholds on. This will most often represent a metric
                calculated on one or more sets of data, e.g. a list of F1 scores of multiple data chunks.
            kwargs: dict[str, Any]
                Optional keyword arguments passed to the implementing subclass.

        Returns:
            lower, upper: tuple[Optional[float], Optional[float]]
                The lower and upper threshold values. One or both might be `None`.
        """

    @classmethod
    def parse_object(cls, obj: dict[str, Any]) -> Self:
        """Parse object as :class:`Threshold`"""
        class_name = obj.pop("type", "")

        try:
            threshold_cls = cls._registry[class_name]
        except KeyError:
            accepted_values = ", ".join(map(repr, cls._registry))
            raise ValueError(f"Expected one of {accepted_values} for threshold type, but received '{class_name}'")

        return threshold_cls(**obj)

    def calculate(
        self,
        data: np.ndarray,
        lower_limit: float | None = None,
        upper_limit: float | None = None,
        override_using_none: bool = False,
        logger: logging.Logger | None = None,
    ) -> tuple[float | None, float | None]:
        """
        Calculate lower and upper threshold values with respect to the provided Threshold and value limits.

        Parameters
        ----------
        data : np.ndarray
            The data used by the Threshold instance to calculate the lower and upper threshold values.
            This will often be the values of a drift detection method or performance metric on chunks of reference
            data.
        lower_limit : float or None, default None
            An optional value that serves as a limit for the lower threshold value. Any calculated lower threshold
            values that end up below this limit will be replaced by this limit value.
            The limit is often a theoretical constraint enforced by a specific drift detection method or performance
            metric.
        upper_threshold_value_limit : float or None, default None
            An optional value that serves as a limit for the lower threshold value. Any calculated lower threshold
            values that end up below this limit will be replaced by this limit value.
            The limit is often a theoretical constraint enforced by a specific drift detection method or performance
            metric.
        override_using_none: bool, default False
            When set to True use None to override threshold values that exceed value limits.
            This will prevent them from being rendered on plots.
        logger: Optional[logging.Logger], default=None
            An optional Logger instance. When provided a warning will be logged when a calculated threshold value
            gets overridden by a threshold value limit.
        """

        lower_value, upper_value = self.thresholds(data)

        if lower_limit is not None and lower_value is not None and lower_value <= lower_limit:
            override_value = None if override_using_none else lower_limit
            if logger:
                logger.warning(
                    f"lower threshold value {lower_value} overridden by lower threshold value limit {override_value}"
                )
            lower_value = override_value

        if upper_limit is not None and upper_value is not None and upper_value >= upper_limit:
            override_value = None if override_using_none else upper_limit
            if logger:
                logger.warning(
                    f"upper threshold value {upper_value} overridden by upper threshold value limit {override_value}"
                )
            upper_value = override_value

        return lower_value, upper_value


class ConstantThreshold(Threshold, threshold_type="constant"):
    """A `Thresholder` implementation that returns a constant lower and or upper threshold value.

    Attributes:
        lower: Optional[float]
            The constant lower threshold value. Defaults to `None`, meaning there is no lower threshold.
        upper: Optional[float]
            The constant upper threshold value. Defaults to `None`, meaning there is no upper threshold.

    Raises:
        ValueError: raised when an argument was given using an incorrect type or name
        ValueError: raised when the ConstantThreshold could not be created using the given argument values

    Examples:
        >>> data = np.array(range(10))
        >>> t = ConstantThreshold(lower=None, upper=0.1)
        >>> t.calculate(data)
        (None, 0.1)
    """

    def __init__(self, lower: float | int | None = None, upper: float | int | None = None) -> None:
        """Creates a new ConstantThreshold instance.

        Args:
            lower: Optional[Union[float, int]], default=None
                The constant lower threshold value. Defaults to `None`, meaning there is no lower threshold.
            upper: Optional[Union[float, int]], default=None
                The constant upper threshold value. Defaults to `None`, meaning there is no upper threshold.

        Raises:
            ValueError: raised when an argument was given using an incorrect type or name
            ValueError: raised when the ConstantThreshold could not be created using the given argument values
        """
        self._validate_inputs(lower, upper)

        self.lower = lower
        self.upper = upper

    def thresholds(self, data: np.ndarray) -> tuple[float | None, float | None]:
        return self.lower, self.upper

    @staticmethod
    def _validate_inputs(lower: float | int | None = None, upper: float | int | None = None) -> None:
        if lower is not None and not isinstance(lower, float | int) or isinstance(lower, bool):
            raise ValueError(f"expected type of 'lower' to be 'float', 'int' or None but got '{type(lower).__name__}'")

        if upper is not None and not isinstance(upper, float | int) or isinstance(upper, bool):
            raise ValueError(f"expected type of 'upper' to be 'float', 'int' or None but got '{type(upper).__name__}'")

        # explicit None check is required due to special interpretation of the value 0.0 as False
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"lower threshold {lower} must be less than upper threshold {upper}")


class StandardDeviationThreshold(Threshold, threshold_type="standard_deviation"):
    """A Thresholder that offsets the mean of an array by a multiple of the standard deviation of the array values.

    This thresholder will take the aggregate of an array of values, the mean by default and add or subtract an offset
    to get the upper and lower threshold values.
    This offset is calculated as a multiplier, by default 3, times the standard deviation of the given array.

    Attributes:
        std_lower_multiplier: float
        std_upper_multiplier: float

    Examples:
        >>> data = np.array(range(10))
        >>> t = StandardDeviationThreshold(std_lower_multiplier=2, std_upper_multiplier=2.5)
        >>> t.calculate(data)
        (-1.2445626465380286, 11.680703308172536)
    """

    def __init__(
        self,
        std_lower_multiplier: float | int | None = 3,
        std_upper_multiplier: float | int | None = 3,
        offset_from: Callable[[np.ndarray], Any] = np.nanmean,
    ) -> None:
        """Creates a new StandardDeviationThreshold instance.

        Args:
            std_lower_multiplier: float, default=3
                The number the standard deviation of the input array will be multiplied with to form the lower offset.
                This value will be subtracted from the aggregate of the input array.
                Defaults to 3.
            std_upper_multiplier: float, default=3
                The number the standard deviation of the input array will be multiplied with to form the upper offset.
                This value will be added to the aggregate of the input array.
                Defaults to 3.
            offset_from: Callable[[np.ndarray], Any], default=np.nanmean
                A function that will be applied to the input array to aggregate it into a single value.
                Adding the upper offset to this value will yield the upper threshold, subtracting the lower offset
                will yield the lower threshold.
        """

        self._validate_inputs(std_lower_multiplier, std_upper_multiplier)

        self.std_lower_multiplier = std_lower_multiplier
        self.std_upper_multiplier = std_upper_multiplier
        self.offset_from = offset_from

    def thresholds(self, data: np.ndarray) -> tuple[float | None, float | None]:
        aggregate = self.offset_from(data)
        std = np.nanstd(data)

        lower_threshold = aggregate - std * self.std_lower_multiplier if self.std_lower_multiplier is not None else None

        upper_threshold = aggregate + std * self.std_upper_multiplier if self.std_upper_multiplier is not None else None

        return lower_threshold, upper_threshold

    @staticmethod
    def _validate_inputs(
        std_lower_multiplier: float | int | None = 3, std_upper_multiplier: float | int | None = 3
    ) -> None:
        if (
            std_lower_multiplier is not None
            and not isinstance(std_lower_multiplier, float | int)
            or isinstance(std_lower_multiplier, bool)
        ):
            raise ValueError(
                f"expected type of 'std_lower_multiplier' to be 'float', 'int' or None "
                f"but got '{type(std_lower_multiplier).__name__}'"
            )

        if std_lower_multiplier and std_lower_multiplier < 0:
            raise ValueError(f"'std_lower_multiplier' should be greater than 0 but got value {std_lower_multiplier}")

        if (
            std_upper_multiplier is not None
            and not isinstance(std_upper_multiplier, float | int)
            or isinstance(std_upper_multiplier, bool)
        ):
            raise ValueError(
                f"expected type of 'std_upper_multiplier' to be 'float', 'int' or None "
                f"but got '{type(std_upper_multiplier).__name__}'"
            )

        if std_upper_multiplier and std_upper_multiplier < 0:
            raise ValueError(f"'std_upper_multiplier' should be greater than 0 but got value {std_upper_multiplier}")
