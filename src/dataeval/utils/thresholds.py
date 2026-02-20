"""
Threshold classes for computing lower and upper bounds from data.

Original source code derived from NannyML 0.13.0.

https://github.com/NannyML/nannyml/blob/main/nannyml/thresholds.py.

Licensed under Apache Software License (Apache 2.0)
"""

__all__ = [
    "ConstantThreshold",
    "IQRThreshold",
    "ModifiedZScoreThreshold",
    "ZScoreThreshold",
]

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.config import EPSILON
from dataeval.protocols import Threshold, ThresholdBounds, ThresholdLike, ThresholdLimits

_UNSET = object()


def _validate_numeric(name: str, value: float | None) -> None:
    """Check that *value* is ``float``, ``int``, or ``None`` (not ``bool``)."""
    if value is not None and not isinstance(value, float | int) or isinstance(value, bool):
        raise ValueError(f"expected type of '{name}' to be 'float', 'int' or None but got '{type(value).__name__}'")


def _validate_multiplier(name: str, value: float | None) -> None:
    """Validate a multiplier: must be numeric and non-negative."""
    _validate_numeric(name, value)
    if value is not None and value < 0:
        raise ValueError(f"'{name}' must be >= 0, got {value}")


class _Threshold(Threshold, ABC):
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
        """Return string representation of threshold."""
        attrs = {k: v for k, v in vars(self).items() if k not in ("lower_limit", "upper_limit") or v is not None}
        return f"{self.__class__.__name__}({attrs})"

    def __repr__(self) -> str:
        """Return class representation of threshold."""
        return str(self)

    def __eq__(self, other: object) -> bool:
        """Return equality of thresholds."""
        return isinstance(other, self.__class__) and other.__dict__ == self.__dict__

    def __init__(self, *, lower_limit: float | None = None, upper_limit: float | None = None) -> None:
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def __init_subclass__(cls, threshold_type: str) -> None:
        """Register threshold class."""
        _Threshold._registry[threshold_type] = cls

    @abstractmethod
    def _derive(self, data: NDArray[Any]) -> tuple[float | None, float | None]:
        """Derive lower and upper threshold values when given one or more np.ndarray instances.

        Parameters
        ----------
        data: NDArray[Any]
            An array of values used to calculate the thresholds on. This will most often represent a metric
            calculated on one or more sets of data, e.g. a list of F1 scores of multiple data chunks.

        Returns
        -------
        lower, upper: tuple[Optional[float], Optional[float]]
            The lower and upper threshold values. One or both might be `None`.
        """

    @classmethod
    def parse_object(cls, obj: dict[str, Any]) -> Self:
        """Instantiate a :class:`Threshold` subclass from a dictionary.

        The dictionary must contain a ``"type"`` key whose value matches a
        registered ``threshold_type`` string (e.g. ``"constant"``,
        ``"standard_deviation"``, ``"zscore"``).  The remaining key/value
        pairs are forwarded as keyword arguments to the matching subclass
        constructor.

        Parameters
        ----------
        obj : dict[str, Any]
            Dictionary representation of a threshold.  The ``"type"`` key
            is **popped** from the dict during parsing.

        Returns
        -------
        Threshold
            An instance of the matching :class:`Threshold` subclass.

        Raises
        ------
        ValueError
            If ``"type"`` is missing or does not match any registered
            threshold subclass.
        """
        class_name = obj.pop("type", "")

        try:
            threshold_cls = cls._registry[class_name]
        except KeyError as err:
            accepted_values = ", ".join(map(repr, cls._registry))
            raise ValueError(
                f"Expected one of {accepted_values} for threshold type, but received '{class_name}'",
            ) from err

        return threshold_cls(**obj)

    def __call__(self, data: NDArray[Any]) -> tuple[float | None, float | None]:
        """Calculate lower and upper threshold values, clamped by configured limits.

        Parameters
        ----------
        data : NDArray[Any]
            Array of values passed to :meth:`thresholds` for computation.

        Returns
        -------
        tuple[float | None, float | None]
            ``(lower, upper)`` threshold values after limit clamping.
        """
        lower, upper = self._derive(data)
        if self.lower_limit is not None and lower is not None:
            lower = max(lower, self.lower_limit)
        if self.upper_limit is not None and upper is not None:
            upper = min(upper, self.upper_limit)
        return lower, upper


class ConstantThreshold(_Threshold, threshold_type="constant"):
    """A `Threshold` implementation that returns a constant lower and or upper threshold value.

    Attributes
    ----------
    lower: Optional[float]
        The constant lower threshold value. Defaults to `None`, meaning there is no lower threshold.
    upper: Optional[float]
        The constant upper threshold value. Defaults to `None`, meaning there is no upper threshold.

    Raises
    ------
    ValueError: raised when an argument was given using an incorrect type or name
    ValueError: raised when the ConstantThreshold could not be created using the given argument values

    Examples
    --------
    >>> data = np.array(range(10))
    >>> t = ConstantThreshold(lower=None, upper=0.1)
    >>> t(data)
    (None, 0.1)
    """

    def __init__(self, lower: float | None = None, upper: float | None = None) -> None:
        """Create a new ConstantThreshold instance."""
        super().__init__()
        _validate_numeric("lower", lower)
        _validate_numeric("upper", upper)
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"lower threshold {lower} must be less than upper threshold {upper}")

        self.lower = lower
        self.upper = upper

    def _derive(self, data: NDArray[Any]) -> tuple[float | None, float | None]:  # noqa: ARG002
        """Return the constant lower and upper threshold values.

        The *data* argument is ignored; the values configured at construction time are returned directly.

        Parameters
        ----------
        data : NDArray[Any]
            Unused.  Accepted for interface compatibility.

        Returns
        -------
        tuple[float | None, float | None]
            ``(lower, upper)`` as set during initialisation.
        """
        return self.lower, self.upper


class ZScoreThreshold(_Threshold, threshold_type="zscore"):
    """Threshold based on z-score (standard deviation from mean).

    Flags values where ``|x - mean| / std > multiplier``.
    Supports asymmetric lower/upper multipliers.

    Parameters
    ----------
    multiplier : float or None, default 3.0
        Symmetric multiplier applied to both bounds. Overridden per-side
        by *lower_multiplier* / *upper_multiplier* when provided.
    lower_multiplier : float or None
        Override for the lower bound: ``mean - lower_multiplier * std``.
        ``None`` means no lower bound.
    upper_multiplier : float or None
        Override for the upper bound: ``mean + upper_multiplier * std``.
        ``None`` means no upper bound.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    >>> t = ZScoreThreshold(2.0)
    >>> lower, upper = t(data)
    """

    lower_multiplier: float | None
    upper_multiplier: float | None

    def __init__(
        self,
        multiplier: float | None = 3.0,
        *,
        lower_multiplier: float | None = _UNSET,  # type: ignore[assignment]
        upper_multiplier: float | None = _UNSET,  # type: ignore[assignment]
        lower_limit: float | None = None,
        upper_limit: float | None = None,
    ) -> None:
        super().__init__(lower_limit=lower_limit, upper_limit=upper_limit)
        self.lower_multiplier = multiplier if lower_multiplier is _UNSET else lower_multiplier
        self.upper_multiplier = multiplier if upper_multiplier is _UNSET else upper_multiplier
        _validate_multiplier("lower_multiplier", self.lower_multiplier)
        _validate_multiplier("upper_multiplier", self.upper_multiplier)

    def _derive(self, data: NDArray[Any]) -> tuple[float | None, float | None]:
        """Compute thresholds as ``mean ± multiplier * std``.

        Parameters
        ----------
        data : NDArray[Any]
            Array of values to compute thresholds from.

        Returns
        -------
        tuple[float | None, float | None]
            ``(mean - lower_multiplier * std, mean + upper_multiplier * std)``.
            Returns ``(None, None)`` when the standard deviation is
            effectively zero (``<= EPSILON``).  Either element is ``None``
            when its corresponding multiplier is ``None``.
        """
        std_val = np.nanstd(data)
        if std_val <= EPSILON:
            return None, None
        mean_val = np.nanmean(data)
        lower = float(mean_val - self.lower_multiplier * std_val) if self.lower_multiplier is not None else None
        upper = float(mean_val + self.upper_multiplier * std_val) if self.upper_multiplier is not None else None
        return lower, upper


class ModifiedZScoreThreshold(_Threshold, threshold_type="modzscore"):
    """Threshold based on modified z-score (median absolute deviation (MAD)).

    Uses median and MAD for robust outlier detection. The modified z-score is:
    ``0.6745 * |x - median| / MAD``

    Falls back to ``mean(|x - median|)`` when MAD <= EPSILON.

    Parameters
    ----------
    multiplier : float or None, default 3.5
        Symmetric multiplier applied to both bounds. Overridden per-side
        by *lower_multiplier* / *upper_multiplier* when provided.
    lower_multiplier : float or None
        Override for the lower bound. ``None`` means no lower bound.
    upper_multiplier : float or None
        Override for the upper bound. ``None`` means no upper bound.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    >>> t = ModifiedZScoreThreshold(3.5)
    >>> lower, upper = t(data)
    """

    lower_multiplier: float | None
    upper_multiplier: float | None

    def __init__(
        self,
        multiplier: float | None = 3.5,
        *,
        lower_multiplier: float | None = _UNSET,  # type: ignore[assignment]
        upper_multiplier: float | None = _UNSET,  # type: ignore[assignment]
        lower_limit: float | None = None,
        upper_limit: float | None = None,
    ) -> None:
        super().__init__(lower_limit=lower_limit, upper_limit=upper_limit)
        self.lower_multiplier = multiplier if lower_multiplier is _UNSET else lower_multiplier
        self.upper_multiplier = multiplier if upper_multiplier is _UNSET else upper_multiplier
        _validate_multiplier("lower_multiplier", self.lower_multiplier)
        _validate_multiplier("upper_multiplier", self.upper_multiplier)

    def _derive(self, data: NDArray[Any]) -> tuple[float | None, float | None]:
        """Compute thresholds as ``median ± multiplier * MAD / 0.6745``.

        The median absolute deviation (MAD) provides a robust measure of
        spread.  When ``MAD <= EPSILON`` the method falls back to
        ``mean(|x - median|)``.  If the fallback is also ``<= EPSILON``
        (i.e., all values are identical), ``(None, None)`` is returned.

        Parameters
        ----------
        data : NDArray[Any]
            Array of values to compute thresholds from.

        Returns
        -------
        tuple[float | None, float | None]
            ``(median - lower_multiplier * scale,
            median + upper_multiplier * scale)`` where
            ``scale = MAD / 0.6745``.  Returns ``(None, None)`` when
            spread is effectively zero.  Either element is ``None``
            when its corresponding multiplier is ``None``.
        """
        median_val = np.nanmedian(data)
        abs_diff = np.abs(data - median_val)
        mad = np.nanmedian(abs_diff)
        if mad <= EPSILON:
            mad = np.nanmean(abs_diff)
        if mad <= EPSILON:
            return None, None

        # modified z-score: 0.6745 * |x - median| / MAD > multiplier
        # equivalent to: |x - median| > multiplier * MAD / 0.6745
        scale = mad / 0.6745
        lower = float(median_val - self.lower_multiplier * scale) if self.lower_multiplier is not None else None
        upper = float(median_val + self.upper_multiplier * scale) if self.upper_multiplier is not None else None
        return lower, upper


class IQRThreshold(_Threshold, threshold_type="iqr"):
    """Threshold based on interquartile range.

    Outliers are values outside ``[Q1 - multiplier * IQR, Q3 + multiplier * IQR]``.
    Supports asymmetric lower/upper multipliers.

    Parameters
    ----------
    multiplier : float or None, default 1.5
        Symmetric multiplier applied to both bounds. Overridden per-side
        by *lower_multiplier* / *upper_multiplier* when provided.
    lower_multiplier : float or None
        Override for the lower bound: ``Q1 - lower_multiplier * IQR``.
        ``None`` means no lower bound.
    upper_multiplier : float or None
        Override for the upper bound: ``Q3 + upper_multiplier * IQR``.
        ``None`` means no upper bound.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    >>> t = IQRThreshold(1.5)
    >>> lower, upper = t(data)
    """

    lower_multiplier: float | None
    upper_multiplier: float | None

    def __init__(
        self,
        multiplier: float | None = 1.5,
        *,
        lower_multiplier: float | None = _UNSET,  # type: ignore[assignment]
        upper_multiplier: float | None = _UNSET,  # type: ignore[assignment]
        lower_limit: float | None = None,
        upper_limit: float | None = None,
    ) -> None:
        super().__init__(lower_limit=lower_limit, upper_limit=upper_limit)
        self.lower_multiplier = multiplier if lower_multiplier is _UNSET else lower_multiplier
        self.upper_multiplier = multiplier if upper_multiplier is _UNSET else upper_multiplier
        _validate_multiplier("lower_multiplier", self.lower_multiplier)
        _validate_multiplier("upper_multiplier", self.upper_multiplier)

    def _derive(self, data: NDArray[Any]) -> tuple[float | None, float | None]:
        """Compute thresholds as ``Q1 - mult * IQR`` and ``Q3 + mult * IQR``.

        The interquartile range (IQR) is ``Q3 - Q1``, computed with the
        midpoint method.  When ``IQR <= EPSILON`` (i.e., more than half
        the values are identical), ``(None, None)`` is returned.

        Parameters
        ----------
        data : NDArray[Any]
            Array of values to compute thresholds from.

        Returns
        -------
        tuple[float | None, float | None]
            ``(Q1 - lower_multiplier * IQR, Q3 + upper_multiplier * IQR)``.
            Returns ``(None, None)`` when the IQR is effectively zero.
            Either element is ``None`` when its corresponding multiplier
            is ``None``.
        """
        qrt = np.nanpercentile(data, q=(25, 75), method="midpoint")
        iqr_val = qrt[1] - qrt[0]
        if iqr_val <= EPSILON:
            return None, None
        lower = (qrt[0] - self.lower_multiplier * iqr_val) if self.lower_multiplier is not None else None
        upper = (qrt[1] + self.upper_multiplier * iqr_val) if self.upper_multiplier is not None else None
        return lower, upper


def _resolve_cls(threshold_type: str) -> type:
    """Look up a threshold class by its registered type name."""
    try:
        return _Threshold._registry[threshold_type]
    except KeyError as err:
        accepted = ", ".join(map(repr, _Threshold._registry))
        raise ValueError(f"Expected one of {accepted} for threshold type, but received {threshold_type!r}") from err


def _make_threshold(
    cls: type,
    bounds: ThresholdBounds | None,
    limits: ThresholdLimits | None = None,
) -> Threshold:
    """Construct a threshold instance from a resolved class and bounds value.

    Parameters
    ----------
    cls : type
        A registered :class:`_Threshold` subclass.
    bounds : ThresholdBounds or None
        Either a single ``float`` (symmetric multiplier), a
        ``tuple[float | None, float | None]`` (asymmetric bounds),
        or ``None`` to use the class defaults.
    limits : ThresholdLimits or None, default None
        Optional ``(lower_limit, upper_limit)`` clamp values.
    """
    limit_kwargs: dict[str, float | None] = {}
    if limits is not None:
        limit_kwargs = {"lower_limit": limits[0], "upper_limit": limits[1]}

    if bounds is None:
        return cls(**limit_kwargs)
    if isinstance(bounds, int | float):
        return cls(bounds, **limit_kwargs)
    # tuple[float | None, float | None]
    try:
        return cls(*bounds, **limit_kwargs)
    except TypeError:
        # Class uses keyword-only multipliers (e.g. ZScoreThreshold)
        return cls(lower_multiplier=bounds[0], upper_multiplier=bounds[1], **limit_kwargs)


_DEFAULT_THRESHOLD_TYPE = "modzscore"


def resolve_threshold(value: ThresholdLike | None = None) -> Threshold:
    """Convert a :data:`ThresholdLike` value to a :class:`Threshold` instance.

    Parameters
    ----------
    value : ThresholdLike or None, default None
        The threshold specification:

        - ``None``: default ``ModifiedZScoreThreshold()``
        - ``str``: named threshold with defaults (e.g. ``"zscore"``,
          ``"modzscore"``, ``"iqr"``, ``"constant"``)
        - ``float``: symmetric multiplier for the default method
        - ``tuple[float | None, float | None]``: asymmetric bounds
          for the default method
        - ``tuple[str, ThresholdBounds]``: named threshold with bounds,
          e.g. ``("zscore", 2.5)`` or ``("iqr", (1.0, 3.0))``
        - ``tuple[str, ThresholdBounds | None, ThresholdLimits]``: named
          threshold with bounds and limit clamping, e.g.
          ``("zscore", (1.0, 3.5), (0.0, 1.0))``. Pass ``None`` as
          bounds to use the class defaults.
        - ``tuple[ThresholdBounds | None, ThresholdLimits]``: default
          threshold with bounds and limit clamping, e.g.
          ``(2.5, (0.0, 1.0))`` or ``(None, (0.0, 1.0))`` for default
          multiplier.
        - ``Threshold``: returned as-is

    Returns
    -------
    Threshold
        A configured Threshold instance.

    Raises
    ------
    ValueError
        If a threshold type string does not match any registered
        threshold subclass.

    Examples
    --------
    >>> resolve_threshold()
    ModifiedZScoreThreshold({'lower_multiplier': 3.5, 'upper_multiplier': 3.5})

    >>> resolve_threshold("zscore")
    ZScoreThreshold({'lower_multiplier': 3.0, 'upper_multiplier': 3.0})

    >>> resolve_threshold(2.5)
    ModifiedZScoreThreshold({'lower_multiplier': 2.5, 'upper_multiplier': 2.5})

    >>> resolve_threshold((None, 5.0))
    ModifiedZScoreThreshold({'lower_multiplier': None, 'upper_multiplier': 5.0})

    >>> resolve_threshold(("zscore", 2.5))
    ZScoreThreshold({'lower_multiplier': 2.5, 'upper_multiplier': 2.5})

    >>> resolve_threshold(("iqr", (1.0, 3.0)))
    IQRThreshold({'lower_multiplier': 1.0, 'upper_multiplier': 3.0})

    >>> resolve_threshold(("constant", (0.0, 1.0)))
    ConstantThreshold({'lower': 0.0, 'upper': 1.0})

    >>> resolve_threshold(IQRThreshold(lower_multiplier=1.0, upper_multiplier=2.0))
    IQRThreshold({'lower_multiplier': 1.0, 'upper_multiplier': 2.0})

    >>> resolve_threshold(("zscore", 3.0, (0.0, 1.0)))
    ZScoreThreshold({'lower_limit': 0.0, 'upper_limit': 1.0, 'lower_multiplier': 3.0, 'upper_multiplier': 3.0})

    >>> resolve_threshold(("zscore", (1.0, 3.5), (0.0, 1.0)))
    ZScoreThreshold({'lower_limit': 0.0, 'upper_limit': 1.0, 'lower_multiplier': 1.0, 'upper_multiplier': 3.5})

    >>> resolve_threshold(("zscore", None, (0.0, 1.0)))
    ZScoreThreshold({'lower_limit': 0.0, 'upper_limit': 1.0, 'lower_multiplier': 3.0, 'upper_multiplier': 3.0})

    >>> resolve_threshold(("iqr", (1.0, 2.0), (None, 0.9)))
    IQRThreshold({'upper_limit': 0.9, 'lower_multiplier': 1.0, 'upper_multiplier': 2.0})

    >>> resolve_threshold((None, (0.0, 1.0)))
    ModifiedZScoreThreshold({'lower_limit': 0.0, 'upper_limit': 1.0, 'lower_multiplier': 3.5, 'upper_multiplier': 3.5})

    >>> resolve_threshold((2.5, (0.0, 1.0)))
    ModifiedZScoreThreshold({'lower_limit': 0.0, 'upper_limit': 1.0, 'lower_multiplier': 2.5, 'upper_multiplier': 2.5})
    """
    if isinstance(value, Threshold):
        return value
    if isinstance(value, str):
        return _resolve_cls(value)()
    if isinstance(value, tuple):
        if isinstance(value[0], str):
            name: str = value[0]
            if len(value) == 3:
                # tuple[str, ThresholdBounds | None, ThresholdLimits]
                bounds: ThresholdBounds | None = value[1]  # type: ignore[assignment]
                limits: ThresholdLimits = value[2]  # type: ignore[assignment]
                return _make_threshold(_resolve_cls(name), bounds, limits)
            # tuple[str, ThresholdBounds]
            bounds_only: ThresholdBounds = value[1]  # type: ignore[assignment]
            return _make_threshold(_resolve_cls(name), bounds_only)
        # Disambiguate: (bounds_or_None, (limit_lo, limit_hi)) vs (float|None, float|None)
        # If second element is a tuple, it's limits for the default threshold type.
        if len(value) == 2 and isinstance(value[1], tuple):
            # tuple[ThresholdBounds | None, ThresholdLimits]
            default_bounds: ThresholdBounds | None = value[0]  # type: ignore[assignment]
            default_limits: ThresholdLimits = value[1]  # type: ignore[assignment]
            return _make_threshold(_resolve_cls(_DEFAULT_THRESHOLD_TYPE), default_bounds, default_limits)
        # tuple[float | None, float | None]
        numeric_bounds: tuple[float | None, float | None] = value  # type: ignore[assignment]
        return _make_threshold(_resolve_cls(_DEFAULT_THRESHOLD_TYPE), numeric_bounds)
    if isinstance(value, int | float):
        return _resolve_cls(_DEFAULT_THRESHOLD_TYPE)(value)
    # None
    return _resolve_cls(_DEFAULT_THRESHOLD_TYPE)()
