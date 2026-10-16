"""The metadata factor names DataEval assigns one fixed meaning."""

__all__ = []

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from dataeval.types._factors import FactorLevel


@dataclass(frozen=True)
class KnownFactor:
    """A metadata factor name with one fixed meaning, and the data that name may hold.

    If a known factor is a factor of a :class:`~dataeval.Metadata`, its data has passed
    the checks described here. Later analytics can rely on that without re-checking. See
    :doc:`/concepts/ExtrinsicFactors` for the recommended factors and what each is for.

    Names are lowercase and matched exactly. DataEval cannot check a unit, so using a
    known name is the supplier's promise to follow the definition: a ``timestamp`` in
    milliseconds since stream start passes the checks. Rename the factor if it means
    something else.

    Attributes
    ----------
    name : str
        The factor name, as it appears in the metadata.
    meaning : str
        One sentence saying what the factor measures.
    unit : str
        The unit its values are in.
    levels : tuple[FactorLevel, ...]
        The only levels the factor may be defined at.
    lower : float or None, default None
        The smallest valid value, or ``None`` for no lower bound.
    upper : float or None, default None
        The largest valid value, or ``None`` for no upper bound.
    lower_open : bool, default False
        Whether ``lower`` itself is invalid, so that values must be strictly greater.

    Notes
    -----
    A valid value is a finite real number. Integers count; booleans and strings do not.
    Missing values follow the column's ordinary rules, and being known adds no null rule.

    .. versionadded:: 1.2
    """

    name: str
    meaning: str
    unit: str
    levels: tuple[FactorLevel, ...]
    lower: float | None = None
    upper: float | None = None
    lower_open: bool = False


# Insertion order is the order of the concept page's table.
KNOWN_FACTORS: Mapping[str, KnownFactor] = MappingProxyType({
    f.name: f
    for f in (
        KnownFactor(
            "timestamp",
            "Wall-clock capture time of the unit.",
            "seconds since Unix epoch, UTC",
            ("unit",),
        ),
        KnownFactor(
            "frame_rate",
            "Average frame rate of the video.",
            "frames per second",
            ("sequence",),
            lower=0,
            lower_open=True,
        ),
        KnownFactor("video_duration", "Total duration of the video.", "seconds", ("sequence",), lower=0),
        KnownFactor(
            "latitude",
            "Latitude of the sensor (WGS 84).",
            "decimal degrees",
            ("sequence", "unit"),
            lower=-90,
            upper=90,
        ),
        KnownFactor(
            "longitude",
            "Longitude of the sensor (WGS 84).",
            "decimal degrees",
            ("sequence", "unit"),
            lower=-180,
            upper=180,
        ),
    )
})
"""The known metadata factors by name, read-only.

See :class:`KnownFactor` and :doc:`/concepts/ExtrinsicFactors`.

.. versionadded:: 1.2
"""
