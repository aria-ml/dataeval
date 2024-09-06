from __future__ import annotations

from enum import IntFlag, auto
from functools import reduce
from typing import Iterable, TypeVar, cast

TFlag = TypeVar("TFlag", bound=IntFlag)


class ImageStat(IntFlag):
    """
    Flags for calculating image and channel statistics
    """

    # HASHES
    XXHASH = auto()
    PCHASH = auto()
    # PROPERTIES
    WIDTH = auto()
    HEIGHT = auto()
    SIZE = auto()
    ASPECT_RATIO = auto()
    CHANNELS = auto()
    DEPTH = auto()
    # VISUALS
    BRIGHTNESS = auto()
    BLURRINESS = auto()
    MISSING = auto()
    ZERO = auto()
    # PIXEL STATS
    MEAN = auto()
    STD = auto()
    VAR = auto()
    SKEW = auto()
    KURTOSIS = auto()
    ENTROPY = auto()
    PERCENTILES = auto()
    HISTOGRAM = auto()
    # JOINT FLAGS
    ALL_HASHES = XXHASH | PCHASH
    ALL_PROPERTIES = WIDTH | HEIGHT | SIZE | ASPECT_RATIO | CHANNELS | DEPTH
    ALL_VISUALS = BRIGHTNESS | BLURRINESS | MISSING | ZERO
    ALL_PIXELSTATS = MEAN | STD | VAR | SKEW | KURTOSIS | ENTROPY | PERCENTILES | HISTOGRAM
    ALL_STATS = ALL_PROPERTIES | ALL_VISUALS | ALL_PIXELSTATS
    ALL = ALL_HASHES | ALL_STATS


def is_distinct(flag: IntFlag) -> bool:
    return (flag & (flag - 1) == 0) and flag != 0


def to_distinct(flag: TFlag) -> dict[TFlag, str]:
    """
    Returns a distinct set of all flags set on the input flag and their names

    NOTE: this is supported natively in Python 3.11, but for earlier versions we need
    to use a combination of list comprehension and bit fiddling to determine distinct
    flag values from joint aliases.
    """
    if isinstance(flag, Iterable):  # >= py311
        return {f: f.name.lower() for f in flag if f.name}
    else:  # < py311
        return {f: f.name.lower() for f in list(flag.__class__) if f & flag and is_distinct(f) and f.name}


def verify_supported(flag: TFlag, flags: TFlag | Iterable[TFlag]):
    supported = flags if isinstance(flags, flag.__class__) else cast(TFlag, reduce(lambda a, b: a | b, flags))  # type: ignore
    unsupported = flag & ~supported
    if unsupported:
        raise ValueError(f"Unsupported flags {unsupported} called.  Only {supported} flags are supported.")
