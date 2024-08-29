from enum import IntFlag, auto
from functools import reduce
from typing import Iterable, Set, TypeVar, Union, cast


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
    # STATS
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
    ALL_STATISTICS = MEAN | STD | VAR | SKEW | KURTOSIS | ENTROPY | PERCENTILES | HISTOGRAM
    ALL = ALL_HASHES | ALL_PROPERTIES | ALL_VISUALS | ALL_STATISTICS


TFlag = TypeVar("TFlag", bound=IntFlag)


def to_set(flag: TFlag) -> Set[TFlag]:
    """
    Returns a distinct set of all flags set on the input flag

    NOTE: this is supported natively in Python 3.11, but for earlier versions we need
    to use a combination of list comprehension and bit fiddling to determine distinct
    flag values from joint aliases.
    """
    if isinstance(flag, Iterable):  # >= py311
        return set(flag)
    else:  # < py311
        return {f for f in list(flag.__class__) if f & flag and (f & (f - 1) == 0) and f != 0}


def verify_supported(flag: TFlag, flags: Union[TFlag, Iterable[TFlag]]):
    supported = flags if isinstance(flags, flag.__class__) else cast(TFlag, reduce(lambda a, b: a | b, flags))  # type: ignore
    unsupported = flag & ~supported
    if unsupported:
        raise ValueError(f"Unsupported flags {unsupported} called.  Only {supported} flags are supported.")
