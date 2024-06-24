from enum import Flag, auto
from typing import Union


class auto_all:
    def __get__(self, _, cls):
        return ~cls(0)


class ImageHash(Flag):
    XXHASH = auto()
    PCHASH = auto()
    ALL = auto_all()


class ImageProperty(Flag):
    WIDTH = auto()
    HEIGHT = auto()
    SIZE = auto()
    ASPECT_RATIO = auto()
    CHANNELS = auto()
    DEPTH = auto()
    ALL = auto_all()


class ImageVisuals(Flag):
    BRIGHTNESS = auto()
    BLURRINESS = auto()
    MISSING = auto()
    ZERO = auto()
    ALL = auto_all()


class ImageStatistics(Flag):
    MEAN = auto()
    STD = auto()
    VAR = auto()
    SKEW = auto()
    KURTOSIS = auto()
    ENTROPY = auto()
    PERCENTILES = auto()
    HISTOGRAM = auto()
    ALL = auto_all()


ImageStatsFlags = Union[ImageHash, ImageProperty, ImageVisuals, ImageStatistics]
LinterFlags = Union[ImageProperty, ImageVisuals, ImageStatistics]
