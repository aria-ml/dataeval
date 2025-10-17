"""Module for flag enums that control function behavior."""

from __future__ import annotations

__all__ = ["ImageStats"]

from enum import Flag, auto


class ImageStats(Flag):
    """
    Flag enumeration for controlling image statistics computation.

    This enum provides fine-grained control over which statistics are computed
    during image processing. Flags can be combined using bitwise OR (|) to
    select multiple statistics.

    Individual Stats
    ----------------
    Pixel Statistics (computed on pixel values):

    - `PIXEL_MEAN` : Mean pixel value
    - `PIXEL_STD` : Standard deviation of pixel values
    - `PIXEL_VAR` : Variance of pixel values
    - `PIXEL_SKEW` : Skewness of pixel distribution
    - `PIXEL_KURTOSIS` : Kurtosis of pixel distribution
    - `PIXEL_ENTROPY` : Entropy of pixel histogram (depends on `PIXEL_HISTOGRAM`)
    - `PIXEL_MISSING` : Fraction of missing/NaN pixels
    - `PIXEL_ZEROS` : Fraction of zero pixels
    - `PIXEL_HISTOGRAM` : 256-bin histogram of pixel values

    Visual Statistics (computed on visual properties):

    - `VISUAL_BRIGHTNESS` : Brightness measure (depends on `VISUAL_PERCENTILES`)
    - `VISUAL_CONTRAST` : Contrast measure (depends on `VISUAL_PERCENTILES`)
    - `VISUAL_DARKNESS` : Darkness measure (depends on `VISUAL_PERCENTILES`)
    - `VISUAL_SHARPNESS` : Sharpness measure using edge detection
    - `VISUAL_PERCENTILES` : Percentiles (0, 25, 50, 75, 100)

    Dimension Statistics (computed on image/box dimensions):

    - `DIMENSION_OFFSET_X` : X-offset of bounding box
    - `DIMENSION_OFFSET_Y` : Y-offset of bounding box
    - `DIMENSION_WIDTH` : Width of image/box
    - `DIMENSION_HEIGHT` : Height of image/box
    - `DIMENSION_CHANNELS` : Number of color channels
    - `DIMENSION_SIZE` : Total pixel count
    - `DIMENSION_ASPECT_RATIO` : Width/height ratio
    - `DIMENSION_DEPTH` : Bit depth of image
    - `DIMENSION_CENTER` : Center coordinates [x, y]
    - `DIMENSION_DISTANCE_CENTER` : Distance from box center to image center
    - `DIMENSION_DISTANCE_EDGE` : Distance from box to nearest image edge
    - `DIMENSION_INVALID_BOX` : Whether bounding box is invalid

    Hash Statistics (computed on raw image data):

    - `HASH_XXHASH` : xxHash of raw image
    - `HASH_PCHASH` : Perceptual hash of image

    Convenience Groups
    ------------------
    Sub-groups:

    - `PIXEL_BASIC` : Mean, std, var
    - `PIXEL_DISTRIBUTION` : Skew, kurtosis, entropy, histogram
    - `VISUAL_BASIC` : Brightness, contrast, sharpness
    - `DIMENSION_BASIC` : Width, height, channels
    - `DIMENSION_OFFSET` : Offset X and Y
    - `DIMENSION_POSITION` : Center, distance to center, distance to edge

    Full Categories:

    - `PIXEL` : All pixel statistics
    - `VISUAL` : All visual statistics
    - `DIMENSION` : All dimension statistics
    - `HASH` : All hash statistics
    - `ALL` : All available statistics

    Examples
    --------
    Select specific statistics:

    >>> stats = ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD
    >>> stats = ImageStats.VISUAL_BRIGHTNESS | ImageStats.DIMENSION_WIDTH

    Use convenience groups:

    >>> stats = ImageStats.PIXEL  # All pixel stats
    >>> stats = ImageStats.PIXEL_BASIC | ImageStats.VISUAL  # Basic pixel & all visual

    Notes
    -----
    Some statistics have dependencies on others, the dependencies will be added
    automatically during processing.
    """

    # ===== PIXEL STATS =====
    PIXEL_MEAN = auto()
    PIXEL_STD = auto()
    PIXEL_VAR = auto()
    PIXEL_SKEW = auto()
    PIXEL_KURTOSIS = auto()
    PIXEL_ENTROPY = auto()  # depends on PIXEL_HISTOGRAM
    PIXEL_MISSING = auto()
    PIXEL_ZEROS = auto()
    PIXEL_HISTOGRAM = auto()  # dependency

    # ===== VISUAL STATS =====
    VISUAL_BRIGHTNESS = auto()  # depends on VISUAL_PERCENTILES
    VISUAL_CONTRAST = auto()  # depends on VISUAL_PERCENTILES
    VISUAL_DARKNESS = auto()  # depends on VISUAL_PERCENTILES
    VISUAL_SHARPNESS = auto()
    VISUAL_PERCENTILES = auto()  # dependency

    # ===== DIMENSION STATS =====
    DIMENSION_OFFSET_X = auto()
    DIMENSION_OFFSET_Y = auto()
    DIMENSION_WIDTH = auto()
    DIMENSION_HEIGHT = auto()
    DIMENSION_CHANNELS = auto()
    DIMENSION_SIZE = auto()
    DIMENSION_ASPECT_RATIO = auto()
    DIMENSION_DEPTH = auto()
    DIMENSION_CENTER = auto()
    DIMENSION_DISTANCE_CENTER = auto()
    DIMENSION_DISTANCE_EDGE = auto()
    DIMENSION_INVALID_BOX = auto()

    # ===== HASH STATS =====
    HASH_XXHASH = auto()
    HASH_PCHASH = auto()

    # ===== COARSE-GRAINED GROUPS =====
    # Full category groups
    PIXEL = (
        PIXEL_MEAN
        | PIXEL_STD
        | PIXEL_VAR
        | PIXEL_SKEW
        | PIXEL_KURTOSIS
        | PIXEL_ENTROPY
        | PIXEL_MISSING
        | PIXEL_ZEROS
        | PIXEL_HISTOGRAM
    )

    VISUAL = VISUAL_BRIGHTNESS | VISUAL_CONTRAST | VISUAL_DARKNESS | VISUAL_SHARPNESS | VISUAL_PERCENTILES

    DIMENSION = (
        DIMENSION_OFFSET_X
        | DIMENSION_OFFSET_Y
        | DIMENSION_WIDTH
        | DIMENSION_HEIGHT
        | DIMENSION_CHANNELS
        | DIMENSION_SIZE
        | DIMENSION_ASPECT_RATIO
        | DIMENSION_DEPTH
        | DIMENSION_CENTER
        | DIMENSION_DISTANCE_CENTER
        | DIMENSION_DISTANCE_EDGE
        | DIMENSION_INVALID_BOX
    )

    HASH = HASH_XXHASH | HASH_PCHASH

    # Convenience sub-groups
    PIXEL_BASIC = PIXEL_MEAN | PIXEL_STD | PIXEL_VAR
    PIXEL_DISTRIBUTION = PIXEL_SKEW | PIXEL_KURTOSIS | PIXEL_ENTROPY | PIXEL_HISTOGRAM

    VISUAL_BASIC = VISUAL_BRIGHTNESS | VISUAL_CONTRAST | VISUAL_SHARPNESS

    DIMENSION_BASIC = DIMENSION_WIDTH | DIMENSION_HEIGHT | DIMENSION_CHANNELS
    DIMENSION_OFFSET = DIMENSION_OFFSET_X | DIMENSION_OFFSET_Y
    DIMENSION_POSITION = DIMENSION_CENTER | DIMENSION_DISTANCE_CENTER | DIMENSION_DISTANCE_EDGE

    # Ultimate convenience
    ALL = PIXEL | VISUAL | DIMENSION | HASH


# Dependency mapping: stat -> required dependency
_STAT_DEPENDENCIES: dict[ImageStats, ImageStats] = {
    ImageStats.PIXEL_ENTROPY: ImageStats.PIXEL_HISTOGRAM,
    ImageStats.VISUAL_BRIGHTNESS: ImageStats.VISUAL_PERCENTILES,
    ImageStats.VISUAL_CONTRAST: ImageStats.VISUAL_PERCENTILES,
    ImageStats.VISUAL_DARKNESS: ImageStats.VISUAL_PERCENTILES,
}


def resolve_dependencies(flags: Flag) -> Flag:
    """
    Automatically resolve stat dependencies.

    Some statistics depend on others being computed first. This function
    ensures all required dependencies are included in the flag set.

    Parameters
    ----------
    flags : ImageStats
        The requested statistics flags.

    Returns
    -------
    ImageStats
        Flags with all dependencies resolved.

    Examples
    --------
    >>> flags = ImageStats.PIXEL_ENTROPY
    >>> resolved = resolve_dependencies(flags)
    >>> ImageStats.PIXEL_HISTOGRAM in resolved
    True

    >>> flags = ImageStats.VISUAL_BRIGHTNESS | ImageStats.VISUAL_CONTRAST
    >>> resolved = resolve_dependencies(flags)
    >>> ImageStats.VISUAL_PERCENTILES in resolved
    True
    """
    resolved = flags
    changed = True

    # Iterate until no new dependencies are added
    while changed:
        old_resolved = resolved
        for stat, dependency in _STAT_DEPENDENCIES.items():
            if stat in resolved:
                resolved |= dependency
        changed = resolved != old_resolved

    return resolved
