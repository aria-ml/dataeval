"""Module for flag enums that control function behavior."""

__all__ = ["ImageStats", "VideoStats"]

from enum import Flag, auto


class ImageStats(Flag):
    """
    Flag enumeration for controlling image statistics computation.

    This enum provides fine-grained control over which statistics are computed
    during image processing. Flags can be combined using bitwise OR (|) to
    select multiple statistics.

    Pixel statistics describe the *data* and visual statistics describe the *picture*.
    A pixel statistic reduces the values as they were stored, in their own units. A visual
    statistic reports where those values sit between black and white, so it always resolves
    a full-scale reference and reads against the 0-255 display range — which makes one
    picture answer identically whether it arrived as 8-bit, 16-bit or float, and makes the
    answer NaN for data with no such reference, such as an elevation or temperature band.
    See :doc:`/concepts/ImageStatistics`.

    Individual Stats
    ----------------
    Pixel Statistics (reduced over the values as stored):

    - `PIXEL_MEAN` : Mean pixel value
    - `PIXEL_STD` : Standard deviation of pixel values
    - `PIXEL_VAR` : Variance of pixel values
    - `PIXEL_SKEW` : Skewness of pixel distribution
    - `PIXEL_KURTOSIS` : Kurtosis of pixel distribution
    - `PIXEL_ENTROPY` : Entropy of pixel histogram (depends on `PIXEL_HISTOGRAM`)
    - `PIXEL_MISSING` : Fraction of missing/NaN pixels
    - `PIXEL_ZEROS` : Fraction of zero pixels
    - `PIXEL_HISTOGRAM` : 256-bin histogram of pixel values

    Visual Statistics (read against the 0-255 display range; NaN without a reference):

    - `VISUAL_BRIGHTNESS` : Lower-quartile display value (depends on `VISUAL_PERCENTILES`)
    - `VISUAL_CONTRAST` : Spread of the quartiles relative to their mean, so already
      scale-free (depends on `VISUAL_PERCENTILES`)
    - `VISUAL_DARKNESS` : Upper-quartile display value (depends on `VISUAL_PERCENTILES`)
    - `VISUAL_SHARPNESS` : Spread of edge magnitudes, via a 3x3 edge kernel
    - `VISUAL_PERCENTILES` : Display-range percentiles (0, 25, 50, 75, 100)

    Dimension Statistics (computed on image/box dimensions):

    - `DIMENSION_OFFSET_X` : X-offset of bounding box
    - `DIMENSION_OFFSET_Y` : Y-offset of bounding box
    - `DIMENSION_WIDTH` : Width of image/box
    - `DIMENSION_HEIGHT` : Height of image/box
    - `DIMENSION_CHANNELS` : Number of color channels
    - `DIMENSION_SIZE` : Total pixel count
    - `DIMENSION_ASPECT_RATIO` : Normalized aspect ratio between -1 (vertical) and 1 (horizontal)
    - `DIMENSION_DEPTH` : Bit depth of image
    - `DIMENSION_CENTER` : Center coordinates [x, y]
    - `DIMENSION_DISTANCE_CENTER` : Distance from box center to image center
    - `DIMENSION_DISTANCE_EDGE` : Distance from box to nearest image edge
    - `DIMENSION_INVALID_BOX` : Whether bounding box is invalid
    - `DIMENSION_REL_OFFSET_X` : X-offset of bounding box as a fraction of image width
    - `DIMENSION_REL_OFFSET_Y` : Y-offset of bounding box as a fraction of image height
    - `DIMENSION_REL_WIDTH` : Box width as a fraction of image width
    - `DIMENSION_REL_HEIGHT` : Box height as a fraction of image height
    - `DIMENSION_REL_SIZE` : Box area as a fraction of image area
    - `DIMENSION_REL_DISTANCE_CENTER` : Distance to the image center as a fraction of the half-diagonal
    - `DIMENSION_REL_DISTANCE_EDGE` : Distance to the nearest edge as a fraction of that edge's dimension

    Hash Statistics (computed on raw image data):

    - `HASH_XXHASH` : xxHash of raw image
    - `HASH_PHASH` : Perceptual hash of image
    - `HASH_DHASH` : Difference/gradient hash of image
    - `HASH_PHASH_D4` : Perceptual hash with D4 symmetry (rotation/flip invariant)
    - `HASH_DHASH_D4` : Difference hash with D4 symmetry (rotation/flip invariant)

    Convenience Groups
    ------------------
    Sub-groups:

    - `PIXEL_BASIC` : Mean, std, var
    - `PIXEL_DISTRIBUTION` : Skew, kurtosis, entropy, histogram
    - `VISUAL_BASIC` : Brightness, contrast, sharpness
    - `DIMENSION_BASIC` : Width, height, channels
    - `DIMENSION_OFFSET` : Offset X and Y
    - `DIMENSION_POSITION` : Center, distance to center, distance to edge
    - `DIMENSION_RELATIVE` : Box geometry relative to the image (the seven `DIMENSION_REL_*` stats)
    - `DIMENSION_BOX` : Width, height, aspect ratio, size, offset, distance to center/edge,
      invalid box, and their relative forms
    - `HASH_DUPLICATES_BASIC` : Standard duplicate detection (xxhash + phash + dhash)
    - `HASH_DUPLICATES_D4` : Rotation/flip-invariant detection (xxhash + phash_d4 + dhash_d4)

    Full Categories:

    - `NONE` : No statistics
    - `PIXEL` : All pixel statistics
    - `VISUAL` : All visual statistics
    - `DIMENSION` : All dimension statistics
    - `NO_HASH`: All non-hash statistics
    - `HASH` : All hash statistics
    - `ALL` : All available statistics

    Notes
    -----
    Some statistics have dependencies on others, the dependencies will be added
    automatically during processing.

    Examples
    --------
    Select specific statistics:

    >>> stats = ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD
    >>> stats = ImageStats.VISUAL_BRIGHTNESS | ImageStats.DIMENSION_WIDTH

    Use convenience groups:

    >>> stats = ImageStats.PIXEL  # All pixel stats
    >>> stats = ImageStats.PIXEL_BASIC | ImageStats.VISUAL  # Basic pixel & all visual
    """

    # ===== NONE (no stats) =====
    NONE = 0

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
    DIMENSION_REL_OFFSET_X = auto()
    DIMENSION_REL_OFFSET_Y = auto()
    DIMENSION_REL_WIDTH = auto()
    DIMENSION_REL_HEIGHT = auto()
    DIMENSION_REL_SIZE = auto()
    DIMENSION_REL_DISTANCE_CENTER = auto()
    DIMENSION_REL_DISTANCE_EDGE = auto()

    # ===== HASH STATS =====
    HASH_XXHASH = auto()
    HASH_PHASH = auto()
    HASH_PHASH_D4 = auto()
    HASH_DHASH = auto()
    HASH_DHASH_D4 = auto()

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
        | DIMENSION_REL_OFFSET_X
        | DIMENSION_REL_OFFSET_Y
        | DIMENSION_REL_WIDTH
        | DIMENSION_REL_HEIGHT
        | DIMENSION_REL_SIZE
        | DIMENSION_REL_DISTANCE_CENTER
        | DIMENSION_REL_DISTANCE_EDGE
    )

    HASH = HASH_XXHASH | HASH_PHASH | HASH_DHASH | HASH_PHASH_D4 | HASH_DHASH_D4

    # Convenience sub-groups
    PIXEL_BASIC = PIXEL_MEAN | PIXEL_STD | PIXEL_VAR
    PIXEL_DISTRIBUTION = PIXEL_SKEW | PIXEL_KURTOSIS | PIXEL_ENTROPY | PIXEL_HISTOGRAM

    VISUAL_BASIC = VISUAL_BRIGHTNESS | VISUAL_CONTRAST | VISUAL_SHARPNESS

    DIMENSION_BASIC = DIMENSION_WIDTH | DIMENSION_HEIGHT | DIMENSION_CHANNELS | DIMENSION_ASPECT_RATIO
    DIMENSION_OFFSET = DIMENSION_OFFSET_X | DIMENSION_OFFSET_Y
    DIMENSION_POSITION = DIMENSION_CENTER | DIMENSION_DISTANCE_CENTER | DIMENSION_DISTANCE_EDGE
    DIMENSION_RELATIVE = (
        DIMENSION_REL_OFFSET_X
        | DIMENSION_REL_OFFSET_Y
        | DIMENSION_REL_WIDTH
        | DIMENSION_REL_HEIGHT
        | DIMENSION_REL_SIZE
        | DIMENSION_REL_DISTANCE_CENTER
        | DIMENSION_REL_DISTANCE_EDGE
    )
    DIMENSION_BOX = (
        DIMENSION_WIDTH
        | DIMENSION_HEIGHT
        | DIMENSION_ASPECT_RATIO
        | DIMENSION_SIZE
        | DIMENSION_OFFSET
        | DIMENSION_DISTANCE_CENTER
        | DIMENSION_DISTANCE_EDGE
        | DIMENSION_INVALID_BOX
        | DIMENSION_RELATIVE
    )

    HASH_DUPLICATES_BASIC = HASH_XXHASH | HASH_PHASH | HASH_DHASH
    HASH_DUPLICATES_D4 = HASH_XXHASH | HASH_PHASH_D4 | HASH_DHASH_D4

    NO_HASH = PIXEL | VISUAL | DIMENSION

    # Ultimate convenience
    ALL = NO_HASH | HASH


class VideoStats(Flag):
    """
    Flag enumeration for controlling per-video statistics computation.

    Each statistic describes one whole video, so a result has one row per video. Every
    selected statistic is computed in a single walk of each video's frames. Flags can be
    combined using bitwise OR (|) to select multiple statistics.

    Individual Stats
    ----------------
    Descriptive Statistics (computed from frame timestamps, never from pixels):

    - `DESCRIPTIVE_FRAME_COUNT` : Frames the stream yielded (``n_frames``)
    - `DESCRIPTIVE_DURATION` : Seconds from the first frame to the end of the last
      (``video_duration``)
    - `DESCRIPTIVE_FRAME_RATE` : Frames per second, ``n_frames / video_duration``
      (``frame_rate``)

    Redundancy Statistics (perceptual hash of each frame, compared with its predecessor):

    - `REDUNDANCY_FRACTION` : Share of frames that carry nothing new over the frame before
      (``redundant_fraction``)
    - `REDUNDANCY_LONGEST_RUN` : Frames in the longest stretch of such frames
      (``longest_run``)

    Ego-motion Statistics (the per-frame camera motion of :func:`~dataeval.core.ego_stats`,
    summarized over the video; need OpenCV, from the ``dataeval[opencv]`` extra):

    - `EGO_PAN_SPEED` : Median pan/tilt speed (``pan_speed_median``)
    - `EGO_ZOOM_RATE` : Median zoom rate (``zoom_rate_median``)
    - `EGO_ROLL_RATE` : Median roll rate (``roll_rate_median``)
    - `EGO_SHEAR_RATE` : Median shear rate (``shear_rate_median``)
    - `EGO_MOVER_FRAC` : Median share of flow the camera does not explain
      (``mover_frac_median``)
    - `EGO_MOVER_SPEED` : Median speed of that independent motion (``mover_speed_median``)
    - `EGO_PAN_X_VARIABILITY` : Mean absolute rate of change of horizontal pan
      (``pan_x_variability``)
    - `EGO_PAN_Y_VARIABILITY` : Mean absolute rate of change of vertical pan
      (``pan_y_variability``)
    - `EGO_ZOOM_EXCURSION` : Total absolute zoom (``zoom_rate_excursion``)
    - `EGO_ROLL_EXCURSION` : Total absolute roll (``roll_rate_excursion``)
    - `EGO_MOTION` : Most common per-frame motion label (``motion_mode``)

    Persistence Statistics (experimental: the map shape, column names and keywords may change,
    and selecting any of them raises :class:`~dataeval.exceptions.ExperimentalWarning` once
    per process; per-video maps of 48 x 85 cells, measured where the scene moves behind the
    camera; need OpenCV, from the ``dataeval[opencv]`` extra):

    - `PERSISTENCE_EGO` : Share of each cell's pixels that stay unchanged while the scene behind
      them moves, such as a burned-in overlay or a part of the vehicle; this is what is fixed to
      the camera, not ego motion (``ego_map``)
    - `PERSISTENCE_BLUR` : Natural log of how much each cell softens the same content, relative to
      the video's average, with blur wider than `blur_sigma` cells removed (``blur_map``)
    - `PERSISTENCE_BLUR_RAW` : The same, before that removal (``blur_map_raw``)

    Convenience Groups
    ------------------
    - `NONE` : No statistics
    - `DESCRIPTIVE` : All descriptive statistics
    - `REDUNDANCY` : All redundancy statistics
    - `EGO` : All ego-motion statistics
    - `PERSISTENCE` : All persistence statistics (experimental)
    - `NO_EGO` : All statistics that need no OpenCV. The default for
      :func:`~dataeval.core.video_stats`
    - `ALL` : All available statistics, including the experimental persistence statistics

    Notes
    -----
    Every `EGO` statistic comes out of one optical-flow fit per frame pair, and the flow fit
    dominates the cost. Selecting any one `EGO` flag pays for the whole fit; the rest only
    choose which columns come back.

    Selecting any `PERSISTENCE` flag computes ``ego_map``, which the blur maps rely on, and
    selecting either blur flag pays for one solve that gives both blur maps.

    Examples
    --------
    Select specific statistics:

    >>> stats = VideoStats.DESCRIPTIVE_FRAME_COUNT | VideoStats.DESCRIPTIVE_DURATION

    Use convenience groups:

    >>> stats = VideoStats.DESCRIPTIVE
    """

    # ===== NONE (no stats) =====
    NONE = 0

    # ===== DESCRIPTIVE STATS =====
    DESCRIPTIVE_FRAME_COUNT = auto()
    DESCRIPTIVE_DURATION = auto()
    DESCRIPTIVE_FRAME_RATE = auto()

    # ===== REDUNDANCY STATS =====
    REDUNDANCY_FRACTION = auto()
    REDUNDANCY_LONGEST_RUN = auto()

    # ===== EGO STATS =====
    EGO_PAN_SPEED = auto()
    EGO_ZOOM_RATE = auto()
    EGO_ROLL_RATE = auto()
    EGO_SHEAR_RATE = auto()
    EGO_MOVER_FRAC = auto()
    EGO_MOVER_SPEED = auto()
    EGO_PAN_X_VARIABILITY = auto()
    EGO_PAN_Y_VARIABILITY = auto()
    EGO_ZOOM_EXCURSION = auto()
    EGO_ROLL_EXCURSION = auto()
    EGO_MOTION = auto()

    # ===== PERSISTENCE STATS =====
    PERSISTENCE_EGO = auto()
    PERSISTENCE_BLUR = auto()
    PERSISTENCE_BLUR_RAW = auto()

    # ===== GROUPS =====
    DESCRIPTIVE = DESCRIPTIVE_FRAME_COUNT | DESCRIPTIVE_DURATION | DESCRIPTIVE_FRAME_RATE
    REDUNDANCY = REDUNDANCY_FRACTION | REDUNDANCY_LONGEST_RUN
    EGO = (
        EGO_PAN_SPEED
        | EGO_ZOOM_RATE
        | EGO_ROLL_RATE
        | EGO_SHEAR_RATE
        | EGO_MOVER_FRAC
        | EGO_MOVER_SPEED
        | EGO_PAN_X_VARIABILITY
        | EGO_PAN_Y_VARIABILITY
        | EGO_ZOOM_EXCURSION
        | EGO_ROLL_EXCURSION
        | EGO_MOTION
    )

    PERSISTENCE = PERSISTENCE_EGO | PERSISTENCE_BLUR | PERSISTENCE_BLUR_RAW

    NO_EGO = DESCRIPTIVE | REDUNDANCY

    # Ultimate convenience
    ALL = NO_EGO | EGO | PERSISTENCE
