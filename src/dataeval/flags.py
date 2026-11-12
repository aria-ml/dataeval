"""Module for flag enums that control function behavior."""

__all__ = ["FrameStats", "ImageStats", "VideoStats"]

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

    Blur Statistics (experimental, so names and values may change; NaN without a reference, with too few edges,
    with a NaN pixel, or for a box that extends past the image):

    - `VISUAL_BLUR` : Width of the sharpest edges, in pixels at the image's own resolution: the 5th percentile
      of edge widths, after Liang et al. (2010)
    - `VISUAL_BLUR_DIRECTIONALITY` : How different horizontal and vertical edges are: the absolute log2 ratio
      of their median widths

    Known limits: a width is in pixels, so values compare only at one resolution. `image_stats` warns when one call
    requests blur for images of more than one size; `Outliers` calls it once per dataset, so sizes that differ only
    between datasets go unwarned. `VISUAL_BLUR` follows the sharpest edges, so motion blur moves it weakly and
    unevenly and shows in `VISUAL_BLUR_DIRECTIONALITY` instead, which is unsigned and has a scene-dependent baseline.
    Noise and JPEG artifacts move it by amounts and in directions that depend on the scene, so it cannot tell them
    from focus, and a soft 8-bit image at low contrast reads sharper than at full contrast, for a cause not traced, so
    compare only images of similar contrast. It costs far more than `VISUAL_SHARPNESS`, and nothing caps the image
    size, so time a sample first. It was checked only on a few real frames with synthetic degradations and no
    human-rated blur. See :doc:`/concepts/BlurStatistics`.

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
    - `DIMENSION_DISTANCE_EDGE` : Distance from box to nearest image edge. It is unsigned: a box 5 px outside the
      image reads like one 5 px inside, and `SPATIAL_BOX_IN_IMAGE_FRACTION` tells them apart
    - `DIMENSION_INVALID_BOX` : Whether bounding box is invalid
    - `DIMENSION_REL_OFFSET_X` : X-offset of bounding box as a fraction of image width
    - `DIMENSION_REL_OFFSET_Y` : Y-offset of bounding box as a fraction of image height
    - `DIMENSION_REL_WIDTH` : Box width as a fraction of image width
    - `DIMENSION_REL_HEIGHT` : Box height as a fraction of image height
    - `DIMENSION_REL_SIZE` : Box area as a fraction of image area
    - `DIMENSION_REL_DISTANCE_CENTER` : Distance to the image center as a fraction of the half-diagonal
    - `DIMENSION_REL_DISTANCE_EDGE` : Distance to the nearest edge as a fraction of that edge's dimension, unsigned
      like `DIMENSION_DISTANCE_EDGE`

    Hash Statistics (computed on raw image data):

    - `HASH_XXHASH` : xxHash of raw image
    - `HASH_PHASH` : Perceptual hash of image
    - `HASH_DHASH` : Difference/gradient hash of image
    - `HASH_PHASH_D4` : Perceptual hash with D4 symmetry (rotation/flip invariant)
    - `HASH_DHASH_D4` : Difference hash with D4 symmetry (rotation/flip invariant)

    Spatial Statistics (experimental, so names and values may change; one box measured against the other boxes of its
    image and against the image's edges, so they need boxes and are NaN on an image's own row):

    - `SPATIAL_NN_DISTANCE_NORM` : Distance from the box's center to the center of its nearest other box, over the
      box's own diagonal. It is directed, since each box divides by its own diagonal, and it is center to center,
      not the gap between boxes. It reads the boxes as given, not clipped to the image
    - `SPATIAL_BOX_IN_IMAGE_FRACTION` : The share of the box's area that lies inside the image, in (0, 1]. It is
      always 1.0 on datasets whose annotation tools clip boxes to the image

    A box is a neighbor when it has positive area, finite ordered edges and part of it inside the image. A request
    that includes a spatial statistic also returns details, in the result's ``details``, one value per row:

    - `spatial_status` : `ok`; `invalid_box`, when the box itself has zero area, lies entirely outside the image,
      has reversed edges or is not finite (read from the coordinates as given, so a reversed box is `invalid_box`
      here though `DIMENSION_INVALID_BOX` is False for it, unless the box is a `BoundingBox`, whose constructor has
      already swapped its edges); `incomplete_geometry`, when another box of the image is
      reversed or not finite, so a nearer neighbor may exist that could not be placed (the value is still computed
      from the valid boxes, and the largest `SPATIAL_NN_DISTANCE_NORM` values are then the least reliable); or
      `not_applicable`, on an image's own row
    - `nn_distance_norm_target_index` : The nearest neighbor's index among the image's boxes, -1 for none. On equal
      distances the lowest index is nearest
    - `nn_distance_norm_reason` : `no_neighbor` when the status is `ok` and no other valid box exists, else empty

    Convenience Groups
    ------------------
    Sub-groups:

    - `PIXEL_BASIC` : Mean, std, var
    - `PIXEL_DISTRIBUTION` : Skew, kurtosis, entropy, histogram
    - `VISUAL_BASIC` : Brightness, contrast, sharpness
    - `DIMENSION_BASIC` : Width, height, channels, aspect ratio
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
    - `VISUAL` : All visual statistics, except the blur statistics (`BLUR`)
    - `DIMENSION` : All dimension statistics
    - `STANDARD` : The pixel, visual and dimension statistics (`PIXEL`, `VISUAL` and `DIMENSION`), with no hashes and
      no blur. Its columns change only after a deprecation and a migration note. `image_stats` computes it plus `HASH`
      when given no `stats`, and `HASH` can gain members. `Duplicates` needs the hashes, so add `HASH` for it
    - `NO_HASH` : The same set as `STANDARD`, under its v1.2 name
    - `HASH` : All hash statistics
    - `BLUR` : The two blur statistics (experimental)
    - `SPATIAL` : The two spatial statistics (experimental). In no other group. They need boxes, and are NaN on every
      other row
    - `BASIC` : Mean, std, var, brightness, contrast, sharpness, width, height, channels and aspect ratio
      (`PIXEL_BASIC`, `VISUAL_BASIC` and `DIMENSION_BASIC`), with no hashes. Its columns change only after a
      deprecation and a migration note. A subset of what `image_stats` computes when given no `stats`, so cheaper to
      compute, and the set to start from
    - `ALL` : Every statistic in this release, including the experimental blur and spatial statistics. A later
      release can add statistics to it. Name `BASIC`, `STANDARD` or single statistics when the columns must not
      change

    Notes
    -----
    Some statistics have dependencies on others, the dependencies will be added
    automatically during processing.

    ``ImageStats.ALL & ~ImageStats.BLUR & ~ImageStats.SPATIAL`` is what `ALL` held in v1.2. The `SPATIAL` statistics
    need box rows, so a request for `ALL` with no box rows returns their columns as NaN and warns.

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

    # ===== BLUR STATS =====
    # Appended after the hash flags so that no existing flag value moves
    VISUAL_BLUR = auto()
    VISUAL_BLUR_DIRECTIONALITY = auto()

    # ===== SPATIAL STATS =====
    # Appended after the blur flags so that no existing flag value moves
    SPATIAL_NN_DISTANCE_NORM = auto()
    SPATIAL_BOX_IN_IMAGE_FRACTION = auto()

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

    # Kept out of VISUAL and VISUAL_BASIC, so Outliers' default keeps its columns and its speed
    BLUR = VISUAL_BLUR | VISUAL_BLUR_DIRECTIONALITY

    # Compares a box with the other boxes of its image, so it has a value only on a box's own row
    SPATIAL = SPATIAL_NN_DISTANCE_NORM | SPATIAL_BOX_IN_IMAGE_FRACTION

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

    # NO_HASH is its v1.2 name and must come second, so that Python prints STANDARD. A test pins its columns.
    STANDARD = PIXEL | VISUAL | DIMENSION
    NO_HASH = STANDARD

    # The core cheap statistics, from the three sub-groups above
    BASIC = PIXEL_BASIC | VISUAL_BASIC | DIMENSION_BASIC

    # Ultimate convenience, spelled from the families so that a change to STANDARD cannot take a statistic out of it
    ALL = PIXEL | VISUAL | DIMENSION | HASH | BLUR | SPATIAL


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

    Blur Statistics (experimental: the column name and the `blur_samples` keyword may change, and selecting it,
    directly or through `ALL`, raises :class:`~dataeval.exceptions.ExperimentalWarning` once per process; one value
    per video, the median over a sample of its frames of the measurement behind `ImageStats.VISUAL_BLUR`; needs
    OpenCV, from the ``dataeval[opencv]`` extra):

    - `BLUR_LEVEL` : Median over the sampled frames of the width of the sharpest edges, in pixels (``blur_level``)

    At most `blur_samples` frames per video count toward the median, spread evenly over its length. Frames are shrunk
    to a short side of 720 pixels and never enlarged, so compare only videos of one frame size, and only frames of the
    size of a video's first frame are measured. See :func:`~dataeval.core.video_stats` and
    :doc:`/concepts/BlurStatistics`.

    Convenience Groups
    ------------------
    - `NONE` : No statistics
    - `DESCRIPTIVE` : All descriptive statistics
    - `REDUNDANCY` : All redundancy statistics
    - `EGO` : All ego-motion statistics
    - `PERSISTENCE` : All persistence statistics (experimental)
    - `BLUR` : The blur statistic (experimental)
    - `BASIC` : The descriptive and redundancy statistics, which need no OpenCV. The default for
      :func:`~dataeval.core.video_stats`. Its columns change only after a deprecation and a migration note. The set to
      start from
    - `NO_EGO` : The same set as `BASIC`, under its v1.2 name
    - `ALL` : Every statistic in this release, including the experimental persistence and blur statistics. A later
      release can add statistics to it. Name `BASIC` or single statistics when the columns must not change

    Notes
    -----
    Every `EGO` statistic comes out of one optical-flow fit per frame pair, and the flow fit
    dominates the cost. Selecting any one `EGO` flag pays for the whole fit; the rest only
    choose which columns come back.

    Selecting any `PERSISTENCE` flag computes ``ego_map``, which the blur maps rely on, and
    selecting either persistence blur flag pays for one solve that gives both blur maps.

    Selecting `BLUR_LEVEL` measures a sample of each video's frames, and a long video can measure more
    frames than count toward its median. The cost is seconds per video on the clips timed;
    :doc:`/concepts/BlurStatistics` gives the measurements.

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

    # ===== BLUR STATS =====
    BLUR_LEVEL = auto()

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

    # The group name of the blur statistics, which can grow; today it is the one flag.
    BLUR = BLUR_LEVEL

    # The core cheap statistics. NO_EGO is its v1.2 name and must come second, so that Python prints BASIC.
    BASIC = DESCRIPTIVE | REDUNDANCY
    NO_EGO = BASIC

    # Ultimate convenience: grows between releases, so name BASIC or single flags for columns that must not change
    ALL = BASIC | EGO | PERSISTENCE | BLUR


class FrameStats(Flag):
    """
    Flag enumeration for controlling per-frame statistics computation.

    Each statistic describes one frame against the frame before it, so a result has one row per frame of
    every video, and the first frame of each video has no value. Flags can be combined using bitwise OR (|)
    to select multiple statistics. :func:`~dataeval.core.frame_stats` is experimental.

    Individual Stats
    ----------------
    - `TIMING` : Seconds since the previous frame (``elapsed_time_s``), read from the frame timestamps and
      never from pixels
    - `EXACT_REPEAT` : Whether the frame is identical to the previous one (``repeat_frame``), which decodes
      every frame

    Convenience Groups
    ------------------
    - `NONE` : No statistics
    - `ALL` : Every statistic in this release. A later release can add statistics to it, and it decodes
      every frame because it includes `EXACT_REPEAT`

    Examples
    --------
    Measure timing only, which decodes no pixels:

    >>> stats = FrameStats.TIMING
    """

    NONE = 0
    TIMING = auto()
    EXACT_REPEAT = auto()

    ALL = TIMING | EXACT_REPEAT
