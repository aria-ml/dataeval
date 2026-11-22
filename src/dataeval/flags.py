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
    The context, color and saturation statistics are the exception: they read a declared interval, else 8-bit data
    against (0, 255) or floating-point data within [0, 1] or [0, 255], and are NaN otherwise.
    See :doc:`/concepts/ImageStatistics`.

    A statistic's column is its flag's name after the first underscore, in lower case: `PIXEL_MEAN` gives `mean` and
    `VISUAL_BLUR_EDGE_WIDTH` gives `blur_edge_width`. A channel group's name goes in front of a column and a view's name
    (`background_`, or `surround_` with `per_surround`) in front of that (`background_nir_mean`,
    `surround_color_lightness_mean`), except that the context statistics' own surround columns put the surround first
    (`surround_nir_intensity_mean`). Two parts of a call that would give one name raise.

    The statistics added in 1.3 report a status for each row in their details (`blur_status`, `spatial_status`,
    `saturated_status`, `context_status`, `color_status`, `noise_status`). A value is NaN unless its status is `ok`,
    and every status comes from one list, the first that applies:

    - `not_applicable` : A row the statistic does not define, such as an image's own row for a box statistic
    - `invalid_box` : The box, as converted, lies outside the image, has zero area or is not finite, which is what
      `DIMENSION_INVALID_BOX` reports
    - `reversed_box` : The box has an edge before the one it should follow, and with its edges swapped it is valid,
      so `DIMENSION_INVALID_BOX` is False for it. Only the spatial and context statistics, which read a box's geometry
      as given, report it; the pixel and color statistics measure the swapped box
    - `missing_channel` : The view names a channel the datum does not have
    - `unknown_range` : No interval to read the values against could be established
    - `empty_surround` : The box fills its window, so it has no surround (context only)
    - `insufficient_support` : Too few pixels or values to measure, as each statistic defines it
    - `ok` : Measured

    `track_steps` uses the same `invalid_box` and `reversed_box`, except that it measures a box outside the frame,
    since a step between two frames does not depend on the frame's edge.

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
    - `PIXEL_SATURATED_FRACTION` : Share of values at a clipping endpoint; its endpoint, details and limits are under
      Saturation Statistic below

    Visual Statistics (read against the 0-255 display range; NaN without a reference):

    - `VISUAL_BRIGHTNESS` : Lower-quartile display value (depends on `VISUAL_PERCENTILES`)
    - `VISUAL_CONTRAST` : Spread of the quartiles relative to their mean, so already
      scale-free (depends on `VISUAL_PERCENTILES`)
    - `VISUAL_DARKNESS` : Upper-quartile display value (depends on `VISUAL_PERCENTILES`)
    - `VISUAL_SHARPNESS` : Spread of edge magnitudes, via a 3x3 edge kernel
    - `VISUAL_PERCENTILES` : Display-range percentiles (0, 25, 50, 75, 100)

    Blur Statistics (experimental, so names and values may change; NaN without a reference, with too few edges,
    with a NaN pixel (one under a `mask` of :func:`~dataeval.core.image_stats` is left out), or for a box that extends
    past the image):

    - `VISUAL_BLUR_EDGE_WIDTH` : Width of the sharpest edges, in pixels at the image's own resolution: the 5th
      percentile of edge widths, each the spread of an edge's gradient profile, after Liang et al. (2010). With
      `per_background=True` it also gives `background_blur_edge_width`, taken over the edges outside the boxes and a
      9 px margin around them. Under a `mask`, an edge point must reach the mean gradient taken outside the mask grown
      by 9 px, the edges within 9 px of the mask are left out of the 5th percentile, and the width of each edge stops
      at the mask, so no masked pixel sets which edges count or how wide they are
    - `VISUAL_BLUR_EDGE_DIRECTIONALITY` : How different horizontal and vertical edges are: the absolute log2 ratio
      of their median widths. NaN under a `mask`

    A request that includes a blur statistic also returns details: `blur_status`, `ok`, `not_applicable`,
    `invalid_box`, `missing_channel`, `unknown_range` or `insufficient_support` (which a box that extends past the
    image reads), and, with `VISUAL_BLUR_EDGE_DIRECTIONALITY`, `blur_edge_directionality_reason`,
    `insufficient_axis_support` where directionality alone is NaN, `masked` on every row that measures directionality
    under a `mask`, else empty.

    Known limits: a width is in pixels, so values compare only at one resolution. `image_stats` warns when one call
    requests blur for images of more than one size; `Outliers` calls it once per dataset, so sizes that differ only
    between datasets go unwarned. `VISUAL_BLUR_EDGE_WIDTH` follows the sharpest edges, so motion blur moves it weakly
    and unevenly and shows in `VISUAL_BLUR_EDGE_DIRECTIONALITY` instead, which is unsigned and has a scene-dependent
    baseline. Noise and JPEG artifacts move it by amounts and in directions that depend on the scene, so it cannot tell
    them from focus, and a soft 8-bit image at low contrast reads sharper than at full contrast, for a cause not traced,
    so compare only images of similar contrast. It costs far more than `VISUAL_SHARPNESS`, and nothing caps the image
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

    Spatial Statistics (one box measured against the other boxes of its image and against the image's edges, so they
    need boxes and are NaN on an image's own row):

    - `SPATIAL_NN_DISTANCE_NORM` : Distance from the box's center to the center of its nearest other box, over the
      box's own diagonal. It is directed, since each box divides by its own diagonal, and it is center to center,
      not the gap between boxes. It reads the boxes as given, not clipped to the image
    - `SPATIAL_BOX_IN_IMAGE_FRACTION` : The share of the box's area that lies inside the image, in (0, 1]. It is
      always 1.0 on datasets whose annotation tools clip boxes to the image. It reads the boxes as given
    - `SPATIAL_COVERED_FRACTION` : The share of the box's area that the other boxes of the image cover, in [0, 1],
      counting boxes that overlap each other once: ``area(box ∩ union of the other boxes) / area(box)``, over boxes
      of every class. It measures geometric coverage by annotated boxes, not the probability that the object is
      occluded, and an object nobody annotated covers nothing. 0.0 for a box that nothing overlaps. It reads the boxes
      clipped to the image, and is exact up to floating-point rounding. It is NaN, with the reason `too_many_overlaps`,
      for a box that more than 1,000 other boxes overlap, since the union takes memory that grows with the square of
      that number. That limit is provisional, and the other statistics are not affected by it
    - `SPATIAL_MAX_IOU_SAME_CLASS` : The largest intersection over union of the box with another box of its class, 0.0
      when none overlaps it. It reads the boxes clipped to the image
    - `SPATIAL_MAX_IOU_OTHER_CLASS` : The largest intersection over union of the box with a box of another class, 0.0
      when none overlaps it. A large value is one box lying on another of a different class. That is either two
      objects, such as a rider on a bicycle, or two annotations of one object, which can be a conflicting label. It
      reads the boxes clipped to the image

    The two class statistics read class labels, which a dataset supplies and `image_stats` takes through its `labels`
    keyword. Any negative id is a missing label, and every label is missing where none are given. The class of a box
    that overlaps this one is then unknown, and so is whether it is a rival: where such a box overlaps more than the
    largest known overlap, the statistic is NaN with the reason `missing_labels`, since that box may be the true
    maximum. The statistic does not guess.

    A box is another box's peer when it has positive area, finite ordered edges and part of it inside the image; the
    nearest peer is the neighbor. A request that includes a spatial statistic also returns details, in the result's
    ``details``, one value per row:

    - `spatial_status` : `ok`, `not_applicable`, `invalid_box` or `reversed_box` (see the statuses above)
    - `spatial_incomplete_geometry` : True where another box of the image is reversed or not finite, so a box that
      could not be placed may be nearer to this one or may overlap it. The value is still computed from the valid
      boxes, and the largest `SPATIAL_NN_DISTANCE_NORM` values may then be too large, and the overlap statistics too
      small. False wherever the status is not `ok`
    - `nn_distance_norm_target_index` : The nearest neighbor's index among the image's boxes, -1 for none. On equal
      distances the lowest index is nearest
    - `nn_distance_norm_reason` : `no_neighbor` when the status is `ok`, `spatial_incomplete_geometry` is False and no
      other valid box exists, else empty
    - `spatial_peer_count`, `spatial_skipped_count` : With the covered fraction or either class statistic, the number
      of valid peers the box was compared with and the number of boxes left out for having zero area or lying
      entirely outside the image. A box that is reversed or not finite is in neither count; it sets
      `spatial_incomplete_geometry` on the other boxes.
      -1 where the box is not measured
    - `max_iou_same_class_target_index`, `max_iou_other_class_target_index` : The peer that attains the maximum, by its
      index among the image's boxes, -1 where there is none (the maximum is 0.0 or NaN). On equal values the lowest
      index
    - `max_iou_other_class_label` : That peer's class id, -1 for none
    - `max_iou_same_class_reason`, `max_iou_other_class_reason` : `missing_labels` when the value is NaN for want of
      labels, else empty
    - `covered_fraction_reason` : `too_many_overlaps` when the value is NaN for that reason, else empty

    Saturation Statistic (the share of values at a clipping endpoint, in `PIXEL`):

    - `PIXEL_SATURATED_FRACTION` : The share of finite raw values that sit at the endpoint, within `saturated_atol`. It
      counts channel values, not spatial pixels, so an image clipped in one channel of three reads 1/3. The endpoint is
      never read from the data's maximum. It is the upper bound of the interval the context and color statistics
      read: a declared `value_range`, as :func:`~dataeval.core.image_stats` resolves one for each image, else 255 for
      ``uint8`` data, else 1 or 255 for floating-point data within [0, 1] or [0, 255], else it is unknown and the value
      is NaN. Data of 12 bits stored in uint16 therefore reads NaN until its range is declared as ``(0, 4095)``. Finite
      values above the endpoint stay in the denominator and are not counted

    A request that includes it also returns details, in the result's ``details``, one value per row, named with a band
    group's name in front when the row is a band group's:

    - `saturated_status` : `ok`, `not_applicable`, `invalid_box`, `missing_channel`, `unknown_range` (no endpoint
      resolved) or `insufficient_support` (no position of the datum in the view, as for a background the boxes cover,
      or no finite value)
    - `saturated_above_range_fraction` : The finite values above the endpoint plus `saturated_atol`, over the finite
      values. A share above 0 says the declared range is narrower than the data
    - `saturated_upper` : The endpoint used, NaN when none resolved

    The result's ``measurement_specs`` records `saturated_atol` and a range that the call or a channel group declared,
    so `combine_stats_results` refuses results measured with different ones. A range an image's own metadata declares
    and the endpoint the dtype or a float convention supplies are in the details, which are not compared, so a
    collection of ``uint8`` images and a collection of float ones still combine. `QualityGate` does not accept it,
    since a gate measures frames with no datum metadata to declare a range in.

    Context Statistics (the intensity of the pixels around a box, in `CONTEXT`):

    - `CONTEXT_SURROUND_INTENSITY_MEAN` : The mean of the normalized intensity of the box's surround. The surround is
      the box widened on every side by `surround_padding` times its width and its height, a quarter by default, less the
      box and less every other valid box of the image, each rounded outward to whole pixels and clipped to the image. A
      pixel's intensity is its value mapped to [0, 1] from the interval the view is measured against (see `value_range`
      of :func:`~dataeval.core.image_stats`), averaged over the channels read, so the box and its surround share one
      scale. A pixel counts only when every channel read is finite and inside the interval, and a value outside is never
      clipped. An object that no box annotates stays in the surround. Without a declared interval, unsigned 8-bit data
      reads against (0, 255) and floating-point data within [0, 1] or [0, 255] against that range, and any other data
      is NaN with the status `unknown_range`
    - `CONTEXT_SURROUND_INTENSITY_STD` : The population standard deviation (``ddof=0``) of the same intensities
    - `CONTEXT_TARGET_SURROUND_CONTRAST` : How far the box stands from its surround relative to the surround's level,
      ``(mu_t - mu_s) / max(mu_s, floor)``. `mu_s` is `surround_intensity_mean`, and `mu_t` is the mean intensity of the
      box's own pixels, the box rounded outward and clipped as for the surround, over the pixels at which every channel
      read is finite and inside the interval. It covers the whole box, so it includes the pixels of overlapping
      neighbors and of background inside a loose box. The sign is kept: a box darker than its surround reads below 0.
      `floor` is `context_ratio_floor` of :func:`~dataeval.core.image_stats`, 1/255 by default
    - `CONTEXT_SIGNAL_CLUTTER_RATIO` : The same difference relative to the surround's spread,
      ``(mu_t - mu_s) / max(sd_s, floor)`` with `sd_s` the `surround_intensity_std`

    The two ratios are Weber-style contrasts of the encoded intensity in [0, 1], and not contrasts of physical
    luminance, which gamma and the sensor's response separate from the values. They are not `VISUAL_CONTRAST`, the
    range of an image's percentiles over their mean.

    The context statistics are NaN on an image's own row, and for a box whose surround holds fewer than
    `surround_min_pixels` such pixels, 32 by default, which a small box can fall under and a box whose window other
    boxes cover always does. A ratio is NaN wherever the surround is. Those settings and the floor are keywords of
    :func:`~dataeval.core.image_stats`. The statistics are stable. A request that includes one also returns details, in
    the result's ``details``, one value per row, named for a band group as the columns are (see the naming rule above).
    The surround before other boxes are removed, less the box, is the candidate, and the candidate's pixels that no
    other box covers are the retained ones. A `mask` of :func:`~dataeval.core.image_stats` makes the pixels under it
    unreadable, as a pixel that is not finite or not inside the interval is, so it lowers the measurable counts and the
    valid fractions below and leaves the candidate and the retained pixels as the boxes alone set them:

    - `context_status` : `ok`, `not_applicable`, `invalid_box`, `reversed_box`, `missing_channel`, `unknown_range`,
      `empty_surround` or `insufficient_support` (fewer than `surround_min_pixels` retained pixels measurable, which
      includes a surround that other boxes cover, or that a `mask` covers)
    - `context_incomplete_geometry` : True where another box of the image is reversed or not finite, so that pixels
      of it may lie in the surround; the value is still computed from the valid boxes. False wherever the status is
      not `ok`
    - `surround_candidate_pixels`, `surround_retained_pixels`, `surround_measurable_pixels` : The candidate pixels,
      the retained ones, and the retained ones measurable in every channel read, that is, finite, inside the interval
      and outside any `mask`. -1 where a count does not apply. The first two are reported for the unnamed view only,
      since a band group does not change them
    - `surround_peer_excluded_fraction` : The share of the candidate that other boxes cover, for the unnamed view only,
      NaN where the candidate is empty or not counted
    - `surround_valid_fraction`, `target_valid_fraction` : The measurable pixels over the retained ones, and the share
      of the box's own pixels that are finite, inside the interval in every channel read and outside any `mask`
    - `context_range_low`, `context_range_high` : The interval the row was measured against, NaN where none was
      established
    - `context_ratio_reason`, `target_surround_contrast_floored`, `signal_clutter_ratio_floored` : Present when a ratio
      is requested, each floored value only with its own ratio. `context_ratio_reason` is `insufficient_target_support`
      when the surround was measured and fewer than 90% of the box's pixels are valid, a box under a `mask` included,
      which leaves both ratios NaN with the surround kept, and is empty otherwise, including where a ratio is NaN
      because the surround is not measured, which `context_status` says. A floored value is True where that ratio's
      denominator was below the floor, so the ratio is the difference over the floor, no larger than ``1 / floor`` in
      size, and reads the floor and not the surround. False where the ratio is NaN

    Color Statistics (experimental, so names and values may change; measured only from channels the caller declares
    are sRGB red, green and blue with `color_rgb_channels` of :func:`~dataeval.core.image_stats`, and dropped from a
    request, with a notice, where they are not declared):

    - `VISUAL_COLOR_LIGHTNESS_MEAN` : The mean CIELAB lightness, ``L*``, of the valid pixels. A pixel is valid when its
      three declared channels are finite and inside the interval the view is measured against, and a value outside is
      never clipped. Each channel is mapped to [0, 1] from that interval, decoded from the sRGB transfer function,
      converted to XYZ with the D65 matrix and to CIELAB with the D65 white and the 2-degree observer. The interval is
      the declared one (the call's `value_range` or the datum's own metadata; a channel group's range does not apply),
      and without one the context statistics' rule applies: unsigned 8-bit data against ``(0, 255)``, floating-point
      data within [0, 1] or [0, 255] against that range, and any other data NaN
    - `VISUAL_COLOR_CHROMA_MEAN` : The mean chroma, ``C = hypot(a*, b*)``, of the valid pixels
    - `VISUAL_COLOR_ACHROMATIC_FRACTION` : The share of valid pixels whose chroma is under `color_chroma_threshold`
    - `VISUAL_COLOR_HUE_COS_MEAN`, `VISUAL_COLOR_HUE_SIN_MEAN` : The coordinates of the chroma-weighted mean hue
      direction of the chromatic pixels, those at or over the threshold, ``sum(a*) / sum(C)`` and ``sum(b*) / sum(C)``.
      Each is in [-1, 1] and their squares sum to at most 1, with equality when every chromatic pixel has one hue, so
      ``np.hypot(cos, sin)`` says how concentrated the hues are and ``np.arctan2(sin, cos)`` is the mean angle where
      that is not near 0. NaN, with the reason `achromatic`, where no pixel is chromatic

    The channels are declared and not detected, so a value is only as true as the declaration, and a view is measured
    only if it holds at least `color_min_pixels` valid pixels and 90% of its pixels are valid. They are defined for the
    image, each box, the background and, with `per_surround`, each box's surround, and not for a channel group. The
    surround is the box widened by `surround_padding` of its width and height on every side, rounded outward and
    clipped to the image, with every box of the image masked out, so its columns are named `surround_color_*`, and a
    box whose surround the other boxes cover reads `insufficient_support`. A request that includes them also returns
    details, in the result's ``details``, one value per row:

    - `color_status` : `ok`, `not_applicable`, `invalid_box`, `missing_channel` (the datum has fewer than two
      dimensions or lacks a declared channel), `unknown_range` or `insufficient_support`
    - `color_valid_fraction` : The valid pixels over the pixels of the view that lie inside the datum and that the
      view's mask does not remove, so a box that leaves the image is judged on the part inside it
    - `color_valid_pixels`, `color_excluded_pixels` : The valid pixels, and the pixels the view's mask removes, which
      on an image or box row are the pixels of a `mask` of :func:`~dataeval.core.image_stats`. -1 where the view is not
      measured, and the exclusion is still reported
    - `color_hue_reason` : `achromatic` where hue is NaN for want of a chromatic pixel, else empty

    The result's ``measurement_specs`` records the declared channels, the settings and the range that the call
    declared, and for the surround its region and `surround_padding`, so `combine_stats_results` refuses results
    measured with different ones. `QualityGate` does not accept
    them, and `Outliers` and `Duplicates` cannot take the keyword, so compute with `image_stats` and pass the result
    to their ``from_stats``. See :doc:`/concepts/ImageStatistics` for what they do not establish.

    Noise Statistics (estimated from the pixels alone, in `NOISE`):

    - `PIXEL_NOISE_STD` : The estimated standard deviation of additive white Gaussian noise, the square root of
      `PIXEL_NOISE_VAR`
    - `PIXEL_NOISE_VAR` : The mean over the view's channels of each channel's estimated noise variance. Each channel is
      read with Immerkaer's estimator, :func:`~dataeval.core.noise_immerkaer`, over the 3x3 neighborhoods that hold no
      NaN and no masked pixel, so a masked or missing pixel creates no edge

    A channel needs at least 64 valid neighborhoods, which a square region first has at 10x10, and that floor is fixed.
    The values are in the data's own units. With `normalize_pixel_values` they are divided by the full scale: a
    declared range, else ``(0, 255)`` for unsigned 8-bit data, else ``(0, 1)`` or ``(0, 255)`` for floating-point data
    within it, and any other data is NaN with the status `unknown_range`. The float conventions read each image's own
    values, so declare `value_range` for floating-point data: unclipped noise on 0-1 data moves an image to
    ``(0, 255)``. They are defined for the image, each box, the
    background, each channel group and, with `per_surround`, each box's surround, so a call with
    ``channels={"r": 0, "g": 1, "b": 2}`` adds a reading of each channel beside the joint one. What a reading can and
    cannot tell is in the Notes of :func:`~dataeval.core.noise_immerkaer`. A request that includes them also returns
    details, in the result's ``details``, one value per row, named for a band group as the columns are:

    - `noise_status` : `ok`, `not_applicable`, `invalid_box`, `missing_channel`, `unknown_range` or
      `insufficient_support` (a channel with fewer than 64 valid neighborhoods, as in a square view under 10x10, a datum
      with no image plane or a background the boxes cover)
    - `noise_neighborhoods` : The fewest valid neighborhoods of any channel of the view, 0 where it has none, and -1
      where the view is not read

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
    - `PIXEL` : All pixel statistics, including `PIXEL_SATURATED_FRACTION`. The noise statistics are pixel statistics
      too, in their own opt-in group `NOISE`, outside `PIXEL`
    - `VISUAL` : The core visual statistics. The blur and color statistics are visual statistics too, in their own
      opt-in groups `BLUR` and `COLOR`, outside `VISUAL`
    - `DIMENSION` : All dimension statistics
    - `STANDARD` : The pixel, visual and dimension statistics (`PIXEL`, `VISUAL` and `DIMENSION`), with no hashes and
      no blur. Its columns change only after a deprecation and a migration note. `image_stats` computes it plus `HASH`
      when given no `stats`, and `HASH` can gain members. `Duplicates` needs the hashes, so add `HASH` for it
    - `NO_HASH` : The same set as `STANDARD`, under the name v1.2 gave it, so v1.3 added `PIXEL_SATURATED_FRACTION` to
      it too. Deprecated, and removed in 2.0: name `STANDARD`
    - `HASH` : All hash statistics
    - `BLUR` : The two blur statistics (experimental)
    - `SPATIAL` : The five spatial statistics. They need boxes, and are NaN on every other row. The
      two class statistics also read labels
    - `CONTEXT` : The four context statistics, the surround mean and standard deviation and the two ratios. They need
      boxes, and are NaN on every other row
    - `COLOR` : The five experimental color statistics. They are measured only where
      `color_rgb_channels` is given, and are dropped from a request, with a notice, where it is not
    - `NOISE` : The two noise statistics, `PIXEL_NOISE_STD` and `PIXEL_NOISE_VAR`
    - `BASIC` : Mean, std, var, brightness, contrast, sharpness, width, height, channels and aspect ratio
      (`PIXEL_BASIC`, `VISUAL_BASIC` and `DIMENSION_BASIC`), with no hashes. Its columns change only after a
      deprecation and a migration note. A subset of what `image_stats` computes when given no `stats`, so cheaper to
      compute, and the set to start from
    - `ALL` : Every statistic in this release, including the experimental blur and color statistics, the spatial
      statistics, the context statistics and the noise statistics. A later release can add statistics to it. The
      color statistics are measured only where `color_rgb_channels` is given. Name `BASIC`, `STANDARD` or single
      statistics when the columns must not change

    Notes
    -----
    Some statistics have dependencies on others, the dependencies will be added
    automatically during processing.

    .. versionadded:: 1.3
        The `BLUR`, `SPATIAL`, `CONTEXT`, `COLOR` and `NOISE` statistics, `PIXEL_SATURATED_FRACTION`, and the
        `STANDARD` and `BASIC` groups.

    .. deprecated:: 1.3
        `NO_HASH`, the v1.2 name of `STANDARD`. It is removed in 2.0.

    What `ALL` held in v1.2 is
    ``ImageStats.ALL & ~ImageStats.BLUR & ~ImageStats.SPATIAL & ~ImageStats.PIXEL_SATURATED_FRACTION &
    ~ImageStats.CONTEXT & ~ImageStats.COLOR & ~ImageStats.NOISE``.
    The `SPATIAL` and `CONTEXT` statistics need box rows, so a request for `ALL` with no box rows returns their columns
    as NaN, without a warning, as a statistic reads NaN on any row it cannot measure. Any other request that holds
    them and measures no box rows warns. A request that holds nothing else raises when the call has no boxes at all
    (no `boxes` and none in the dataset, or `per_target=False`).

    Examples
    --------
    Select specific statistics:

    >>> stats = ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD
    >>> stats = ImageStats.VISUAL_BRIGHTNESS | ImageStats.DIMENSION_WIDTH

    Use convenience groups:

    >>> stats = ImageStats.PIXEL  # All pixel stats
    >>> stats = ImageStats.PIXEL_BASIC | ImageStats.VISUAL  # Basic pixel & all visual
    """

    # Naming rules, read by `_stat_names` and by dataeval-flow's `stat_names_for`:
    # - One prefix per family. A member's column is its name after the first underscore, lowercased, so the name must
    #   parse to a column that says what it measures and collides with no other (a new noise family cannot be
    #   `NOISE_STD`, which parses to `std`).
    # - `BLUR` and `COLOR` are opt-in groups of visual statistics outside `VISUAL`, and `NOISE` one of pixel statistics
    #   outside `PIXEL`; their members carry `VISUAL_` or `PIXEL_` so that the column keeps the family word
    #   (`blur_edge_width`, `color_lightness_mean`, `noise_std`).
    # - A detail's stem is its family, its sub-category or the shared stem of the statistics it covers; `<stem>_status`
    #   is a row's state and `<stem>_reason` one statistic's own NaN cause. A keyword's prefix names what it configures.
    # - Members are appended, never inserted, so that no flag value moves.

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
    VISUAL_BLUR_EDGE_WIDTH = auto()
    VISUAL_BLUR_EDGE_DIRECTIONALITY = auto()

    # ===== SPATIAL STATS =====
    # Appended after the blur flags so that no existing flag value moves
    SPATIAL_NN_DISTANCE_NORM = auto()
    SPATIAL_BOX_IN_IMAGE_FRACTION = auto()
    SPATIAL_COVERED_FRACTION = auto()
    SPATIAL_MAX_IOU_SAME_CLASS = auto()
    SPATIAL_MAX_IOU_OTHER_CLASS = auto()

    # ===== SATURATION =====
    # Appended after the spatial flags so that no existing flag value moves.
    PIXEL_SATURATED_FRACTION = auto()

    # ===== CONTEXT =====
    # Appended after the saturation flag so that no existing flag value moves.
    CONTEXT_SURROUND_INTENSITY_MEAN = auto()
    CONTEXT_SURROUND_INTENSITY_STD = auto()
    # The box against its surround. Appended after the surround flags so that no flag value moves.
    CONTEXT_TARGET_SURROUND_CONTRAST = auto()
    CONTEXT_SIGNAL_CLUTTER_RATIO = auto()

    # ===== COLOR =====
    # Appended after the context flags so that no flag value moves.
    VISUAL_COLOR_LIGHTNESS_MEAN = auto()
    VISUAL_COLOR_CHROMA_MEAN = auto()
    VISUAL_COLOR_ACHROMATIC_FRACTION = auto()
    VISUAL_COLOR_HUE_COS_MEAN = auto()
    VISUAL_COLOR_HUE_SIN_MEAN = auto()

    # ===== NOISE =====
    # Appended after the color flags so that no flag value moves.
    PIXEL_NOISE_STD = auto()
    PIXEL_NOISE_VAR = auto()

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
        | PIXEL_SATURATED_FRACTION
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
    BLUR = VISUAL_BLUR_EDGE_WIDTH | VISUAL_BLUR_EDGE_DIRECTIONALITY

    # Compares a box with the other boxes of its image, so it has a value only on a box's own row
    SPATIAL = (
        SPATIAL_NN_DISTANCE_NORM
        | SPATIAL_BOX_IN_IMAGE_FRACTION
        | SPATIAL_COVERED_FRACTION
        | SPATIAL_MAX_IOU_SAME_CLASS
        | SPATIAL_MAX_IOU_OTHER_CLASS
    )

    # Reads the pixels around a box, so it has a value only on a box's own row
    CONTEXT = (
        CONTEXT_SURROUND_INTENSITY_MEAN
        | CONTEXT_SURROUND_INTENSITY_STD
        | CONTEXT_TARGET_SURROUND_CONTRAST
        | CONTEXT_SIGNAL_CLUTTER_RATIO
    )

    # Reads three declared channels as sRGB, so it has a value only where `color_rgb_channels` is given
    COLOR = (
        VISUAL_COLOR_LIGHTNESS_MEAN
        | VISUAL_COLOR_CHROMA_MEAN
        | VISUAL_COLOR_ACHROMATIC_FRACTION
        | VISUAL_COLOR_HUE_COS_MEAN
        | VISUAL_COLOR_HUE_SIN_MEAN
    )

    # Kept out of PIXEL, so the default request and Outliers' default keep their columns and their speed
    NOISE = PIXEL_NOISE_STD | PIXEL_NOISE_VAR

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
    ALL = PIXEL | VISUAL | DIMENSION | HASH | BLUR | SPATIAL | CONTEXT | COLOR | NOISE


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
    per video, the median over a sample of its frames of the measurement behind
    `ImageStats.VISUAL_BLUR_EDGE_WIDTH`; needs OpenCV, from the ``dataeval[opencv]`` extra):

    - `BLUR_EDGE_WIDTH` : Median over the sampled frames of the width of the sharpest edges, in pixels
      (``blur_edge_width_median``)

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
    - `NO_EGO` : The same set as `BASIC`, under its v1.2 name. Deprecated, and removed in 2.0: name `BASIC`
    - `ALL` : Every statistic in this release, including the experimental persistence and blur statistics. A later
      release can add statistics to it. Name `BASIC` or single statistics when the columns must not change

    Notes
    -----
    Every `EGO` statistic comes out of one optical-flow fit per frame pair, and the flow fit
    dominates the cost. Selecting any one `EGO` flag pays for the whole fit; the rest only
    choose which columns come back.

    Selecting any `PERSISTENCE` flag computes ``ego_map``, which the blur maps rely on, and
    selecting either persistence blur flag pays for one solve that gives both blur maps.

    Selecting `BLUR_EDGE_WIDTH` measures a sample of each video's frames, and a long video can measure more
    frames than count toward its median. The cost is seconds per video on the clips timed;
    :doc:`/concepts/BlurStatistics` gives the measurements.

    .. versionadded:: 1.3
        The `BLUR` statistic and the `BASIC` group.

    .. deprecated:: 1.3
        `NO_EGO`, the v1.2 name of `BASIC`. It is removed in 2.0.

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
    BLUR_EDGE_WIDTH = auto()

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
    BLUR = BLUR_EDGE_WIDTH

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
    - `TIMING_ELAPSED` : Seconds since the previous frame (``elapsed_time_s``), read from the frame timestamps
      and never from pixels
    - `REPEAT_EXACT` : Whether the frame is identical to the previous one (``repeat_frame``), which decodes
      every frame

    Convenience Groups
    ------------------
    - `NONE` : No statistics
    - `TIMING` : The timing statistics, today `TIMING_ELAPSED`
    - `REPEAT` : The repeat statistics, today `REPEAT_EXACT`
    - `ALL` : Every statistic in this release. A later release can add statistics to it, and it decodes
      every frame because it includes `REPEAT_EXACT`

    Examples
    --------
    Measure timing only, which decodes no pixels:

    >>> stats = FrameStats.TIMING
    """

    NONE = 0
    TIMING_ELAPSED = auto()
    REPEAT_EXACT = auto()

    # One member each today. The group names come second, so that Python prints the statistic's name.
    TIMING = TIMING_ELAPSED
    REPEAT = REPEAT_EXACT

    ALL = TIMING_ELAPSED | REPEAT_EXACT
