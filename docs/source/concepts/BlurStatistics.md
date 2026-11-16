<!-- markdownlint-disable MD051 -->

# Blur statistics

The blur statistics measure how wide the edges in an image are, in pixels. They are two columns of {func}`.image_stats`,
`blur_edge_width` and `blur_edge_directionality`, plus `background_blur_edge_width` when you ask for the scene outside
the boxes, and one column of {func}`.video_stats`, `blur_edge_width_median`, a median of the image measurement over a
sample of the video's frames, so the method and the limits of the image measurement apply to both. You request them with
the `ImageStats.BLUR` and `VideoStats.BLUR` groups. This page is a spinoff of [Image statistics](ImageStatistics.md),
which describes the pixel and visual statistics and the full-scale reference that the visual statistics rely on. Read
that page first if those terms are new.

The blur statistics are experimental ⚠️. The first image request and the first video request in a process each raise
{class}`~dataeval.exceptions.ExperimentalWarning`, and a request for `ImageStats.ALL` or `VideoStats.ALL` raises it too.
The column names and values may change, and for videos so may the `blur_samples` keyword.

## Why measure the width of edges

`sharpness` measures how strong the edges in a picture are. It is the spread of a 3 × 3 edge filter, so it moves with
the contrast of the scene and with noise, and it cannot say how soft a lens is. The blur statistics measure the **edge
width**, the number of pixels an edge takes to go from dark to light.

Halving the contrast of three 8-bit frames with no added blur took `sharpness` from 57.5 to 32.8, from 40.7 to 22.5 and
from 40.6 to 21.8, and left `blur_edge_width` unchanged to two decimals. A soft 8-bit image at low contrast is the
exception, as [Soft 8-bit images at low contrast](#soft-8-bit-images-at-low-contrast) describes.

The persistence blur maps of {func}`.video_stats` are relative to a video's own average, so a lens that is soft
everywhere alike reads 0 in them. `blur_edge_width_median` gives an absolute whole-frame edge width in pixels. It is a
different measure from the maps' and does not combine with them.

## How the edge width is measured

The method follows Liang et al. ([2010](#ref1)). It finds the points where the smoothed gradient peaks along an axis,
measures how far the gradient magnitude spreads around each one, and corrects for the direction of the gradient.
DataEval summarizes the edge widths with percentiles, which is a change from the paper's own score, in two columns for
an image and one for a video:

| Column | Flag | What it is |
| --- | --- | --- |
| `blur_edge_width` | `VISUAL_BLUR_EDGE_WIDTH` | The 5th percentile of the edge widths, in pixels. It follows the sharpest edges. Larger is softer. |
| `blur_edge_directionality` | `VISUAL_BLUR_EDGE_DIRECTIONALITY` | The absolute value of the base-2 logarithm of the median width of edges with a horizontal gradient over the median width of edges with a vertical gradient. It is 0 when the two agree and 1 when one is twice the other. |
| `background_blur_edge_width` | `VISUAL_BLUR_EDGE_WIDTH`, with `per_background=True` | The same percentile over only the edges outside the boxes and a margin of 9 pixels around them. |
| `blur_edge_width_median` | `BLUR_EDGE_WIDTH` | For a video, the median of `blur_edge_width` over the sampled frames. |

`blur_edge_width` describes the widths of the sharpest detected edges and is not a severity score for the whole image.
An edge width depends on the content of the scene and on how the edges are detected, so a comparison of
`blur_edge_width` between images of different resolution, texture or processing mixes those differences with blur.
Compare images of similar resolution and content.

## Request the statistics in DataEval

### Request the image statistics

`ImageStats.ALL` and `ImageStats.BLUR`, the group of the two flags, are the only named groups that hold them. `VISUAL`
does not, and neither does the default of {func}`.image_stats`, so a default {func}`.image_stats` or {class}`.Outliers`
run never runs the blur measurement. Ask for blur by flag:

```python
image_stats(images, stats=ImageStats.BLUR)
Outliers(flags=ImageStats.VISUAL | ImageStats.BLUR).evaluate(images)
```

Cost is the reason the default leaves them out. On one machine (32 cores, under WSL2), `blur_edge_width` on a 1280 × 720
frame took about 0.1 s against about 5 ms for `sharpness`. A 3840 × 2160 frame took about 1.3 s and allocated a peak of
about 800 MiB. Each figure is one frame at that size, so read it as an order of magnitude and not as a benchmark. The
image path does not cap the size of an image, so time a sample before you request blur for a dataset of large frames.

### Request the blur of the scene outside the boxes

With `per_background=True`, `VISUAL_BLUR_EDGE_WIDTH` also returns `background_blur_edge_width`, and
`background_<group>_blur_edge_width` for each named channel group. On a {class}`.Metadata` it is the unit-level factor
`unit_background_blur_edge_width`, beside `instance_blur_edge_width` for the boxes. The background holds NaN where the
boxes were, and the smoothing would spread those holes over every edge, so the edges are found on the whole image. The
value is then the 5th percentile of the widths of the edges that lie outside every box and a margin of 9 pixels around
each. The margin is there for loose boxes: an edge of the object just outside its box would otherwise count as
background. It is a constant fixed on study data and is provisional, as the size of 720 is.

Three things follow from the definition. The pool is smaller than `background_fraction` says, because that fraction is
measured on the boxes alone. A sharp object that nobody annotated counts as background, and a few such edges can set the
value. And the value is the width of the sharpest edges outside the boxes, not a measure of how defocused the background
is. The limits of `blur_edge_width` below carry over. `blur_edge_directionality` has no background value. Dividing an
object's `blur_edge_width` by `unit_background_blur_edge_width` is not a validated comparison, and no per-box ratio was
tested.

### Request the video statistics

`VideoStats.ALL` includes the group `VideoStats.BLUR`. `VideoStats.BASIC`, the default of {func}`.video_stats` and the
set that v1.2 named `NO_EGO`, does not, so a default call never runs the blur measurement and never needs OpenCV for
it. Ask for blur by flag:

```python
video_stats(videos, stats=VideoStats.BLUR)
```

The measurement needs OpenCV, from the `dataeval[opencv]` extra. At most `blur_samples` frames count toward the median,
64 by default and provisional, spread evenly over the video's whole length. Each is shrunk as
[Compare only at one resolution](#compare-only-at-one-resolution) describes and measured as an image is. A frame is
measured when it joins the sample and can be dropped as the video grows, so a long video measures more frames than
count.

On the same machine as the image timing, a `BLUR` request took 4.9 s to 25.6 s on three clips of 132 to 3,033 frames,
decoding included, and about 0.11 s to 0.15 s per measured frame beyond decoding. The 3,033-frame clip measured 166
frames, and 38 counted. Read these as an order of magnitude.

## Reading the values

A `blur_edge_width` or `blur_edge_width_median` value is an edge width in pixels at the resolution it was measured at.
It supports ranking images of one resolution, or videos of one frame size, against each other. It does not establish a
calibrated level of defocus, and values from different resolutions do not compare. A change in the value does not
identify its cause: noise and JPEG compression also move it, and `blur_edge_width` cannot tell them apart from a focus
defect. The subsections give the cases where a value is NaN, what moves it, and where the measurement stops.

### Where the values are NaN

A statistic with no reading reports NaN, and `blur_status` in the details says why. `blur_edge_width` and
`blur_edge_directionality` are NaN when:

- The picture has fewer than 100 edge points, as a flat picture, a tiny picture and a tiny bounding box do, or the
  median edge width is 0 (`insufficient_support`).
- A float image holds even one NaN pixel. The answer is withheld and not computed over the part that is left
  (`insufficient_support`).
- A bounding box extends past the border of the image (`insufficient_support`, while `invalid_box` is False), or lies
  outside it or has zero area (`invalid_box`).
- The data has no value range, such as signed float data (`unknown_range`). A warning names `value_range`. Declare the
  interval to get a reading, as [When there is no reference](ImageStatistics.md#when-there-is-no-reference) describes.

`background_blur_edge_width` has its own list. It is NaN when fewer than 100 edge points lie outside the boxes and their
margin, as a flat background or boxes that cover most of the image leave, or when the median width of those points is 0.
It is NaN when the boxes cover the whole image, when the image holds a NaN pixel anywhere, even inside a box, because
the edges are found on the whole image, and when the data has no value range. A box that extends past the border is
clipped to the image and is not a NaN case for the background.

Only `blur_edge_directionality` is NaN when fewer than 50 edge points have a horizontal gradient, when fewer than 50
have a vertical gradient, or when the median edge width of one axis is 0, and `blur_edge_directionality_reason` then
reads `insufficient_axis_support`. An image of vertical stripes reads a finite `blur_edge_width` and a NaN
`blur_edge_directionality`.

For a video, a sampled frame with too few edge points or a median edge width of 0 reads NaN and is left out of the
median, and `blur_edge_width_median` is NaN only when no sampled frame has a finite value. The video path reads the
8-bit gray frame of {func}`.to_canonical_grayscale`, which turns a NaN pixel into 0, stretches a float frame within 0 to
1 to 0 to 255 and clips other float values to 0 to 255. A float frame with a NaN pixel is therefore measured, and the
cases of a NaN pixel, a bounding box past the border and data with no value range do not apply to a video.

### What moves the values

The table reports the range over each of two sets of three frames from VisDrone and Wikimedia Commons drone clips. Set
(a) was reduced to 1280 × 720 by area averaging, degraded as 8-bit RGB and read by `image_stats`. Set (b) was reduced to
a short side of 720, converted to Rec. 601 luma as float32, degraded and read by the core function behind `image_stats`.
The 8-bit contrast, `sharpness` and half-size figures above are from set (a). The sets differ in frames and in pipeline,
so their numbers are not interchangeable, and the cause of the gaps was not traced.

| Change to the frame | `blur_edge_width`, set (a) | `blur_edge_directionality`, set (a) | `blur_edge_width`, set (b) | `blur_edge_directionality`, set (b) |
| --- | --- | --- | --- | --- |
| None | 0.62 to 0.67 | 0.02 to 0.18 | 0.62 to 0.83 | 0.01 to 0.24 |
| Contrast halved, no blur added | 0.62 to 0.67 | 0.02 to 0.18 | 0.62 to 0.83 | 0.01 to 0.24 |
| Gaussian blur, sigma 1 | 0.85 to 0.88 | 0.02 to 0.13 | 0.86 to 1.02 | 0.03 to 0.39 |
| Gaussian blur, sigma 2 | 1.20 to 1.22 | 0.00 to 0.07 | 1.22 to 1.39 | 0.04 to 0.35 |
| Gaussian blur, sigma 3 | 1.40 to 1.60 | 0.04 to 0.09 | 1.63 to 1.81 | 0.00 to 0.27 |
| Gaussian blur, sigma 4 | 1.18 to 1.69 | 0.10 to 0.12 | 2.04 to 2.31 | 0.02 to 0.20 |
| Noise, standard deviation 10 | 0.60 to 0.65 | 0.01 to 0.11 | 0.59 to 0.62 | 0.00 to 0.06 |
| JPEG, quality 20 | 0.65 to 0.67 | 0.01 to 0.20 | 0.64 to 0.71 | 0.00 to 0.13 |
| Horizontal motion blur, 15 pixels | 0.64 to 0.68 | 0.58 to 0.83 | 0.70 to 0.87 | 0.48 to 1.18 |
| Vertical motion blur, 15 pixels | 0.65 to 0.84 | 0.45 to 0.61 | 0.66 to 1.05 | 0.55 to 0.92 |

Read the table with these limits in mind:

- `blur_edge_width` rose with the sigma of the Gaussian blur on every frame, from 0 to 4 on set (b) and from 0 to 3 on
  set (a). On set (a) it fell on one frame between sigma 3 and sigma 4 (1.40 to 1.18).
- `blur_edge_width` is not a motion blur detector. A blur along one axis leaves the sharp edges of the other axis alone,
  so it responds weakly and unevenly. Across the twelve cases (six frames, two directions), a 15 pixel motion blur
  changed `blur_edge_width` by 0% to 30%, and a Gaussian blur of sigma 1 changed it by 23% to 38%.
- `blur_edge_directionality` sees the imbalance that motion blur makes: it read 0.45 to 1.18 after a 15 pixel motion
  blur in all twelve cases. It is unsigned, so it does not say which axis is softer, and its value on a clean frame
  depends on the scene: 0.01 to 0.24 across the six clean frames.
- Noise and JPEG compression are not neutral, and `blur_edge_width` cannot tell them apart from a focus defect. The
  change depends on the scene. Noise with a standard deviation of 10 lowered `blur_edge_width` on all six frames, by 1%
  to 29%, so those frames read sharper than they are, and on set (b) a standard deviation of 5 lowered one frame by 22%.
  JPEG at quality 20 lowered `blur_edge_width` on one frame and raised it on the other five.

### What moves the values of a video

The video check used the first 150 frames of each of 11 clips (all 132 of the shortest): seven VisDrone sequences and
four Wikimedia Commons clips. A Gaussian blur was added to the real frames, so the blur is simulated. On the clean
clips, `blur_edge_width_median` ran from 0.618 to 0.785.

- With the blur added at the decoded size, `blur_edge_width_median` rose at every step from sigma 0 to 3 on all 11
  clips. A blurred clip read higher than a different clean clip in 86 of 110 pairs at sigma 1, 109 at sigma 2 and 110 at
  sigma 3. At every sigma the closest pair involved the 3840 × 2160 clip, where sigma 1 at the decoded size is about
  0.33 pixel after the shrink.
- With the blur added after the shrink, so sigma is in measured pixels, a blurred clip read higher than a different
  clean clip in all 330 pairs. `blur_edge_width_median` rose at every step on only 8 of 11 clips and fell from sigma 2
  to 3 on the other three, for a cause not traced. Treat a large value as soft and not as a scale.
- Shrunk first to a short side of 360 or 480, the mean rise in `blur_edge_width_median` from sigma 0 to 1 was 0.245 and
  0.241, against 0.223 at 720. These are simulated small videos, and no other size was tried.
- At the same frame size, on three of the color clips, `blur_edge_width_median` was within 0.64% of the median
  `image_stats` `blur_edge_width` of the same frames. On four frames, clean and blurred at sigma 1.5, reading luma
  instead of the band average changed `blur_edge_width` by -2.19% to +0.77%. Given gray frames, the two paths agreed to
  float32 precision.
- The finding in [Soft 8-bit images at low contrast](#soft-8-bit-images-at-low-contrast) applies to
  `blur_edge_width_median`. The video check did not repeat it.

### Limits of the measurement

- A smooth float image with no real edge can still read a finite `blur_edge_width`. A 1024 × 1024 float32 diagonal ramp
  read 0.87.
- A video is measured at the size of its first frame. A sampled frame of any other size is left out of the median and
  logged, once for each size.
- The edge width measurement stops at ties in the gradient. A noise-free step edge reads 0.71, and so does the same step
  blurred with a box filter, which leaves the gradient flat across the blur. Build a synthetic test image with a
  Gaussian blur or with noise.

### Soft 8-bit images at low contrast

A soft 8-bit image at low contrast reads sharper than the same image at full contrast. The check blurred each of three
1280 × 720 frames with a Gaussian, scaled the contrast about mid-gray, rounded the result to 8 bits and read it with
`image_stats`. The table gives the range over the three frames of the change in `blur_edge_width`.

| Gaussian blur before the contrast change | Contrast halved | Contrast quartered |
| --- | --- | --- |
| None | -0.4% to 0.0% | -1.4% to -0.1% |
| Sigma 1 | -1.8% to -0.1% | -5.7% to -0.4% |
| Sigma 2 | -15.5% to -1.5% | -26.1% to -5.9% |
| Sigma 3 | -37.1% to -13.5% | -45.4% to -30.3% |

At sigma 1 or less (the first two rows), the width barely depends on contrast. For float input the width also barely
depends on contrast: on one frame, the same halving kept in float32 moved `blur_edge_width` by 0.01% at sigma 2 and at
sigma 3, against -5.2% and -32.5% after rounding to 8 bits. The float result rests on one soft frame and the three
frames of set (b) with no added blur. Why rounding to 8 bits moves the edge width this far was not traced, and the
evidence is three frames and a synthetic change of contrast. Until it is traced, compare the `blur_edge_width` of soft
8-bit images only against images of similar contrast.

### How far the evidence goes

The image evidence is six real frames in two sets of three, one frame per scene, with synthetic degradations. The check
of soft 8-bit images at low contrast reused the VisDrone frame of set (b) and took a third pair of frames from the two
clips of set (b), at frame index 39 instead of 120. The video evidence is 11 clips with simulated blur. Every clip has a
short side of 720 or more, so no video below 720 was measured. The runs at 480 and 360 use the same footage shrunk
first, so they are simulated small videos. The evidence holds no blur rated by people and no footage from a lens with a
real defect. The background value was checked on real frames and real boxes from two datasets, VisDrone2019-MOT frames
and CPPE-5 images, with a blur added to the background, to the boxes, or to neither. Every added blur was built from the
same boxes the statistic leaves out, and was a Gaussian, a circular-aperture blur or a blur that grows down the image,
so natural depth of field is not covered, and the margin was fixed on the same data. Values were measured on one machine
and were not compared across machines, between a CPU and a GPU, or across library versions. A second run of the video
check gave identical saved outputs apart from the timings.

## When to use the statistics

Use the statistics to rank images of one resolution against each other, to rank videos of one frame size against each
other and to compare a dataset with itself. [What moves the values of a video](#what-moves-the-values-of-a-video) gives
the limits for videos. Do not read a `blur_edge_width` or `blur_edge_width_median` value as a calibrated level of
defocus. [How far the evidence goes](#how-far-the-evidence-goes) gives the scope of the evidence behind this use.

### Compare only at one resolution

An edge width is in pixels of the image as it arrives, so the same scene at half the size has narrower edges, and values
from images of different resolutions do not compare. On three frames blurred with a Gaussian of sigma 3 pixels, halving
the width and height of the frame moved `blur_edge_width` from between 1.40 and 1.60 to between 1.03 and 1.07. Measure
at one resolution, and do not put a {class}`.Resize` in front of the statistic. A resize declares the blur statistics as
invalidated. [Transforms and the statistics they
invalidate](ImageStatistics.md#transforms-and-the-statistics-they-invalidate) describes the declarations.

{func}`.image_stats` warns once per call when blur is requested on images of more than one size. {class}`.Outliers`
makes one call for each dataset it is given, so datasets that each hold one size but differ from each other do not
warn.

A video frame is shrunk by area averaging to a short side of 720 pixels before it is measured, and never enlarged, so
videos with a short side of 720 or more are measured at one scale. The video check could not test whether videos of
different native sizes then compare. A video below 720 is measured at its own size, and its values compare only with
those of videos measured at that size. {func}`.video_stats` likewise warns once per call when blur is requested and its
videos are measured at more than one short side. The size of 720 is provisional. It is the scale at which the image
evidence was gathered, and a sweep did not choose it.

### Blur that changes within a video

`blur_edge_width_median` does not say how much the blur varies within a video. Measure the frames with
{func}`.image_stats` and `ImageStats.VISUAL_BLUR_EDGE_WIDTH`, and roll up the per-frame `blur_edge_width` of each video
with `std` or `variability`, as [Volatility and drift of a per-frame
statistic](MetadataLevels.md#volatility-and-drift-of-a-per-frame-statistic) shows for brightness. This recipe was not
run on blur. The roll-up reads a NaN `blur_edge_width` as missing and answers missing for a video with any frame
missing, unless you lower its `min_coverage`. The video path shrinks a frame above a short side of 720 and reads luma,
where `image_stats` keeps the size and averages the bands, so above that size the per-frame values are on another scale,
and for a color frame they differ from `blur_edge_width_median` even at the same size.

## Related concept pages

- [Image statistics](ImageStatistics.md): the pixel and visual statistics, the value range and the transforms that
  invalidate statistics
- [Data integrity](DataIntegrity.md): the evaluators these statistics feed
- [Metadata levels](MetadataLevels.md): how a per-frame statistic is rolled up to each video
- [Validation and trust](ValidationAndTrust.md#stability-status): the experimental features, including the blur
  statistics

## See this in practice

### How-to guides

- [Respond to a StatsInvalidatedWarning](../notebooks/h2_place_transforms.py): what an operation invalidates and the
  three ways to respond

## References

1. [Liang, L., Wang, S., Chen, J., Ma, S., Zhao, D., & Gao, W. (2010). No-reference perceptual image quality metric
   using gradient profiles for JPEG2000. *Signal Processing: Image Communication*, 25(7), 502-516.
   doi: 10.1016/j.image.2010.01.007 [paper](https://doi.org/10.1016/j.image.2010.01.007)]{#ref1}
