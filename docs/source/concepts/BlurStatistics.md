<!-- markdownlint-disable MD051 -->

# Blur statistics

The blur statistics measure how wide the edges in an image are, in pixels. They are two columns of
{func}`.image_stats`, `blur` and `blur_directionality`, and you request them with the
`ImageStats.BLUR` group. This page is a spinoff of [Image statistics](ImageStatistics.md), which describes the pixel and
visual statistics and the full-scale reference that the visual statistics rely on. Read that page first if those terms
are new.

The blur statistics are experimental ⚠️. The first request in a process raises
{class}`~dataeval.exceptions.ExperimentalWarning`, and the column names and values may change. A request for
`ImageStats.ALL` raises it too.

## Why measure the width of edges

`sharpness` measures how strong the edges in a picture are. It is the spread of a 3x3 edge filter, so it moves with the
contrast of the scene and with noise, and it cannot say how soft a lens is. The blur statistics measure the **edge
width**, the number of pixels an edge takes to go from dark to light.

Halving the contrast of three 8-bit frames with no added blur took `sharpness` from 57.5 to 32.8, from 40.7 to 22.5 and
from 40.6 to 21.8, and left `blur` unchanged to two decimals. A soft 8-bit image at low contrast is the exception, as
[Soft 8-bit images at low contrast](#soft-8-bit-images-at-low-contrast) describes.

## How the edge width is measured

The method follows Liang et al. ([2010](#ref1)). It finds the points where the smoothed gradient peaks along an axis,
measures how far the gradient magnitude spreads around each one, and corrects for the direction of the gradient.
DataEval summarizes the edge widths with percentiles, which is a change from the paper's own score, in two columns:

| Column | Flag | What it is |
| --- | --- | --- |
| `blur` | `VISUAL_BLUR` | The 5th percentile of the edge widths, in pixels. It follows the sharpest edges. Larger is softer. |
| `blur_directionality` | `VISUAL_BLUR_DIRECTIONALITY` | The absolute value of the base-2 logarithm of the median width of edges with a horizontal gradient over the median width of edges with a vertical gradient. It is 0 when the two agree and 1 when one is twice the other. |

## Request the statistics in DataEval

### Request the image statistics

`ImageStats.ALL` includes the two flags, which form the `ImageStats.BLUR` group. `VISUAL`, `VISUAL_BASIC`, `NO_HASH`,
`BASIC` and the default of {func}`.image_stats` do not, so a default {func}`.image_stats` or {class}`.Outliers` run never
runs the blur measurement. Ask for blur by flag:

```python
image_stats(images, stats=ImageStats.BLUR)
Outliers(flags=ImageStats.VISUAL | ImageStats.BLUR).evaluate(images)
```

Cost is the reason the default leaves them out. On one machine (32 cores, under WSL2), `blur` on a 1280x720 frame took
about 0.1 s against about 5 ms for `sharpness`. A 3840x2160 frame took about 1.3 s and allocated a peak of about
800 MiB. Each figure is one frame at that size, so read it as an order of magnitude and not as a benchmark. The image
path does not cap the size of an image, so time a sample before you request blur for a dataset of large frames.

With `per_background=True`, the background view returns no blur column, because the background holds NaN where bounding
boxes were and the smoothing would spread those holes over every edge.

## When to use the statistics

Use the statistics to rank images of one resolution against each other and to compare a dataset with itself. Do not read
a `blur` value as a calibrated level of defocus. [How far the evidence goes](#how-far-the-evidence-goes) gives the scope
of the evidence behind this use.

### Compare only at one resolution

An edge width is in pixels of the image as it arrives, so the same scene at half the size has narrower edges, and
values from images of different resolutions do not compare. On three frames blurred with a Gaussian of sigma 3 pixels,
halving the width and height of the frame moved `blur` from between 1.40 and 1.60 to between 1.03 and 1.07. Measure at
one resolution, and do not put a {class}`.Resize` in front of the statistic. A resize declares the blur statistics as
invalidated.
[Transforms and the statistics they invalidate](ImageStatistics.md#transforms-and-the-statistics-they-invalidate)
describes the declarations.

{func}`.image_stats` warns once per call when blur is requested on images of more than one size. {class}`.Outliers`
makes one call for each dataset it is given, so datasets that each hold one size but differ from each other do not
warn.

## Reading the values

### Where the values are NaN

A statistic with no reading reports NaN. Both columns are NaN when:

- The picture has fewer than 100 edge points, as a flat picture, a tiny picture and a tiny bounding box do, or the
  median edge width is 0.
- A float image holds even one NaN pixel. The answer is withheld and not computed over the part that is left.
- A bounding box extends past the border of the image.
- The data has no value range, such as signed float data. A warning names `value_range`. Declare the interval to get a
  reading, as [When there is no reference](ImageStatistics.md#when-there-is-no-reference) describes.

Only `blur_directionality` is NaN when fewer than 50 edge points have a horizontal gradient, when fewer than 50 have a
vertical gradient, or when the median edge width of one axis is 0. An image of vertical stripes reads a finite `blur`
and a NaN `blur_directionality`.

### What moves the values

The table reports the range over each of two sets of three frames from VisDrone and Wikimedia Commons drone clips. Set
(a) was reduced to 1280x720 by area averaging, degraded as 8-bit RGB and read by `image_stats`. Set (b) was reduced to a
short side of 720, converted to Rec. 601 luma as float32, degraded and read by the core function behind `image_stats`.
The 8-bit contrast, `sharpness` and half-size figures above are from set (a). The sets differ in frames and in
pipeline, so their numbers are not interchangeable, and the cause of the gaps was not traced.

| Change to the frame | `blur`, set (a) | `blur_directionality`, set (a) | `blur`, set (b) | `blur_directionality`, set (b) |
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

- `blur` rose with the sigma of the Gaussian blur on every frame, from 0 to 4 on set (b) and from 0 to 3 on set (a). On
  set (a) it fell on one frame between sigma 3 and sigma 4 (1.40 to 1.18).
- `blur` is not a motion blur detector. A blur along one axis leaves the sharp edges of the other axis alone, so it
  responds weakly and unevenly. Across the twelve cases (six frames, two directions), a
  15 pixel motion blur changed `blur` by 0% to 30%, and a Gaussian blur of sigma 1 changed it by 23% to 38%.
- `blur_directionality` sees the imbalance that motion blur makes: it read 0.45 to 1.18 after a 15 pixel motion blur in
  all twelve cases. It is unsigned, so it does not say which axis is softer, and its value on a clean frame depends on
  the scene: 0.01 to 0.24 across the six clean frames.
- Noise and JPEG compression are not neutral, and `blur` cannot tell them apart from a focus defect. The change depends
  on the scene. Noise with a standard deviation of 10 lowered `blur` on all six frames, by 1% to 29%, so those frames
  read sharper than they are, and on set (b) a standard deviation of 5 lowered one frame by 22%. JPEG at quality 20
  lowered `blur` on one frame and raised it on the other five.

### Limits of the measurement

- A smooth float image with no real edge can still read a finite `blur`. A 1024x1024 float32 diagonal ramp read 0.87.
- The edge width measurement stops at ties in the gradient. A noise-free step edge reads 0.71, and so does the same step
  blurred with a box filter, which leaves the gradient flat across the blur. Build a synthetic test image with a
  Gaussian blur or with noise.

### Soft 8-bit images at low contrast

A soft 8-bit image at low contrast reads sharper than the same image at full contrast. The check blurred each of three
1280x720 frames with a Gaussian, scaled the contrast about mid-gray, rounded the result to 8 bits and read it with
`image_stats`. The table gives the range over the three frames of the change in `blur`.

| Gaussian blur before the contrast change | Contrast halved | Contrast quartered |
| --- | --- | --- |
| None | -0.4% to 0.0% | -1.4% to -0.1% |
| Sigma 1 | -1.8% to -0.1% | -5.7% to -0.4% |
| Sigma 2 | -15.5% to -1.5% | -26.1% to -5.9% |
| Sigma 3 | -37.1% to -13.5% | -45.4% to -30.3% |

The first two rows show that at sigma 1 or less the width barely depends on contrast. For float input it barely depends
on contrast: on one frame, the same halving kept in float32 moved `blur` by 0.01% at sigma 2 and at sigma 3,
against -5.2% and -32.5% after rounding to 8 bits. That claim rests on one soft frame and the three frames of set (b)
with no added blur. Why rounding to 8 bits moves the edge width this far was not traced. The evidence is three frames
and a synthetic change of contrast. Until it is traced, compare the `blur` of soft 8-bit images only against images of
similar contrast.

### How far the evidence goes

The evidence is six real frames in two sets of three, one frame per scene, with synthetic degradations. The check of
soft 8-bit images at low contrast reused the VisDrone frame of set (b) and took a third pair of frames from the two
clips of set (b), at frame index 39 instead of 120. The evidence holds no blur rated by people and no footage from a
lens with a real defect. Values were measured on one machine and were not compared across machines or library versions.

## Related concept pages

- [Image statistics](ImageStatistics.md) describes the pixel and visual statistics, the value range and the transforms
  that invalidate statistics.
- [Data integrity](DataIntegrity.md) describes the evaluators these statistics feed.
- [Stability status](ValidationAndTrust.md#stability-status) in Validation and trust lists the blur statistics among
  the experimental features.

## See this in practice

### How-to guides

- [Respond to a StatsInvalidatedWarning](../notebooks/h2_place_transforms.py) shows what an operation invalidates and
  the three ways to respond.

## References

1. [Liang, L., Wang, S., Chen, J., Ma, S., Zhao, D., & Gao, W. (2010). No-reference perceptual image quality metric
   using gradient profiles for JPEG2000. *Signal Processing: Image Communication*, 25(7), 502-516.
   doi: 10.1016/j.image.2010.01.007 [paper](https://doi.org/10.1016/j.image.2010.01.007)]{#ref1}
