<!-- markdownlint-disable MD051 -->

# Image statistics

{func}`.image_stats` measures images and the boxes annotated in them. Its statistics fall into families, which the
[map of families](#statistic-families) lists. Four questions decide how to read a family's values: what it measures,
what a result can support, what it cannot establish, and when a value can be compared across a dataset.
{class}`~dataeval.flags.ImageStats` lists every statistic, and {func}`~dataeval.core.image_stats` gives the parameters.

A statistic read outside its scope gives a wrong comparison. A brightness compared across two sensors includes the
difference between the cameras as well as between the scenes, and a blur width compared across two resolutions
includes the difference in resolution. Each family below states the condition under which its values compare.

Every statistic is descriptive. A value describes the data and is not evidence that a sample is bad, so an unusual
value is a reason to look at a sample and not a finding about it.
[Validation evidence](../reference/ValidationEvidence.md#image-statistics-and-hashing) records how each family was
checked.

## Statistic families

| Family | What it measures | Rows |
| --- | --- | --- |
| [Pixel](#pixel-statistics) (`PIXEL`) | The stored values: their moments, histogram and entropy, and the shares that are missing, zero or at a clipping endpoint | Image and box |
| [Noise](#noise-statistics) (`NOISE`) | The level of additive white Gaussian noise, estimated from the pixels | Image and box |
| [Visual](#visual-statistics) (`VISUAL`) | Appearance on the 0–255 display range: brightness, darkness, contrast and sharpness | Image and box |
| [Blur](#blur-statistics) (`BLUR`) ⚠️ | The width of the sharpest edges in pixels, and how horizontal and vertical edges differ | Image and box |
| [Color](#color-statistics) (`COLOR`) ⚠️ | Lightness, chroma, the share of neutral pixels and the mean hue, in CIELAB, from channels declared as sRGB | Image and box |
| [Dimension](#dimension-statistics) (`DIMENSION`) | Size, shape, position, channel count and bit depth, in pixels and relative to the image | Image and box |
| [Hash](#hash-statistics) (`HASH`) | Exact and perceptual hashes of the content | Image and box |
| [Spatial](#spatial-statistics) (`SPATIAL`) | A box against the other boxes of its image and the image's edges | Box only |
| [Context](#context-statistics) (`CONTEXT`) | The intensity around a box, and how far the box stands from it | Box only |

⚠️ marks an experimental family, whose names and values may change.

Each statistic is measured over a **region** of an image: the whole image, a box, the background or the surround of a
box. The API reference calls a region a view; on this page a view is a {class}`.View`. Five points apply across the
families:

- A column holds image rows and box rows together, so compare an image with an image and a box with a box.
- A box row measures the whole rectangle, including any background inside a loose box.
- Several families also measure the background, the pixels outside every box, under a `background_` name.
  [Data integrity](DataIntegrity.md#image-statistics-in-integrity-checks) defines that region.
- A value that a statistic cannot measure is NaN, not 0, and several families report why in a status detail.
- Any column can become a factor in {class}`.Metadata`. {class}`.Outliers` flags images and boxes whose values are
  unusual, and {class}`.Duplicates` compares the hashes.

## Raw values and visual interpretation

A **pixel statistic** reduces the values as they were stored. `PIXEL_MEAN` on a
16-bit image is a mean of values that can range from 0 to 65,535, because that is what the file
contains. It describes the data.

A **visual statistic** approximates how the image looks to a person. `VISUAL_BRIGHTNESS`
describes appearance, which is a position between black and white and not a value in the
sensor's units. A visual statistic therefore first resolves the image's **full-scale
reference** (what counts as white) and reports against the 0–255 display range.

That difference is why the same picture, stored four ways, gives the same value for each
of the three visual statistics shown. Here `brightness` is the lower quartile of the display
values and `darkness` is the upper quartile, so `darkness` is the larger of the two and rises
as the image gets brighter:

| Encoding | `brightness` | `darkness` | `sharpness` |
| --- | --- | --- | --- |
| `uint8` | 61.0 | 191.0 | 112.11 |
| `uint16` (the same values ×257) | 61.0 | 191.0 | 112.11 |
| `float` in [0, 1] | 61.0 | 191.0 | 112.11 |
| `float` in [0, 255] | 61.0 | 191.0 | 112.11 |

The pixel statistics of the same picture differ:

| Encoding | `mean` | `std` |
| --- | --- | --- |
| `uint8` | 126.88 | 75.42 |
| `uint16` (the same values ×257) | 32607.54 | 19382.91 |

Both tables are correct and describe different properties. Visual statistics characterize normalized appearance, and
pixel statistics characterize the stored values. "Is this image dark?" is a question about appearance and has one
answer however the picture is stored. "What is the mean of these values?" is a question about the data and has two
answers, because there are two sets of values.

Scaling to the display range removes differences of encoding only. It is a linear map of the reference interval onto
0–255, so it does not undo a sensor's response curve, a gamma or a gain. A visual statistic therefore cannot separate a
difference in the scene from a difference in the camera, and a comparison across sensors or exposure settings includes
both.

## Where the reference comes from

The reference is not the image's brightest pixel. Normalizing each image against its own maximum would bring a
photograph of a night sky and one of a beach to the same full scale, so brightness values would not be comparable
across images.

Instead the reference is **decoded** where an encoding exists, because integer image formats use power-of-two
ranges:

| Input | Reference |
| --- | --- |
| Integer, largest value 2 to 255, such as `uint8` | 255 |
| Integer, largest value 256 to 4095, such as a 12-bit sensor's output in `uint16` | 4095 |
| Integer, largest value 4096 to 65535, such as 16-bit output | 65535 |
| `float` within [0, 1] | 1.0 (already normalized) |
| `float` within [0, 255] | 255 (an 8-bit image in a float array) |

For an integer array, DataEval reads the smallest of 1, 8, 12, 16 and 32 bits whose range holds the array's largest
value, so the reference follows the dtype only through that value. Two frames from one sensor are read against different
references when their brightest pixels fall in different ranges, and a `uint16` frame with no value above 255 is read as
8-bit. The array's dtype does not record how a sensor's output was scaled: a 12-bit output shifted into the high bits of
a `uint16` has a largest value above 4095 and is read as 16-bit. Check the acquisition scaling and any clipping before
comparing intensity statistics across images, and declare the interval with `value_range` when it is known, as the next
section describes.

Two of these are float conventions and not encodings. They are recognized because they are two common ways an image
arrives as a float array: a `ToTensor`-style pipeline produces the first, and a resize or other interpolation produces
the second.

Not every statistic uses the decode. Without a declared interval:

| Statistics | Interval |
| --- | --- |
| Visual and blur statistics, the histogram, and the other pixel statistics under `normalize_pixel_values=True` | The decode above |
| Context, color and saturation statistics, and the noise statistics under `normalize_pixel_values=True` | `(0, 255)` for `uint8`, and the two float conventions above; any other data is NaN with the status `unknown_range` |

The context, color and saturation statistics, and the noise statistics when normalized, never read an integer image's
own maximum, so two frames of one sensor cannot read on different scales, and a 12-bit sensor's output in `uint16`
needs a declaration. The float conventions still read the data's extremes: a 0-255 float image whose values all lie at
or below 1 is read as the 0-1 convention. The reverse matters for the noise statistics: noise added to a 0-1 float
image and not clipped puts some values above 1, which reads that image on the 0-255 convention and its noise 255 times
too small, and a value below 0 leaves it with no interval. Declare `value_range` for floating-point data.

## When there is no reference

Some data has none, such as elevation below sea level, mean-centered reflectance, temperature in Celsius or a 16-bit
band holding physical units. The dynamic range of such data is a property of the sensor, not of a file format, and
nothing about the array reveals it. There is no encoding to decode.

For that data, visual statistics report NaN. This is not an error. The question of how bright an elevation map is has no
answer, and NaN represents an unknown value. Pixel statistics (and other statistics that do not rely on an interval) are
unaffected, because the mean of a temperature band is a valid mean.

Declare the interval to get a reading back:

```python
image_stats(elevation_data, stats=ImageStats.VISUAL, value_range=(-2000.0, 2000.0))
```

A declaration says "these are physical values spanning this range." It implies no bit
depth, so `DIMENSION_DEPTH` reports NaN for such data, except for integer data declared as `(0, 2**n - 1)`, which
reports `n`. A dataset can declare the interval for each image, as a `(low, high)` pair under `value_range` in the
datum's metadata, and {func}`~dataeval.core.image_stats` gives the order when more than one is declared.

Pixel statistics that need an interval also answer NaN: `PIXEL_HISTOGRAM`,
`PIXEL_ENTROPY`, and the rest of the statistics under `normalize_pixel_values=True`,
which is a request to divide by a range that does not exist.

When this happens, a warning names `value_range`.

The statistics in the second row of [the table above](#where-the-reference-comes-from) need an interval too, by the
rule that row gives.

## Statistics that depend on scale

Few statistics depend on scale. Of the pixel statistics, only five change when the values are rescaled:

| Changes with scale | Already scale-free |
| --- | --- |
| `mean`, `std`, `var` | `skew`, `kurtosis` (standardized moments) |
| `noise_std`, `noise_var` | `entropy`, `histogram` (binned over the reference either way) |
| | `missing`, `zeros`, `saturated_fraction` (counts) |

`normalize_pixel_values=True` applies to those five only. It divides `mean`, `std` and `var` by the same
reference the visual family uses, which makes a pixel distribution comparable across bit depths. It divides the
[noise statistics](#noise-statistics) by the interval the context and color statistics read, in
[the table above](#where-the-reference-comes-from), so data with no such interval reads NaN with the status
`unknown_range`.

It does not change visual statistics, which are already scaled to the display range.

## Choosing between pixel and visual statistics

Use a visual statistic when the question is about appearance and you want one answer across a dataset that mixes
encodings, such as which images are unusually dark, which are blurry, or whether the test set is shot in different
lighting than the train set. Where the images come from different sensors, a difference may come from the camera as well
as the scene, as described above.

Use a pixel statistic when the question is about the values themselves, such as what fraction of the values are zero or
[saturated](#saturation-at-a-clipping-endpoint), how widely they spread across their range, or whether there are NaNs.
Where the dataset is uniformly encoded, its raw values are already comparable and nothing extra is needed.

Where a dataset mixes encodings and you want comparable pixel distributions, pass
`normalize_pixel_values=True`. Check whether `VISUAL_BRIGHTNESS` answers the question
more directly; a normalized `PIXEL_MEAN` is often a less direct route to the same result.

## Pixel statistics

The pixel statistics reduce the stored values in their own units: their mean, spread and shape, a 256-bin histogram,
and the shares of values that are missing, zero or at a clipping endpoint. Two of them need a definition:

- Entropy is the Shannon entropy, in nats, of the 256-bin histogram. It describes how the values spread across
  their range. Rearranging the pixels leaves it unchanged, so it does not measure texture.
- Kurtosis is excess kurtosis, the fourth standardized moment minus 3, so a normal distribution of values reads 0.

A pixel statistic answers a question about the data, such as whether a file holds NaNs or how widely its values spread.
It does not describe appearance. The same picture can have a mean of 126.88 in one encoding and 32607.54 in another,
as [the tables above](#raw-values-and-visual-interpretation) show. Values compare across images of one encoding. Across
encodings, the statistics that change with scale compare only when normalized, as
[Statistics that depend on scale](#statistics-that-depend-on-scale) describes, and the others are already scale-free.

Use the pixel statistics to check the stored data, such as a dataset that mixes bit depths or a pipeline step that
introduced missing values.

### Saturation at a clipping endpoint

`ImageStats.PIXEL_SATURATED_FRACTION` is a pixel statistic: the share of finite raw values that lie within a tolerance
of a clipping endpoint. It counts channel values as stored, and not pixels or colors, so an RGB image clipped in one
channel reads 1/3, and an RGBA image reads at least 1/4 for its opaque alpha channel. To leave the alpha channel out,
measure the color bands as a named [group of channels](../notebooks/h2_measure_channel_groups.py) and read that group's
column. Values above the endpoint stay in the denominator and are not counted as saturated.

DataEval does not decode the endpoint from the data. It is the upper bound of the interval that
[the table above](#where-the-reference-comes-from) gives the context, color and saturation statistics, and the value is
NaN when none applies. A depth decoded from the data depends on the data's own maximum, so one sensor's images could be
read against different endpoints. Twelve-bit data stored in `uint16`, which the table above decodes as 4095 for
scaling, therefore reads NaN until its range is declared as `(0, 4095)`.

A reading of 0.0 shows that no value sits at the stated endpoint and does not establish that an image is unclipped. On
floating-point data that has been resized or otherwise computed on, a clipped region may land slightly off the
endpoint, so such data needs a tolerance, which {func}`.image_stats` describes.

## Noise statistics

The noise statistics estimate the standard deviation and the variance of **additive white Gaussian noise**, noise that
is independent from pixel to pixel and normally distributed with one variance, from the pixels alone. They are pixel
statistics, in the data's own units.

DataEval reads each channel $I$ with the estimator of Immerkær ([1996](#ref1)). The channel is convolved with a mask
that takes the second difference along the rows and then along the columns, and the estimate is the mean absolute
response, scaled:

$$
M = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}, \qquad
\hat{\sigma} = \sqrt{\frac{\pi}{2}} \, \frac{1}{6} \, \operatorname{mean} \lvert I * M \rvert .
$$

The squared coefficients of $M$ sum to 36, so on white Gaussian noise of standard deviation $\sigma$ each response has
a standard deviation of $6\sigma$ and an absolute value whose mean is $6\sigma\sqrt{2/\pi}$. The mean runs over the
3x3 neighborhoods that hold no NaN and no masked pixel, so a masked or missing pixel creates no edge. `noise_var` is the
mean of the channels' $\hat{\sigma}^2$ and `noise_std` is its square root. A small region has too few neighborhoods to
read, so a small box can read NaN. Normalized values are divided by an interval, as
[Statistics that depend on scale](#statistics-that-depend-on-scale) describes. The noise of a box's
[surround](#context-statistics) can be measured too. {func}`~dataeval.core.noise_immerkaer` computes the estimate for
one array.

A difference between readings provides evidence of a difference in noise only between images of the same content. Fine
detail of the scene itself, or texture, and edges that are not parallel to a side of the image add to a reading, and
pixels that carry no noise, such as letterbox bars or a clipped area, lower it. A high value, and an image that
{class}`.Outliers` flags on one, can therefore come from the content as well as from noise. A single reading over a
small region is widely spread, so compare the distributions of many readings. The estimator was checked on simulated
noise only. The Notes of {func}`~dataeval.core.noise_immerkaer` state the patterns the mask removes, what raises or
lowers a reading, and the kinds of noise that were not measured.

Use the noise statistics to compare the noise of images of the same content, such as the frames of one fixed camera at
different gains.

## Visual statistics

The visual statistics read the values against the full-scale reference, on the 0–255 display range. Four of them need a
definition:

- Brightness is the 25th percentile of the display values, the lower quartile.
- Darkness is the 75th percentile, the upper quartile. It is at least the brightness, and it rises as the image
  gets brighter.
- Contrast is the range of the five reported percentiles (0, 25, 50, 75 and 100) divided by their mean,
  (max − min) / mean, and has no units. The range runs from the darkest pixel to the brightest, so one extreme pixel
  raises it. It describes one image or box, and is not the context contrast, which compares a box with its surround.
- Sharpness is the spread of edge magnitudes from a 3x3 edge kernel.

A visual statistic answers a question about appearance, with one answer however the picture is stored. It cannot
separate the scene from the camera, as [Raw values and visual interpretation](#raw-values-and-visual-interpretation)
describes, and it is NaN for data with [no reference](#when-there-is-no-reference). Values compare across encodings
that have a reference. Between images from different sensors, a difference includes the cameras. A resize makes
sharpness measure the interpolation kernel, as
[the transforms section](#transforms-and-the-statistics-they-invalidate) describes.

Use the visual statistics for questions about appearance across a dataset that mixes encodings.

## Blur statistics

The blur statistics are visual statistics that measure the width of the sharpest edges in pixels, and how much the
widths of horizontal and vertical edges differ. They are experimental ⚠️. A width is in pixels at the image's own
resolution, so values compare only at one resolution, and only between images of similar contrast. They are not a
calibrated level of defocus. [Blur statistics](BlurStatistics.md) describes the method and its limits.

Use the blur statistics to compare focus within footage of one resolution.

## Color statistics

The color statistics are visual statistics that measure the color of an image, a box or the background in the CIELAB
color space: the mean lightness and chroma, the share of neutral pixels, and the mean hue. They are experimental ⚠️.
The color of a box's [surround](#context-statistics) can be measured too.

Nothing in the pixels says which channels are red, green and blue, or that they are sRGB, so the caller declares them,
and the color statistics are not computed without that declaration. The values are only as true as the declaration: a
false-color or thermal image read as sRGB gives numbers that have no color meaning.

A color needs an absolute scale. The scale is the one in [the table above](#where-the-reference-comes-from), never an
integer image's own maximum, which would make the same pixels read as different colors. A 16-bit image therefore needs
a declared range.

Lightness runs from 0 for black to 100 for white. Chroma is 0 for a neutral pixel and grows with how colorful it
is, and the share of neutral pixels counts those whose chroma is under a threshold. "Chroma" and "hue" are CIELAB terms
and have nothing to do with a codec's chroma subsampling. The lightness is that of the sRGB color once its transfer
function is undone, which is neither `brightness`, a percentile of display values, nor `mean`, a mean of stored values.

Hue is a position on the color wheel, which is a circle, so an average of angles misleads. The arithmetic mean of 350
and 10 degrees is 180, while the two hues lie on either side of 0. `color_hue_cos_mean` and `color_hue_sin_mean` are the
coordinates of the average direction of the colorful pixels, weighted so that a more colorful pixel counts more. The
point lies near the rim when those pixels share one hue and near the center when their hues are mixed, and
`atan2(color_hue_sin_mean, color_hue_cos_mean)` gives the mean angle where it is not near the center. Both are NaN where
no pixel is colorful.

The threshold between neutral and colorful is a default chosen without study data and is provisional.

The statistics describe stored color, so a difference between two sets of images can be a difference in the scene, the
camera or its white balance, and they cannot tell these apart. Video in a gamma other than sRGB, wide-gamut files and
chroma subsampling were not measured, and the statistics read all of them as sRGB.

Use the color statistics to compare the color of sets of images from one camera and one white balance, such as a train
and a test split.

## Dimension statistics

The dimension statistics describe the geometry of an image or a box: its width, height, area, aspect ratio and offset,
the distances from its center to the image's center and from its sides to the image's edges, its channel count and its
bit depth. Two of them need a definition:

- Aspect ratio is 1 − short side / long side, positive for a box wider than it is tall and negative for a taller
  one. A square reads 0, a 2:1 box 0.5 and a 1:2 box −0.5, and every value lies between −1 and 1.
- Distance to the edge is the smallest of the four absolute gaps between the box and the image edges, in pixels. It
  is unsigned, so a box that extends past an edge reads as a positive distance, as one inside the image does. The
  [spatial statistics](#spatial-statistics) give the share of a box inside the image, which tells the two apart.

The statistics in pixels compare at one resolution. Their relative forms place a box within its image, as a fraction of
the image's width, height, area or half-diagonal, so they compare boxes across resolutions. On an image's own row they
read the whole frame, such as a relative size of 1. A resize makes the statistics report the resize target, as
[the transforms section](#transforms-and-the-statistics-they-invalidate) describes. The bit depth is decoded from the
data, or from a declared interval as [When there is no reference](#when-there-is-no-reference) describes.

The statistics describe geometry as it was stored or annotated. An unusual size or aspect ratio is a reason to inspect a
box or an image, and does not show that the box was drawn wrongly or the image resized.

Use the dimension statistics to check that images and boxes have the sizes and shapes the dataset should have.

## Hash statistics

The hash statistics summarize content as hashes. xxHash is an exact hash of the stored bytes. pHash and dHash are
perceptual hashes, which give similar values for images that look alike after compression, a small crop or a change of
brightness, and their D4 variants also match the eight rotations and flips of a square. Two hashes compare only within
one hash type.

A matching exact hash is evidence of identical bytes and not a guarantee of it, since a non-cryptographic hash can
collide. A close perceptual hash says two images look alike, not that one was made from the other, and it does not
survive arbitrary warping or rotation by other angles.
[Duplicate detection](DataIntegrity.md#duplicate-detection-hashing-and-clustering) describes the methods, and
{class}`.Duplicates` uses them.

Use the hash statistics to look for exact and near copies within a dataset or across its splits.

## Spatial statistics

The spatial statistics measure a box against the other boxes of its image and against the image's edges, so they have
values on box rows only. They are:

- the distance from the box's center to the center of the nearest other box, over the box's own diagonal;
- the share of the box's area inside the image;
- the share of the box's area that the other boxes cover;
- the largest intersection over union of the box with another box of its class, and with a box of another class.

Boxes are rectangles, and an object that nobody annotated covers nothing, so the statistics describe the annotations
and not the objects. The covered fraction is not occlusion: it counts the area of the box that other boxes cover, and
does not say which object lies in front. The distance is center to center, not the gap between two boxes, and it is
directed, since each box divides by its own diagonal, so two boxes can read different distances to each other.

A large overlap with a box of another class marks a pair that may be two objects, such as a rider on a bicycle, or two
annotations of one object, which can be a conflicting label. The class overlaps need class labels, and passing labels
to them is experimental ⚠️. Where a missing label could hide the largest overlap, the value is NaN, not a guess.

The ratios are scale-free, so they compare across resolutions. The share inside the image depends on whether the
annotation tool clips boxes to the image: it reads 1.0 wherever the boxes were clipped, including after a
{class}`.Resize` or a {class}`.Crop`.

Use the spatial statistics to choose boxes for annotation review, such as crowded boxes, boxes cut by the image edge
and boxes that lie on a box of another class.

## Context statistics

The context statistics compare a box with its **surround**: a frame around the box, whose width is a fixed share of
the box's width and height, less the box and less every other box of the image. They have values on box rows only. The
surround mean and standard deviation describe the intensity of the surround, where intensity is the value mapped to
[0, 1] from the image's interval and averaged over the channels, and a pixel counts only if every channel lies in the
interval. The surround is one box's own neighborhood, while the background pools the pixels outside every box over the
whole image.

An object that no box annotates stays in the surround, and a box near the image edge reads a surround that the edge has
cut. A small box, or one whose surround the other boxes cover, has too few surround pixels to read and is NaN.

The color and noise statistics can also be measured over each box's surround, beside the box's own values. That
measurement is experimental ⚠️.

Values compare across images measured against one interval, and across boxes annotated by one convention, since a
loose box reads nearer its surround than a tight one.

Use the context statistics to compare objects with what they are seen against, such as one class over different
backgrounds.

### The target-surround ratios

`target_surround_contrast` and `signal_clutter_ratio` compare a box's mean intensity with its surround's, as a
difference relative to the surround's level and to its spread, in the manner of a Weber contrast. The sign says
whether the box is brighter or darker than its surround. The intensity is the encoded value mapped to [0, 1], so the
ratios are not contrasts of physical luminance, which gamma and the sensor's response separate from the stored values.
{class}`.ImageStats` gives the formulas. Four limits decide what a value can support.

- The box is the target. Its mean covers every pixel of the box, including the pixels of overlapping neighbors and of
  background inside a loose box, so a loose box reads nearer its surround than the object does.
- The intensity is averaged over the channels, so a box that differs from its surround in hue and not in brightness
  reads near 0.
- A dark or flat surround puts a small number under the division. Each denominator has a floor, and where the floor
  was used the details say so: the value is then mostly the difference over the floor, and says little of the surround.
- Intensity is `(value - low) / (high - low)`, so when a declared range does not start at 0 the denominator of the
  contrast depends on that low end. Declare the range where images are compared.

A ratio near 0 therefore does not establish that an object is hard to see, and a large one does not establish that it
is easy to find.

## Transforms and the statistics they invalidate

A transform can be placed in the view or in the extractor, and the wrong placement raises no error. Model
preprocessing placed in the view is seen by every evaluator, so {class}`.Outliers` then reports brightness and contrast
on ImageNet-normalized tensors, which do not describe the data. Dataset-defining manipulation placed in the extractor is
seen only by the embeddings. If 512 × 512 tiles are what ships, the statistics still describe source-resolution
imagery, which does not ship.

Two properties place a transform. Place it in the {class}`.View` when an evaluator that never touches the model must
see it, and in the extractor's `transforms=` otherwise. Use a curated operation for a deterministic transform with an
unambiguous rewrite of the targets, and {class}`.TorchvisionTransform` for any other.

| Kind | Example | Placement |
| --- | --- | --- |
| Model-input adaptation | resize-to-224, ImageNet normalize, dtype cast | the extractor's `transforms=` |
| Dataset definition | crop out a HUD, downsample to the deployed resolution, band selection | {class}`.Resize`, {class}`.Crop` and {class}`.SelectChannels` in a view |
| Augmentation or corruption | random flip, color jitter, blur | {class}`.TorchvisionTransform` |

Use {class}`.Resize` for the resolution your system deploys at, because that is a fact about the data under evaluation
and not about the model. Use {class}`.Crop` to remove a fixed region that is never evidence, such as a burned-in
overlay, a scan border or a dead sensor region. Use {class}`.SelectChannels` to narrow or combine channels, for a mono
sensor stored as RGB or for the bands of a multispectral cube that you evaluate.

Every operation declares the statistics it invalidates. A resize makes the dimension statistics, such as `width`,
`height` and `aspect_ratio`, report the resize target, makes `sharpness` measure the interpolation kernel, and rescales
the blur statistics, which are edge widths in pixels. It changes the spatial statistics, because the boxes are clipped
to the new canvas and those left outside it are dropped. It changes the context statistics, because the surround of a
box is made of interpolated pixels. It changes the saturation statistic, because a resize returns floats, which take
their endpoint from a float convention read off the data's extremes. It changes the noise statistics, because
interpolation smooths the noise and a letterbox adds flat bars. It leaves the other pixel statistics, the channel count
and the hashes alone, so resize-then-hash may be a better near-duplicate check across heterogeneous source resolutions
than hashing at source resolution.

The quality evaluators intersect the declaration with the requested {class}`.ImageStats`, and where the two overlap
they raise a {class}`~dataeval.exceptions.StatsInvalidatedWarning`. Nothing is blocked. The invalidated statistics now
describe the transform. [Respond to a StatsInvalidatedWarning](../notebooks/h2_place_transforms.py) shows the three
responses.

The declarations are conservative. {class}`.TorchvisionTransform` declares `ImageStats.ALL`, the spatial, context,
saturation and noise statistics included, because an arbitrary transform can move anything. {class}`.SelectChannels`
with `"gray"` declares the blur, saturation, context and noise statistics too, because a luminance mix changes which
picture is read.

Random transforms are seeded for each datum from the datum's `id`, so the view is stable. The statistics pass, the
embedding pass and the duplicates pass see the same pixels, and iterating twice gives identical bytes. A view built
through {class}`.TorchvisionTransform` may not be reconstructable from its sidecar, because provenance records the
`repr` of each operation. The `repr` of a curated operation can be pasted back to rebuild it. A `v2.Compose([...])` has
a multi-line `repr`, and a lambda inside one records nothing recoverable.

## Related concept pages

- [Data integrity](DataIntegrity.md): the evaluators these statistics feed
- [Blur statistics](BlurStatistics.md): the experimental edge-width statistics

## See this in practice

### How-to guides

- [Measure named groups of channels](../notebooks/h2_measure_channel_groups.py): naming groups of bands so
  that each group is measured separately
- [Respond to a StatsInvalidatedWarning](../notebooks/h2_place_transforms.py): what an operation invalidates
  and the three ways to respond
- [Specify custom statistics on object detection datasets](../notebooks/h2_custom_image_stats_object_detection.py):
  statistics of images, boxes and the background region
- [Add intrinsic factors to Metadata](../notebooks/h2_add_intrinsic_factors.py): pixel and visual statistics as
  metadata factors
- [Find crowded, truncated and conflicting annotations](../notebooks/h2_find_annotation_review_candidates.py): the
  spatial statistics as an annotation review list
- [Identify duplicates](../notebooks/h2_deduplicate.py): the hashes behind exact and near-duplicate detection

### Tutorials

- [Clean a dataset](../notebooks/tt_clean_dataset.py): the pixel, visual and dimension statistics through
  {class}`.Outliers`
- [Compare objects to their surroundings](../notebooks/tt_compare_objects_to_surroundings.py): box statistics beside
  the statistics of the background region

## References

1. [Immerkær, J. (1996). Fast noise variance estimation. *Computer Vision and Image Understanding*, 64(2), 300-302.
   doi: 10.1006/cviu.1996.0060 [paper](https://doi.org/10.1006/cviu.1996.0060)]{#ref1}
