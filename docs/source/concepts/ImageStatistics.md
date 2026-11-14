# Image statistics

{func}`.image_stats` measures four families of statistics about an image (pixel, visual, dimension, and hash), plus
the experimental [blur statistics](BlurStatistics.md). Two of the families, pixel and visual, answer different kinds of
questions, and the distinction determines whether a value can be compared across a dataset. The flags and their
parameters are in {func}`~dataeval.core.image_stats`, and the full statistic list is in
{class}`~dataeval.flags.ImageStats`.

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

and why its pixel statistics differ:

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

## When there is no reference

Some data has none. Elevation below sea level, mean-centered reflectance, temperature
in Celsius, a 16-bit band holding physical units: the dynamic range of these is a
property of the sensor, not of a file format, and nothing about the array reveals it.
There is no encoding to decode.

For that data, visual statistics report NaN. This is not an error: "how bright is this elevation map" has no answer,
and NaN represents an unknown value. Pixel statistics (and other statistics that do not rely on an interval) are
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

When this happens, a warning names `value_range`, because an all-NaN column is easy to miss.

## Statistics that depend on scale

Few statistics depend on scale. Of the pixel family, only three change when the values
are rescaled:

| Changes with scale | Already scale-free |
| --- | --- |
| `mean`, `std`, `var` | `skew`, `kurtosis` (standardized moments) |
| | `entropy`, `histogram` (binned over the reference either way) |
| | `missing`, `zeros`, `saturation` (counts) |

`normalize_pixel_values=True` exists for those three, and only for them. It divides by
the same reference the visual family uses, which makes a pixel distribution comparable
across bit depths.

It does not change visual statistics, which are already scaled to the display range.

## Definitions of selected statistics

Each row below states what {func}`.image_stats` computes for a statistic whose name does not give its definition.

:::{list-table}
:widths: 28 72
:header-rows: 1

- - Statistic
  - What it measures
- - `VISUAL_BRIGHTNESS`
  - The 25th percentile of the display values, which is the lower quartile.
- - `VISUAL_DARKNESS`
  - The 75th percentile of the display values, which is the upper quartile. It is at least `VISUAL_BRIGHTNESS` and
    rises as the image gets brighter. The picture in the first table above reads 61 for brightness and 191 for
    darkness.
- - `VISUAL_CONTRAST`
  - The range of the five reported percentiles (0, 25, 50, 75 and 100) divided by their mean: (max − min) / mean. The
    result has no units. The range runs from the darkest pixel to the brightest, so one extreme pixel raises it. It
    describes one image or box, and is not the `CONTEXT` contrast, which compares a box with its surround.
- - `PIXEL_ENTROPY`
  - The Shannon entropy, in nats, of the 256-bin histogram of the pixel values. It describes how the values spread
    across their range. Rearranging the pixels leaves it unchanged, so it does not measure texture.
- - `PIXEL_KURTOSIS`
  - Excess kurtosis, the fourth standardized moment minus 3, so a normal distribution of values reads 0.
- - `DIMENSION_ASPECT_RATIO`
  - 1 − short side / long side of the box. The value is positive for a box wider than it is tall and negative for a
    taller box. A square reads 0, a 2:1 box 0.5 and a 1:2 box −0.5, and every value lies between −1 and 1.
- - `DIMENSION_DISTANCE_EDGE`
  - The smallest of the four absolute gaps between the box and the image edges, in pixels. The value is unsigned, so a
    box that extends past an edge reads as a positive distance, as one inside the image does, and the value does not
    say whether the box lies inside.
- - `CONTEXT_SURROUND_MEAN` and `CONTEXT_SURROUND_STD`
  - The mean and the population standard deviation of the intensity of the pixels around a box: a frame a quarter of the
    box's width and height wide, less the box and less every other valid box. Intensity is the value mapped to [0, 1]
    from the image's interval and averaged over the channels, and a pixel counts only if every channel lies in the
    interval. This reads one box's own neighborhood, where the `background_` prefix below pools the pixels outside every
    box over the whole image, and an object that no box annotates stays in the neighborhood.
- - `CONTEXT_TARGET_SURROUND_CONTRAST` and `CONTEXT_SIGNAL_CLUTTER_RATIO`
  - How far a box's own mean intensity stands from its surround's, relative to the surround's level
    (`target_surround_contrast`) and to its spread (`signal_clutter_ratio`). The sign says whether the box is brighter or
    darker than its surround. A floor under each denominator keeps a dark or flat surround from giving a huge value, and
    the details record where it decided. The {class}`.ImageStats` docstring gives the formulas, and
    [the section below](#the-target-surround-ratios) the limits.
- - Background statistics (the `background_` prefix)
  - The same statistics measured over the pixels outside every annotated box. The boxes are unioned, so the values
    describe the scene. See
    [Image statistics in integrity checks](DataIntegrity.md#image-statistics-in-integrity-checks).

:::

## Choosing between them

Use a visual statistic when the question is about appearance and you want one answer across a dataset that mixes
encodings, such as which images are unusually dark, which are blurry, or whether the test set is shot in different
lighting than the train set. Where the images come from different sensors, a difference may come from the camera as well
as the scene, as described above.

Use a pixel statistic when the question is about the values themselves, such as what fraction of the values are zero or
[saturated](#saturation-at-a-clipping-endpoint), how widely they spread across their range, or whether there are NaNs.
Where the dataset is uniformly encoded, which is common, its raw values are already comparable and nothing extra is
needed.

Where a dataset mixes encodings and you want comparable pixel distributions, pass
`normalize_pixel_values=True`. Check whether `VISUAL_BRIGHTNESS` answers the question
more directly; a normalized `PIXEL_MEAN` is often a less direct route to the same result.

## Saturation at a clipping endpoint

`ImageStats.PIXEL_SATURATION` is a pixel statistic: the share of finite raw values that lie within a tolerance of a clipping
endpoint. It counts channel values as stored, and not pixels or colors, so an RGB image clipped in one channel reads
1/3, and an RGBA image reads at least 1/4 for its opaque alpha channel. Measure the color bands as a named
[group of channels](../notebooks/h2_measure_channel_groups.py) and read that group's column to leave it out. Values above
the endpoint stay in the denominator, are not counted as saturated, and are reported separately.

DataEval does not decode the endpoint from the data. It is the upper bound of the declared `value_range`, then the
maximum of an unsigned integer dtype, and the value is NaN when none applies. A depth decoded from the data depends on
the data's own maximum, so one sensor's images could be read against different endpoints. Twelve-bit data stored in
`uint16`, which the table above decodes as 4095 for scaling, is therefore measured against 65535 and reads 0.0 unless
its range is declared as `(0, 4095)`.

A reading of 0.0 shows that no value sits at the stated endpoint and does not establish that an image is unclipped. On
floating-point data that has been resized or otherwise computed on, a clipped region may not land exactly on the
endpoint, and `saturation_atol` sets a tolerance. {func}`.image_stats` gives the rule order and the details.

## The target-surround ratios

`target_surround_contrast` and `signal_clutter_ratio` compare a box's mean intensity with the pixels around it, as a
difference over a reference level, in the manner of a Weber contrast. The intensity is the encoded value mapped to
[0, 1], so the ratios are not contrasts of physical luminance, which gamma and the sensor's response separate from the
stored values. Four limits decide what a value can support.

- The box is the target. Its mean covers every pixel of the box, including the pixels of overlapping neighbors and of
  background inside a loose box, so a loose box reads nearer its surround than the object does.
- The intensity is averaged over the channels, so a box that differs from its surround in hue and not in brightness
  reads near 0.
- A dark or flat surround puts a small number under the division. Each denominator has a floor, 1/255 by default, so a
  value never exceeds 255 in size at the default, and where the floor was used the details say so: the value is then
  mostly the difference over the floor, and says little of the surround.
- Intensity is `(value - low) / (high - low)`, so when a declared range does not start at 0 the denominator of the
  contrast depends on that low end. Declare the range where images are compared.

A box with fewer than 90% of its pixels valid reads NaN for both ratios, with the surround kept, since a mean over a
small part of a box describes the part and not the box. {class}`.Outliers` with `ImageStats.ALL` and `per_target=True`
judges the two ratios too, and both are signed.

## Blur statistics

The blur statistics, `blur` and `blur_directionality`, measure the width of edges in pixels. They are experimental ⚠️.
`ImageStats.ALL` includes them, and `ImageStats.VISUAL` and the default of {func}`.image_stats` do not.
[Blur statistics](BlurStatistics.md) describes them.

## Transforms and the statistics they invalidate

A transform can be placed in the view or in the extractor, and choosing the wrong placement fails silently. Model
preprocessing placed in the view is seen by every evaluator: {class}`.Outliers` then reports brightness and contrast on
ImageNet-normalized tensors, which do not describe the data. Dataset-defining manipulation placed in the extractor is
seen only by the embeddings: if 512 × 512 tiles are what ships, the statistics still describe source-resolution imagery,
which does not ship.

Two questions place a transform. Does an evaluator that never touches your model need to see it? If so, it belongs in
the {class}`.View`, and if not, in the extractor's `transforms=`. Is it deterministic, with an unambiguous rewrite of
the targets? If so, use a curated operation, and if not, {class}`.TorchvisionTransform`.

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
the blur statistics, which are edge widths in pixels. It also changes the spatial statistics, because the boxes are
clipped to the new canvas and those left outside it are dropped, the surround statistics, because the ring around a box
is made of interpolated pixels, and the saturation statistic, because a resize returns floats, which have no dtype
endpoint. It leaves the other pixel statistics, the channel count and the hashes alone, so
resize-then-hash may be a better near-duplicate check across heterogeneous source resolutions than hashing at source
resolution. The quality evaluators intersect the declaration with the {class}`.ImageStats` you asked for, and where the
two overlap they raise a {class}`~dataeval.exceptions.StatsInvalidatedWarning`. Nothing is blocked. The invalidated
statistics now describe the transform. [Respond to a StatsInvalidatedWarning](../notebooks/h2_place_transforms.py) shows
the three responses.

The declarations are conservative. {class}`.TorchvisionTransform` declares `ImageStats.ALL`, the spatial, surround and
saturation statistics included, because an arbitrary transform can move anything. {class}`.SelectChannels` with `"gray"`
declares the blur, saturation and surround statistics too, because a luminance mix changes which picture is read. Random
transforms are seeded for each datum from the datum's `id`, so the view is stable: the statistics pass, the embedding
pass and the duplicates pass see the same pixels, and iterating twice gives identical bytes. A view built through
{class}`.TorchvisionTransform` may not be reconstructable from its sidecar, because provenance records the `repr` of
each operation. A curated operation has a `repr` you can paste back. A `v2.Compose([...])` has a multi-line `repr`, and
a lambda inside one records nothing recoverable. This is one more reason to prefer the curated operations.

## Related concept pages

- [Data integrity](DataIntegrity.md): the evaluators these statistics feed
- [Blur statistics](BlurStatistics.md): the experimental edge-width statistics

## See this in practice

### How-to guides

- [Measure named groups of channels](../notebooks/h2_measure_channel_groups.py): naming groups of bands so
  that each group is measured separately
- [Respond to a StatsInvalidatedWarning](../notebooks/h2_place_transforms.py): what an operation invalidates
  and the three ways to respond
