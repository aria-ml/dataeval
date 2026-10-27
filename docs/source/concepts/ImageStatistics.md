# Image Statistics

`image_stats` measures four families of statistics about an image (pixel, visual,
dimension, and hash), and two of them, pixel and visual, answer fundamentally
different kinds of questions. Knowing which is which is the difference between a
number you can compare across a dataset and one you cannot.

## Raw values and visual interpretation

A **pixel statistic** reduces the values as they were stored. `PIXEL_MEAN` on a
16-bit image is a mean of numbers that run to 65535, because that is what the file
contains. It describes the data.

A **visual statistic** stands in for how the image looks to a person.
`VISUAL_BRIGHTNESS` is a claim about appearance, and appearance is a position between
black and white rather than a value in whatever units the sensor wrote. So a visual
statistic first resolves the image's **full-scale reference** (what counts as white)
and reports against the 0–255 display range.

That single difference is why the same picture, handed over four ways, reads
identically to every visual statistic:

| Encoding | `brightness` | `darkness` | `sharpness` |
| --- | --- | --- | --- |
| `uint8` | 61.0 | 191.0 | 112.11 |
| `uint16` (the same values ×257) | 61.0 | 191.0 | 112.11 |
| `float` in [0, 1] | 61.0 | 191.0 | 112.11 |
| `float` in [0, 255] | 61.0 | 191.0 | 112.11 |

and why its pixel statistics do not:

| Encoding | `mean` | `std` |
| --- | --- | --- |
| `uint8` | 126.88 | 75.42 |
| `uint16` (the same values ×257) | 32607.54 | 19382.91 |

Neither table is wrong. They answer different questions. "Is this image dark?" is a
visual question and has one answer. "What is the mean of these values?" is a question
about the data and has two, because there are two sets of values.

## Where the reference comes from

The reference is not the image's brightest pixel. Reading it off the data would mean
every image is normalized against its own maximum, so a photograph of a night sky and
one of a beach would both report mid-gray, the opposite of comparable.

Instead it is **decoded** where the data carries an encoding, because integer image
formats are power-of-two:

| Input | Reference |
| --- | --- |
| `uint8` | 255 |
| `uint16` holding a 12-bit sensor's output | 4095 |
| `uint16` holding 16-bit output | 65535 |
| `float` within [0, 1] | 1.0 (already normalized) |
| `float` within [0, 255] | 255 (an 8-bit image in a float array) |

Two of those are float conventions rather than encodings, and they are recognized
because they are the two ordinary ways an ordinary image arrives: a `ToTensor`-style
pipeline produces the first, a resize or any interpolation produces the second.

## When there is no reference

Some data has none. Elevation below sea level, mean-centered reflectance, temperature
in Celsius, a 16-bit band holding physical units: the dynamic range of these is a
property of the sensor, not of a file format, and nothing about the array reveals it.
There is no encoding to decode.

For that data, visual statistics report NaN. Not an error: "how bright is this
elevation map" is a question with no answer, and NaN is how a statistic says so. Pixel
statistics (and other statistics that do not rely on an interval) are unaffected,
because the mean of a temperature band is a perfectly good mean.

Declare the interval to get a reading back:

```python
image_stats(elevation_data, stats=ImageStats.VISUAL, value_range=(-2000.0, 2000.0))
```

A declaration says "these are physical values spanning this range." It implies no bit
depth, so `DIMENSION_DEPTH` reports NaN for such data whether or not you declare one.

Pixel statistics that need an interval also answer NaN: `PIXEL_HISTOGRAM`,
`PIXEL_ENTROPY`, and the rest of the statistics under `normalize_pixel_values=True`,
which is a request to divide by a range that does not exist.

When this happens, a warning names `value_range`, because an all-NaN column is easy to miss.

## Which statistics care about scale at all

Fewer than you might expect. Of the pixel family, only three change when the values
are rescaled:

| Changes with scale | Already scale-free |
| --- | --- |
| `mean`, `std`, `var` | `skew`, `kurtosis` (standardized moments) |
| | `entropy`, `histogram` (binned over the reference either way) |
| | `missing`, `zeros` (counts) |

`normalize_pixel_values=True` exists for those three, and only for them. It divides by
the same reference the visual family uses, which makes a pixel distribution comparable
across bit depths.

It does not touch visual statistics, which already are scaled.

## Choosing between them

Reach for a visual statistic when the question is about appearance and you want one
answer across a mixed dataset, such as which images are unusually dark, which are
blurry, or whether the test set is shot in different lighting than the train set.

Reach for a pixel statistic when the question is about the values themselves, such as
what fraction of this band is saturated, how much information this image carries, or
whether there are NaNs. Where the dataset is uniformly encoded, which is the usual
case, its raw values are already comparable and nothing extra is needed.

Where a dataset mixes encodings and you want comparable pixel distributions, pass
`normalize_pixel_values=True`. But consider whether `VISUAL_BRIGHTNESS` is the
statistic you actually wanted; a normalized `PIXEL_MEAN` is often a less direct route
to the same question.

## Transforms and the statistics they invalidate

A transform has two possible homes, and choosing the wrong one fails silently. If you put model preprocessing in the
view, every evaluator sees it: {class}`.Outliers` reports brightness and contrast on ImageNet-normalized tensors, which
are not numbers about your data. If you put dataset-defining manipulation in the extractor, only the embeddings see it:
if 512x512 tiles are what ships, your statistics still describe source-resolution imagery that does not.

Two questions place a transform. Does an evaluator that never touches your model need to see it? If so, it belongs in
the {class}`.View`, and if not, in the extractor's `transforms=`. Is it deterministic, with an unambiguous rewrite of
the targets? If so, use a curated operation, and if not, {class}`.TorchvisionTransform`.

| Kind | Example | Home |
| --- | --- | --- |
| Model-input adaptation | resize-to-224, ImageNet normalize, dtype cast | the extractor's `transforms=` |
| Dataset definition | crop out a HUD, downsample to the deployed resolution, band selection | {class}`.Resize`, {class}`.Crop` and {class}`.SelectChannels` in a view |
| Augmentation or corruption | random flip, color jitter, blur | {class}`.TorchvisionTransform` |

Use {class}`.Resize` for the resolution your system deploys at, because that is a fact about the data under evaluation
and not about the model. Use {class}`.Crop` to remove a fixed region that is never evidence, such as a burned-in
overlay, a scan border or a dead sensor region. Use {class}`.SelectChannels` to narrow or combine channels, for a mono
sensor stored as RGB or for the bands of a multispectral cube that you evaluate.

Every operation declares the statistics it invalidates. A resize makes the dimension statistics, such as `width`,
`height` and `aspect_ratio`, report the resize target, and makes `sharpness` measure the interpolation kernel. It
leaves the pixel statistics, the channel count and the hashes alone, so resize-then-hash is a better near-duplicate
check across heterogeneous source resolutions, not a worse one. The quality evaluators intersect the declaration with
the {class}`.ImageStats` you asked for, and where the two overlap they raise a
{class}`~dataeval.exceptions.StatsInvalidatedWarning`. Nothing is blocked. The invalidated statistics now describe the
transform. [How to respond to a StatsInvalidatedWarning](../notebooks/h2_place_transforms.py) shows the three
responses.

The declarations are conservative. {class}`.TorchvisionTransform` declares `ImageStats.ALL`, because an arbitrary
transform can move anything. Random transforms are seeded for each datum from the datum's `id`, so the view is stable:
the statistics pass, the embedding pass and the duplicates pass see the same pixels, and iterating twice gives
identical bytes. A view built through {class}`.TorchvisionTransform` may not be reconstructable from its sidecar,
because provenance records the `repr` of each operation. A curated operation has a `repr` you can paste back. A
`v2.Compose([...])` has a multi-line `repr`, and a lambda inside one records nothing recoverable. That is the last
reason to prefer the curated operations.

## See also

- {doc}`DataIntegrity` — the evaluators these statistics feed
- {func}`~dataeval.core.image_stats` — the flags and their parameters
- {class}`~dataeval.flags.ImageStats` — the full statistic list

## See this in practice

### How-to guides

- [How to measure named groups of channels](../notebooks/h2_measure_channel_groups.py) — naming groups of bands so
  that each group is measured separately.
- [How to respond to a StatsInvalidatedWarning](../notebooks/h2_place_transforms.py) — what an operation invalidates
  and the three ways to respond.
