# Image Statistics

`compute_stats` measures four families of statistics about an image (pixel, visual,
dimension, and hash), and two of them—pixel and visual—answer fundamentally different
kinds of questions. Knowing which is which is the difference between a number you can
compare across a dataset and one you cannot.

## Raw values and visual interpretation

A **pixel statistic** reduces the values as they were stored. `PIXEL_MEAN` on a
16-bit image is a mean of numbers that run to 65535, because that is what the file
contains. It describes the data.

A **visual statistic** stands in for how the image looks to a person.
`VISUAL_BRIGHTNESS` is a claim about appearance, and appearance is a position
between black and white rather than a value in whatever units the sensor wrote. So
a visual statistic first resolves the image's **full-scale reference** — what
counts as white — and reports against the 0–255 display range.

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

Neither table is wrong. They answer different questions. *Is this image dark?* is
a visual question and has one answer. *What is the mean of these values?* is a
question about the data and has two, because there are two sets of values.

## Where the reference comes from

The reference is not the image's brightest pixel. Reading it off the data would
mean every image is normalized against its own maximum, so a photograph of a night
sky and one of a beach would both report mid-grey — the opposite of comparable.

Instead it is **decoded** where the data carries an encoding, because integer image
formats genuinely are power-of-two:

| Input | Reference |
| --- | --- |
| `uint8` | 255 |
| `uint16` holding a 12-bit sensor's output | 4095 |
| `uint16` holding 16-bit output | 65535 |
| `float` within [0, 1] | 1.0 — already normalized |
| `float` within [0, 255] | 255 — an 8-bit image in a float array |

Two of those are float conventions rather than encodings, and they are recognized
because they are the two ordinary ways an ordinary image arrives: a `ToTensor`-style
pipeline produces the first, a resize or any interpolation produces the second.

## When there is no reference

Some data has none. Elevation below sea level, mean-centred reflectance, temperature
in Celsius, a 16-bit band holding physical units — the dynamic range of these is a
property of the *sensor*, not of a file format, and nothing about the array reveals
it. There is no encoding to decode.

For that data, **visual statistics report NaN**. Not an error: *how bright is this
elevation map* is a question with no answer, and NaN is how a statistic says so.
Pixel statistics are unaffected, because the mean of a temperature band is a
perfectly good mean.

Declare the interval to get a reading back:

```python
compute_stats(elevation_data, stats=ImageStats.VISUAL, value_range=(-2000.0, 2000.0))
```

A declaration says *these are physical values spanning this range*. It implies no bit
depth, so `DIMENSION_DEPTH` reports NaN for such data whether or not you declare one.

Pixel statistics that need an interval answer NaN too: `PIXEL_HISTOGRAM` and
`PIXEL_ENTROPY` always, and the rest of the family under
`normalize_pixel_values=True`, which is a request to divide by a range that does not
exist. A warning names `value_range` when this happens, because an all-NaN column is
easy to miss.

Statistics that need no interval are unaffected. An unnormalized mean of temperature
readings is a perfectly good mean, and reporting it is the whole point of keeping the
pixel family in native units.

## Which statistics care about scale at all

Fewer than you might expect. Of the pixel family, only three change when the values
are rescaled:

| Changes with scale | Already scale-free |
| --- | --- |
| `mean`, `std`, `var` | `skew`, `kurtosis` — standardized moments |
| | `entropy`, `histogram` — binned over the reference either way |
| | `missing`, `zeros` — counts |

`normalize_pixel_values=True` exists for those three, and only for them. It divides
by the same reference the visual family uses, which makes a pixel *distribution*
comparable across bit depths.

It does not touch visual statistics, which already are.

## Choosing between them

Reach for a **visual** statistic when the question is about appearance and you want
one answer across a mixed dataset — *which images are unusually dark*, *which are
blurry*, *is the test set shot in different lighting than the train set*.

Reach for a **pixel** statistic when the question is about the values themselves —
*what fraction of this band is saturated*, *how much information does this image
carry*, *are there NaNs*. Where the dataset is uniformly encoded, which is the usual
case, its raw values are already comparable and nothing extra is needed.

Where a dataset genuinely mixes encodings and you want comparable pixel
distributions, pass `normalize_pixel_values=True`. But consider whether
`VISUAL_BRIGHTNESS` is the statistic you actually wanted; a normalized `PIXEL_MEAN`
is often a less direct route to the same question.

## See also

- {doc}`DataIntegrity` — the evaluators these statistics feed
- {func}`~dataeval.core.compute_stats` — the flags and their parameters
- {class}`~dataeval.flags.ImageStats` — the full statistic list
