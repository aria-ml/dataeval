(channelstats_ref)=
# Channel Statistics

The `channelstats` function is similar to the `imagestats` function except that it only calculates
the normal statistics for each pixel value on a per channel basis.

* Pixel Statistics (Per Channel)
 - mean pixel value (float16)
 - pixel value standard deviation (float16)
 - pixel value variance (float16)
 - pixel value skew (float16)
 - pixel value kurtosis (float16)
 - entropy of the image (float16)
 - pixel percentiles (min, max, 25th, 50th, and 75th percentile values) (float16)
 - histogram of pixel values (uint32)

## Tutorials

To see how the `channelstats` function can be used while doing exploratory data analysis, check out the _EDA Part 1_ tutorial.

{doc}`Exploratory Data Analysis Part 1<../../tutorials/EDA_Part1>`

## How To Guides

There is a how-to guide that applies to the `channelstats` function.

* [How to customize the metrics for data cleaning](../../how_to/linting_flags.md)

## DataEval API

```{eval-rst}
.. autofunction:: dataeval.metrics.channelstats
```