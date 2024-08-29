(imagestats_ref)=
# Image Statistics

The `imagestats` function assists with understanding the dataset.
This function can be used in conjunction with the `Linter` class to determine
if there are any issues with any of the images in the dataset.

This class can be used to get a big picture view of the dataset and it's underlying distribution.

The stats delivered by the class is broken down into 3 main categories:
* statistics covering image properties,
* statistics covering the visual aspect of images,
* and normal statistics about pixel values.

Below shows the statistics each category calculates.

* Image Properties
 - height
 - width
 - size
 - aspect ratio
 - number of channels
 - pixel value range

* Image Visuals
 - image brightness
 - image blurriness
 - missing values (NaNs)
 - number of 0 value pixels

* Pixel Statistics
 - mean pixel value
 - pixel value standard deviation
 - pixel value variance
 - pixel value skew
 - pixel value kurtosis
 - entropy of the image
 - pixel percentiles (min, max, 25th, 50th, and 75th percentile values)
 - histogram of pixel values

In addition to the above stats, the `imagestats` function also defines a hash for each image to be used
in conjunction with the `Duplicates` class in order to identify duplicate images.

## Tutorials

To see how the `imagestats` function can be used while doing exploratory data analysis, check out the _EDA Part 1_ tutorial.

{doc}`Exploratory Data Analysis Part 1<../../tutorials/EDA_Part1>`

## How To Guides

There is a how-to guide that applies to the `imagestats` function.

* [How to customize the metrics for data cleaning](../../how_to/linting_flags.md)

## DataEval API

```{eval-rst}
.. autofunction:: dataeval.metrics.imagestats
```
