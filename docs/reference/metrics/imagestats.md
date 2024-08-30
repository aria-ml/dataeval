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
    * height (uint16)
    * width (uint16)
    * channels (uint8)
    * size (uint32)
    * aspect ratio (float16)
    * pixel value range (uint8)

* Image Visuals
    * image brightness (float16)
    * image blurriness (float16)
    * missing values (NaNs) (float16)
    * number of 0 value pixels (float16)

* Pixel Statistics
    * mean pixel value (float16)
    * pixel value standard deviation (float16)
    * pixel value variance (float16)
    * pixel value skew (float16)
    * pixel value kurtosis (float16)
    * entropy of the image (float16)
    * pixel percentiles (min, max, 25th, 50th, and 75th percentile values) (float16)
    * histogram of pixel values (uint32)

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
