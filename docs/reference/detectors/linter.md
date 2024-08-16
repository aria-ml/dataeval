(linter-ref)=
# Linter

Linting identifies potential issues (lints) in the data of the type that are typically
identified and removed during the manual process of cleaning the data.
The `Linter` class can be used to verify that the data have been cleaned or to help automate the data-cleaning process.

Currently, the `Linter` class identifies issues in two major categories, _image properties_ and _visual quality_.
Issues detected in the visual quality category include the brightness and blurriness of an image
as well as images with missing data or high number of zeros (usually really dark images) in the image.
Issues detected in the image properties category include the width, height, depth and value range of the images.
Being able to identify and remove images that lie in the extremes in each of these categories
can help ensure that you have high quality data for training.

## Tutorials
To see how the Linter class can be used while doing exploratory data analysis, check out the _EDA Part 1_ tutorial.

{doc}`Exploratory Data Analysis Part 1<../../tutorials/EDA_Part1>`

## How-To Guides

There are a couple how-to guides for the `Linter` class.

* {doc}`Linting How-To Guide<../../how_to/notebooks/LintingTutorial>`
* [How to customize the metrics for data cleaning](../../how_to/linting_flags.md)


## DataEval API

```{eval-rst}
.. autoclass:: dataeval.detectors.Linter
   :members:
   :inherited-members:
```
