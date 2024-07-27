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

Check out this tutorial to begin using the `Linting` class

{doc}`Linting Tutorial<../../tutorials/notebooks/LintingTutorial>`

## How To Guides

There is also an _Exploratory Data Analysis_ guide which shows how to use the `Linter`
in conjunction with several other data analysis classes from DataEval.

{doc}`Exploratory Data Analysis<../../how_to/EDA>`

## DataEval API

```{eval-rst}
.. autoclass:: dataeval.detectors.Linter
   :members:
   :inherited-members:
```
