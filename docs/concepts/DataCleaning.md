# Cleaning Datasets
% Includes the Stats classes, the Linter class and the Duplicates class

## What is it
### Linter

Linting identifies potential issues (lints) in the data of the type that are typically
identified and removed during the manual process of cleaning the data.
The `Linter` class can be used to verify that the data have been cleaned or to help automate the data-cleaning process.

Currently, the `Linter` class identifies issues in two major categories, _image properties_ and _visual quality_.
Issues detected in the visual quality category include the brightness and blurriness of an image
as well as images with missing data or high number of zeros (usually really dark images) in the image.
Issues detected in the image properties category include the width, height, depth and value range of the images.
Being able to identify and remove images that lie in the extremes in each of these categories
can help ensure that you have high quality data for training.

## When to use it

The Linting class should be used during the initial EDA process or if you are trying to verify that you have the right data in your dataset.

The Duplicates class should be used if you need to check for duplicates in your dataset.

## Theory behind it

There are 3 different methods that the Linter class can use for detecting abnormal images.

- zscore
- modzscore
- iqr

The default value used for the Linter class is `modzscore`.

The [z score](https://en.wikipedia.org/wiki/Standard_score) method is based on the difference between the data point and the mean of the data.  
The default threshold value for `zscore` is 3.  
Z score $= |x_i - \mu| / \sigma$

The [modified z score](https://www.statology.org/modified-z-score/) method is based on the difference between the data point and the median of the data.  
The default threshold value for `modzscore` is 3.5.  
Modified z score $= 0.6745 * |x_i - x̃| / MAD$, where [$MAD$](https://en.wikipedia.org/wiki/Median_absolute_deviation) is the median absolute deviation

The [interquartile range](https://en.wikipedia.org/wiki/Interquartile_range) method is based on the difference between the data point and the difference between the 75th and 25th qartile.  
The default threshold value for `iqr` is 1.5.  
Interquartile range $= threshold * (Q_3 - Q_1)$
