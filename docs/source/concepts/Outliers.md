# Outliers

## What is it

The Outliers class is a data cleaning class used to identify images that are
reducing the quality or integrity of the dataset. It identifies the images by
calculating the statistical outliers of a dataset using various statistical
tests applied to each metric from the [imagestats](Stats.md) function.

Because the statistical tests rely on a data distribution, the results are
based on the data evaluated. This means that changing the data either through
adding to it or removing some of it, will result in different results.

## When to use it

The Outliers class should be used anytime someone is working with a new
dataset. Whether you are tasked with trying to find the right dataset to train
a model or with verifying someone else's stuff, the Outliers class helps you
gain an understanding of the data.

The [Data Cleaning Guide](../notebooks/tt_clean_dataset.ipynb) shows how
the Outliers class can be used in conjunction with other DataEval classes to
explore and clean a dataset.

## Theory behind it

The Outliers class relies upon the [stat functions](Stats.md) to create a data
distribution for specific metrics. The class then analyzes the distribution
through a statistical test.

There are 3 different statistical tests that the Outliers class can use for
detecting abnormal images:

- zscore
- modzscore
- iqr

The [z score](https://en.wikipedia.org/wiki/Standard_score) method is based on
determining if the distance between a data point and the dataset mean is above
a specified threshold. The default threshold value for `zscore` is 3. The
equation for `zscore` is:

$$
\frac{|x_i - \mu|}{\sigma}
$$

The [modified z score](https://www.statology.org/modified-z-score/) method is
based on determining if the distance between a data point and the dataset
median is above a specified threshold. The default threshold value for
`modzscore` is 3.5. The equation for `modzscore` is:

$$
\frac{0.6745 * |x_i - x̃|}{MAD}
$$

where [$MAD$](https://en.wikipedia.org/wiki/Median_absolute_deviation) is the
median absolute deviation.

The [interquartile range](https://en.wikipedia.org/wiki/Interquartile_range)
method is based on determining if the distance between a data point and the
75th quartile or a data point and the 25th quartile is greater than the
difference between the 75th and 25th quartile multiplied by a specified
threshold. The default threshold value for `iqr` is 1.5. The equation for `iqr`
is:

$$
distance > threshold * (Q_3 - Q_1)
$$

where distance is the greater of $25th\ quartile - value$ or
$value - 75th\ quartile$

## Class Options

The {class}`.Outliers` API will give more information on how to use the
functionality. The user has the option to specify:

- the statistical method used by the class
- the threshold the Outliers should compare with for identifying outliers

For more information on the algorithms/methods used to create the data
distribution, see the [Stats](Stats.md) concept page.

## Class Output

The Outliers class outputs a results dictionary with the keys being the images
which were extreme in at least one metric as measured by the chosen statistical
test. The value for each image in the results dictionary is a dictionary with
the metric and the image's value in the given metric as the key-value pair.

The Outliers class does not identify the reason that the image is flagged.
The functionality of mapping issues to specific aspects of the image is outside
the scope of the Outliers class.
