# Linter Class

## What is it

The Linter class is a data cleaning class used to identify images that are reducing the quality or integrity of the dataset.
It identifies the images by calculating the statistical outliers of a dataset using various statistical tests applied to each metric from the base [ImageStats](../reference/metrics/imagestats.md) class.

## When to use it

The Linter class should be used anytime someone is working with a new dataset.
Whether you are tasked with trying to find the right dataset to train a model or with verifying someone else's stuff,
the Linter class helps you gain an understanding of the data.

The [Exploratory Data Analysis](../tutorials/EDA_Part1.ipynb) tutorial shows how the Linter class can be used in conjunction with other DataEval classes to explore and clean a dataset. 

## Theory behind it

The Linter class heavily relies upon the ImageStats class to create a data distribution for specific metrics.
The Linter then analyzes the distribution through a statistical test.

There are 3 different statistical tests that the Linter class can use for detecting abnormal images:

- zscore,
- modzscore, and
- iqr.

The [z score](https://en.wikipedia.org/wiki/Standard_score) method is based on determining if the distance between a data point and the dataset mean is above a specified threshold.  
The default threshold value for `zscore` is 3. The equation for `zscore` is:  
Z score $= |x_i - \mu| / \sigma$

The [modified z score](https://www.statology.org/modified-z-score/) method is based on determining if the distance between a data point and the dataset median is above a specified threshold.  
The default threshold value for `modzscore` is 3.5. The equation for `modzscore` is:  
Modified z score $= 0.6745 * |x_i - x̃| / MAD$, where [$MAD$](https://en.wikipedia.org/wiki/Median_absolute_deviation) is the median absolute deviation

The [interquartile range](https://en.wikipedia.org/wiki/Interquartile_range) method is based on determining if the distance between a data point and the 75th quartile or a data point and the 25th quartile is greater than the difference between the 75th and 25th quartile multiplied by a specified threshold.  
The default threshold value for `iqr` is 1.5. The equation for `iqr` is:  
Interquartile range $= distance > threshold * (Q_3 - Q_1)$, where distance is the greater of $25th quartile - value$ or $value - 75th quartile$

The [Linter](../reference/detectors/linter.md) API will give more information on how to use the functionality.
The user has the option to specify:
- the metrics that the ImageStats class should create a distribution for by selecting which flags the Linter consumes,
- the statistical method used by the class, and
- the threshold the Linter should compare with for identifying outliers.

The Linter class outputs a results dictionary with the keys being the images which were extreme in at least one metric as measured by the chosen statistical test.
The value for each image in the results dictionary is a dictionary with the metric and the image's value in the given metric as the key-value pair.

For more information on the flags and metrics used, see the [list](DataCleaning.md#data-cleaning-metrics).
If you are having trouble changing the flags/metrics, check out our [Linting Flags](../how_to/linting_flags.md) how-to.