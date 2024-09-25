# How to customize the metrics for data cleaning

There are 4 categories of metrics for data cleaning which are available in the [ImageStat](../reference/flags/dataeval.flags.ImageStat.rst) flag class.

* Image Hashing (`ALL_HASHES`)
    * `XXHASH`
    * `PCHASH`
  
* Image Properties (`ALL_PROPERTIES`)
    * `WIDTH`
    * `HEIGHT`
    * `SIZE`
    * `ASPECT_RATIO`
    * `CHANNELS`
    * `DEPTH`

* Image Visuals (`ALL_VISUALS`)
    * `BRIGHTNESS`
    * `BLURRINESS`
    * `MISSING`
    * `ZERO`

* Pixel Statistics (`ALL_PIXELSTATS`)
    * `MEAN`
    * `STD`
    * `VAR`
    * `SKEW`
    * `KURTOSIS`
    * `ENTROPY`
    * `PERCENTILES`
    * `HISTOGRAM`

To select a custom set of metrics, load in the category:

```python
from dataeval.metrics.stats import ImageStat
```

Then select the desired metrics and pass them to the desired function or class.

`imagestats` function example:

```python
# Select the desired data cleaning metrics
flags = ImageStat.SIZE | ImageStat.MEAN

# Compute the stats for the dataset
result = imagestats(dataset, flags=flags)
```

`channelstats` function example:

```python
# Select the desired data cleaning metrics
flags = ImageStat.MEAN | ImageStat.STD | ImageStat.ENTROPY

# Compute the stats for the dataset
result = channelstats(dataset, flags=flags)
```

`Outliers` class example:

```python
# Select the desired data cleaning metrics
flags = ImageStat.ALL_VISUALS

# Set the flags for the class
outliers = Outliers(dataset, flags=flags)
# Evaluate the dataset
results = outliers.evaluate()
```