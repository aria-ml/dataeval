# How to customize the metrics for data cleaning

There are 4 categories of metrics for data cleaning which are available on the [ImageStat](../reference/flags/imagestat.md) class.

* Image Hashing
  - `XXHASH`
  - `PCHASH`
  
* Image Properties
  - `WIDTH`
  - `HEIGHT`
  - `SIZE`
  - `ASPECT_RATIO`
  - `CHANNELS`
  - `DEPTH`

* Image Statistics
  - `MEAN`
  - `STD`
  - `VAR`
  - `SKEW`
  - `KURTOSIS`
  - `ENTROPY`
  - `PERCENTILES`
  - `HISTOGRAM`

* Image Visuals
  - `BRIGHTNESS`
  - `BLURRINESS`
  - `MISSING`
  - `ZERO`

Additionally there are flag groups which are convenient groupings of all metrics in a category:

* `ALL_HASHES`
* `ALL_PROPERTIES`
* `ALL_STATISTICS`
* `ALL_VISUALS`

To select a custom set of metrics, load in the category:

```python
from dataeval.metrics import ImageStat
```

Then select the desired metrics and pass them to the desired class.

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

`Linter` class example:

```python
# Select the desired data cleaning metrics
flags = ImageStat.BRIGHTNESS

# Set the flags for the class
lints = Linter(dataset, flags=flags)
# Evaluate the dataset
results = lints.evaluate()
```