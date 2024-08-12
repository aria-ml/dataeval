# How to customize the metrics for data cleaning

There are 4 categories of metrics for data cleaning:

* ImageHash
  - XXHASH
  - PCHASH
  
* ImageProperties
  - WIDTH
  - HEIGHT
  - SIZE
  - ASPECT_RATIO
  - CHANNELS
  - DEPTH

* ImageStatistics
  - MEAN
  - STD
  - VAR
  - SKEW
  - KURTOSIS
  - ENTROPY
  - PERCENTILES
  - HISTOGRAM

* ImageVisuals
  - BRIGHTNESS
  - BLURRINESS
  - MISSING
  - ZERO

To select a custom set of metrics, load in the category:

```python
from dataeval.flags import ImageHash, ImageProperties, ImageStatistics, ImageVisuals
```

Then select the desired metrics and pass them to the desired class.

ImageStats class example:

```python
# Select the desired data cleaning metrics
flags = [ImageProperties.SIZE, ImageStatistics.MEAN]

# Set the flags for the class
stats = ImageStats(flags=flags)
# Add the dataset
stats.update(dataset)
# Compute the stats
result = stats.compute()
```

ChannelStats class example:

```python
# Select the desired data cleaning metrics
flags = [ImageStatistics.MEAN, ImageStatistics.STD, ImageStatistics.ENTROPY]

# Set the flags for the class
stats = ChannelStats(flags=flags)
# Add the dataset
stats.update(dataset)
# Compute the stats
result = stats.compute()
```

Linter class example:

```python
# Select the desired data cleaning metrics
flags = [ImageVisuals.BRIGHTNESS]

# Set the flags for the class
lints = Linter(dataset, flags=flags)
# Evaluate the dataset
results = lints.evaluate()
```