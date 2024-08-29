(statisticsflag_ref)=
# ImageStat Flags

Each category of flags contains a specific set of image metrics that can be used with the `imagestats` and `channelstats` functions, and the `Linter` class.
The how-to [How to customize the metrics for data cleaning](../../how_to/linting_flags.md) shows how to customize the metrics from a flag category.

In addition to the below values, supported group categories are:
* `ALL_HASHES` : `XXHASH | PCHASH`
* `ALL_PROPERTIES` : `WIDTH | HEIGHT | SIZE | ASPECT_RATIO | CHANNELS | DEPTH`
* `ALL_VISUALS` : `BRIGHTNESS | BLURRINESS | MISSING | ZERO`
* `ALL_STATISTICS` : `MEAN | STD | VAR | SKEW | KURTOSIS | ENTROPY | PERCENTILES | HISTOGRAM`
* `ALL` : All flags

```{eval-rst}
.. autoflag:: dataeval.flags.ImageStat
```
