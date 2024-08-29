# Image Statistics Functions

## imagestats

### What is the imagestats function

The imagestats function is an aggregate metric that calculates various values for each individual image for the selected metrics.  

The available metrics are defined by their respective flag class:
* [ImageHash](../reference/flags/imagehash.md)
* [ImageProperty](../reference/flags/imageproperty.md)
* [ImageStatistics](../reference/flags/imagestatistics.md)
* [ImageVisuals](../reference/flags/imagevisuals.md)

## channelstats

### What is the channelstats function

The channelstats function is an aggregate metric that calculates various values for each individual image on a per channel basis for the selected metrics.
Unlike the imagestats function, this function only works with the [ImageStatistics](../reference/flags/imagestatistics.md) flags.

### When to use the channelstats function

This function is best used when you have multiple channels in a dataset and are looking for channelwise differences.
The output from this function cannot currently be used with the Linter class and must be used on its own.
