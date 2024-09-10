# Image Statistics Functions

## imagestats

### What is the imagestats function

The imagestats function is an aggregate metric that calculates various values for each individual image for the selected metrics.  

The `imagestats` function assists with understanding the dataset.
It can be used to get a big picture view of the dataset and it's underlying distribution.

The stats delivered by the class is broken down into 3 main categories:
* statistics covering image properties,
* statistics covering the visual aspect of images,
* and normal statistics about pixel values.

The available metrics are defined in the [ImageStat](../reference/flags/imagestat.md) flag class.

This function can be used in conjunction with the `Outliers` class to determine if there are any issues with any of the images in the dataset.

This function can be used in conjunction with the `Duplicates` class in order to identify duplicate images.

## channelstats

### What is the channelstats function

The channelstats function is an aggregate metric that calculates various values for each individual image on a per channel basis for the selected metrics.
Unlike the imagestats function, this function only works with the Pixel Statistics subset of the [ImageStat](../reference/flags/imagestat) flag class.

### When to use the channelstats function

This function is best used when you have multiple channels in a dataset and are looking for channelwise differences.
The output from this function cannot currently be used with the Outliers class and must be used on its own.
