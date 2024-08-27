# ImageStats and ChannelStats Classes

## ImageStats

### What is the ImageStats Class

The ImageStats class is a base class that holds all of the values for each individual image for the selected metrics.  
The class composes of a reset, update and compute function which, respectively, allows one to clear the stored values from all metrics,
run each metric on each new data point, and then aggregate all of the metrics from each individual data point into a single set.  

The available metrics are defined by their respective flag class:
* [ImageHash](../reference/flags/imagehash.md)
* [ImageProperty](../reference/flags/imageproperty.md)
* [ImageStatistics](../reference/flags/imagestatistics.md)
* [ImageVisuals](../reference/flags/imagevisuals.md)

### When to use the ImageStats Class

This class is best used when you have a large dataset which cannot all be contained in memory or 
when you are planning on using both the Linter and Duplicates classes on the same dataset.

If your whole dataset fits into memory, then you should just use the Linter class and/or 
the Duplicates class as using the base class does not provide any additional speed up.
<!-- !!TODO: Uncomment once the improvement goes through!!!!!
If you are only going to use at either the Linter class or the Duplicates class,
the just use the respective class as both classes can process batches.-->

## ChannelStats

### What is the ChannelStats Class

The ChannelStats class is a base class that holds all of the values for each individual image on a per channel basis for the selected metrics.
Like the ImageStats class, the ChannelStats class is composed of a reset, update and compute function.
However, the ChannelStats class only works with the [ImageStatistics](../reference/flags/imagestatistics.md) and [ImageVisuals](../reference/flags/imagevisuals.md) flags.

### When to use the ChannelStats Class

This class is best used when you have multiple channels in a dataset and are looking for channelwise differences.
This class cannot currently be used with the Linter class and must be used on its own.
