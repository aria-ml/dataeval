# Image Statistics Functions

The image statistics functions assist with understanding the dataset.
It can be used to get a big picture view of the dataset and it's underlying distribution.

## dimensionstats

### What is the dimensionstats function

The dimensionstats function is an aggregate metric that calculates various dimension based statistics for each individual image:
- width
- height
- channels
- size
- aspect_ratio
- depth

This function can be used in conjunction with the `Outliers` class to determine if there are any issues with any of the images in the dataset.

## hashstats

### What is the hashstats function

The hashstats function is an aggregate metric that calculates various hash values for each individual image:
- [xxhash](https://github.com/Cyan4973/xxHash) - exact image matching
- [pchash](https://en.wikipedia.org/wiki/Perceptual_hashing) - perceptual hash based near image matching

This function can be used in conjunction with the `Duplicates` class in order to identify duplicate images.

## pixelstats

### What is the pixelstats function

The pixelstats function is an aggregate metric that calculates normal statistics about pixel values for each individual image:
- mean
- std
- var
- skew
- kurtosis
- entropy
- percentiles
- histogram

This function can be used in conjunction with the `Outliers` class to determine if there are any issues with any of the images in the dataset.

## visualstats

### What is the visualstats function

The visualstats function is an aggregate metric that calculates visual quality statistics for each individual image:
- brightness
- blurriness
- contrast
- darkness
- missing (as a percentage of total pixels)
- zeros (as a percentage of total pixels)
