# Statistical Analysis

The image statistics functions assist with understanding the dataset.
It can be used to get a big picture view of the dataset and it's underlying
distribution.

## dimensionstats

The dimensionstats function is an aggregate metric that calculates various
dimension based statistics for each individual image:

- width
- height
- channels
- size
- aspect_ratio
- depth

This function can be used in conjunction with the Outliers class to determine
if there are any issues with any of the images in the dataset.

## hashstats

The hashstats function is an aggregate metric that calculates various hash
values for each individual image:

- [xxhash](https://github.com/Cyan4973/xxHash) - exact image matching
- [pchash](https://en.wikipedia.org/wiki/Perceptual_hashing) - perceptual hash
  based near image matching

This function can be used in conjunction with the
{term}`duplicates<Duplicates>` class in order to identify duplicate images.

## labelstats

The labelstats function provides summary statistics across classes and labels:

- label_counts_per_class
- label_counts_per_image
- image_counts_per_label
- image_indices_per_label
- image_count
- label_count
- class_count

## pixelstats

The pixelstats function is an aggregate metric that calculates normal
statistics about pixel values for each individual image:

- mean
- std
- var
- skew
- kurtosis
- entropy
- percentiles
- histogram

This function can be used in conjunction with the Outliers class to determine
if there are any issues with any of the images in the dataset.

## visualstats

The visualstats function is an aggregate metric that calculates visual quality
statistics for each individual image:

- brightness
- sharpness
- contrast
- darkness
- missing (as a percentage of total pixels)
- zeros (as a percentage of total pixels)
