# Clustering

## What is it

As part of data exploration, we often want to know how the data groups.
The Clusterer class uses hierarchal clustering to group the data and flags
duplicate images as well as outlier images.

The Clusterer identifies both exact duplicate and near duplicate images
based on their distance. Near duplicate images are defined as images whose
distance is within the standard deviation of the cluster to which they belong.
By being based on their respective cluster, near {term}`duplicates<Duplicates>`
accounts for differences in the density of the cluster.

The Clusterer identifies Outliers based on their distance. After defining where
the splits are in the data for the different groups, outliers are defined as
samples that lie outside of 2 standard deviations of the average intra-cluster
distance.

## When to use it

The Clusterer can be used during the EDA process to perform the following:

- group a dataset into clusters
- verify labeling as a quality control
- identify outliers in your dataset
- identify duplicates in your dataset

## Class Output

The Clusterer class does not identify the reason that the image is an
outlier. The functionality of mapping issues to specific aspects of the
image is outside the scope of the Clusterer class.
