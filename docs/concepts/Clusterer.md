# Clustering

## What is it

As part of data exploration, we often want to know how the data groups.
The Clusterer class uses hierarchal clustering to group the data and flags duplicate images as well as outlier images.

The Clusterer identifies both exact duplicate and near duplicate images based on their distance.
Near duplicate images are defined as images whose distance is within the standard deviation of the cluster to which they belong.
By being based on their respective cluster, near duplicates accounts for differences in the density of the cluster.

The Clusterer identifies outliers based on their distance.
After defining where the splits are in the data for the different groups,
outliers are defined as samples that lie outside of 2 standard deviations of the average intra-cluster distance.


## When to use it

The Clusterer can be used during the EDA process to perform the following:

- group a dataset into clusters
- verify labeling as a quality control
- identify outliers in your dataset
- identify duplicates in your dataset

## Theory behind it