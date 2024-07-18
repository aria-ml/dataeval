(clusterer-ref)=

# Clustering, Deduplication and Outlier Detection

As part of data exploration, we often want to know how the data groups.
The Clusterer class uses hierarchal clustering to group the data and flags duplicate images as well as outlier images.

The Clusterer identifies both exact duplicate and near duplicate images based on their distance.
Near duplicate images are defined as images whose distance is within the standard deviation of the cluster to which they belong. By being based on their respective cluster, near duplicates accounts for differences in the density of the cluster.

The Clusterer identifies outliers based on their distance. After defining where the splits are in the data for the different groups, outliers are defined as samples that lie outside of 2 standard deviations of the average intra-cluster distance.

## Tutorials

Check out this tutorial to begin using the `Clusterer` class

{doc}`Clusterer Tutorial<../../tutorials/notebooks/ClustererTutorial>`

## How To Guides

There are currently no how to's for the Clusterer.
If there are scenarios that you want us to explain, contact us!

## DataEval API

```{eval-rst}
.. autoclass:: dataeval.detectors.Clusterer
   :members:
   :inherited-members:
```
