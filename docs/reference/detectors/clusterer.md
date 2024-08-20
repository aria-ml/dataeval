(clusterer-ref)=
# Clusterer

As part of data exploration, we often want to know how the data groups.
The Clusterer class uses hierarchal clustering to group the data and flags duplicate images as well as outlier images.

The Clusterer identifies both exact duplicate and near duplicate images based on their distance.
Near duplicate images are defined as images whose distance is within the standard deviation of the cluster to which they belong.
By being based on their respective cluster, near duplicates accounts for differences in the density of the cluster.

The Clusterer identifies outliers based on their distance.
After defining where the splits are in the data for the different groups,
outliers are defined as samples that lie outside of 2 standard deviations of the average intra-cluster distance.

## How-To Guides

Check out this **how to** to begin using the `Clusterer` class

{doc}`Clusterer Tutorial<../../how_to/notebooks/ClustererTutorial>`

## Tutorials

There is also an _Exploratory Data Analysis_ tutorial which shows how to use the `Clusterer`
in conjunction with several other data analysis classes from DataEval.

{doc}`Exploratory Data Analysis<../../tutorials/EDA_Part1>`

## DataEval API

```{eval-rst}
.. autoclass:: dataeval.detectors.Clusterer
   :members:
   :inherited-members:
```
