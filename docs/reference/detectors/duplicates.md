(duplicates-ref)=

# Duplicates Detection

The duplicate detector helps prune out exact and near matches.
Exact matches are found using a byte hash of the image information,
while near matches (such as a crop of another image or a distoration of another image) use a perception based hash.

## How-To Guides

Check out this **how to** to begin using the `Duplicates` class

{doc}`Linting Tutorial<../../how_to/notebooks/LintingTutorial>`

## Tutorials

There is also an _Exploratory Data Analysis_ tutorial which shows how to use the `Duplicates` in conjunction with several other data analysis classes from DataEval.

{doc}`Exploratory Data Analysis<../../tutorials/EDA_Part1>`

## DataEval API

```{eval-rst}
.. autoclass:: dataeval.detectors.Duplicates
   :members:
   :inherited-members:
```
