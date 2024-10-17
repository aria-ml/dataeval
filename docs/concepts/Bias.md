# Detecting Bias in Datasets
% Includes Parity, Coverage, Balance

## What is it

## When to use it

```{currentmodule} dataeval.metrics.bias
```

The {func}`coverage` function should be used when you have lots of images, but only a small fraction from certain regimes/labels.

The {func}`parity` function and similar should be used when you would like to determine if two datasets have statistically independent labels.

## Theory behind it