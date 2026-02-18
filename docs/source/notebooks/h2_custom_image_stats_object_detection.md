---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: dataeval
  language: python
  name: python3
---

# How to specify custom statistics on object detection datasets

+++

## Problem statement

When working with object detection datasets, you often need to analyze [image statistics](../concepts/ImageStats.md) at
different granularities:

- Image-level statistics: Properties of entire images
- Box-level statistics: Properties of individual bounding boxes within images

This guide will show you how to use {func}`.calculate` with custom {class}`.ImageStats` flags to capture statistics on
full images and individual bounding boxes.

+++

### When to use

Use this approach when you need fine-grained control over which statistics to compute, especially when:

- Working with object detection datasets with bounding boxes
- Analyzing both full images and cropped regions (boxes)
- Optimizing computation by selecting only relevant statistics

+++

### What you will need

1. An object detection dataset (we'll use SeaDrone from maite-datasets)
1. A Python environment with the following packages installed:
   - `dataeval`
   - `maite-datasets`

+++

## Getting started

First import the required libraries needed to set up the example.

```{code-cell} ipython3
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    %pip install -q dataeval maite-datasets
except Exception:
    pass
```

```{code-cell} ipython3
from maite_datasets.object_detection import SeaDrone

from dataeval.core import calculate
from dataeval.flags import ImageStats
from dataeval.selection import Limit, Select
```

## Load the dataset

Begin by loading an object detection dataset. For this example we are using SeaDrone, an object detection dataset
containing aerial images captured by drones over marine environments.

We'll use a subset of the dataset to keep computation time reasonable.

```{code-cell} ipython3
# Load the SeaDrone dataset
sd_dataset = SeaDrone(root="./data", image_set="val", download=True)

# Limit to first 50 images for demonstration
dataset = Select(sd_dataset, Limit(50))

print(f"Dataset size: {len(dataset)} images")
print(f"Sample image shape: {dataset[0][0].shape}")
print(f"Sample targets (boxes): {len(dataset[0][1].boxes)} boxes in first image")
```

## Statistics on full images only

Let calculate statistics on the full images with a custom set of basic statistics.

The {class}`.ImageStats` enum provides fine-grained control over which statistics to compute.

You can combine flags using the `|` (bitwise OR) operator.

```{code-cell} ipython3
# Calculate custom individual statistics for full images only (per_image=True, per_target=False)
results_image_only = calculate(
    data=dataset,
    stats=ImageStats.PIXEL_MEAN | ImageStats.DIMENSION_ASPECT_RATIO | ImageStats.VISUAL_SHARPNESS,
    per_image=True,
    per_target=False,
)

print(f"Computed statistics: {list(results_image_only['stats'])}")
print(f"\nNumber of results: {len(results_image_only['source_index'])}")
print(f"Total images processed: {results_image_only['image_count']}")
```

### Understanding SourceIndex

The `source_index` field contains {class}`.SourceIndex` objects that track where each statistic came from:

- `item`: The item index in the dataset
- `box`: The bounding box index (None for full images)
- `channel`: The channel index (None when per_channel=False)

```{code-cell} ipython3
# Display first 5 source indices
print("First 5 SourceIndex entries (image-level only):")
for i, src in enumerate(results_image_only["source_index"][:5]):
    print(f"  {i}: item={src.item}, target={src.target}, channel={src.channel}")

print(f"\nAll entries have target=None: {all(src.target is None for src in results_image_only['source_index'])}")
```

## Statistics on bounding boxes only

Now let's compute statistics for just bounding box within the images.

```{code-cell} ipython3
# Calculate basic pixel statistics for targets only (per_image=False, per_target=True)
results_target_only = calculate(
    data=dataset,
    stats=ImageStats.PIXEL_BASIC,
    per_image=False,
    per_target=True,
    per_channel=False,
)

print(f"Computed statistics: {list(results_target_only['stats'])}")
print(f"Number of target-level results: {len(results_target_only['source_index'])}")
print(f"Total targets processed: {sum(results_target_only['object_count'])}")

# Display source indices for targets from first image
print("\nSourceIndex entries for targets in first few images:")
for i, src in enumerate(results_target_only["source_index"][:5]):
    print(f"  {i}: image={src.item}, target={src.target}, channel={src.channel}")
```

## Statistics on both full images and bounding boxes

We can also compute statistics at both levels simultaneously.

```{code-cell} ipython3
# Calculate basic dimension statistics for full images, boxes, and channels (per_image=True, per_target=True)
results_both = calculate(
    data=dataset,
    stats=ImageStats.DIMENSION_BASIC,
    per_image=True,
    per_target=True,
)

print(f"Number of results (images + boxes): {len(results_both['source_index'])}")
print(f"Total images processed: {results_both['image_count']}")
print(f"Total boxes processed: {sum(results_both['object_count'])}")
print(f"Statistics calculated for each image: {list(results_both['stats'])}")

# Separate image-level and box-level results
image_indices = [i for i, src in enumerate(results_both["source_index"]) if src.target is None]
target_indices = [i for i, src in enumerate(results_both["source_index"]) if src.target is not None]

print(f"\nImage-level results: {len(image_indices)}")
print(f"Target-level results: {len(target_indices)}")
```

## Key takeaways

From this analysis, we've learned:

1. **Custom Statistics Selection**: The {class}`.ImageStats` flags allow fine-grained control over which statistics to
   compute, optimizing performance by avoiding unnecessary calculations.

1. **Granular Analysis**: Using `per_image` and `per_target` parameters, we can analyze statistics at different levels:

   - Full images provide context about overall scene properties
   - Bounding boxes reveal properties of individual objects

1. **SourceIndex Tracking**: The {class}`.SourceIndex` objects allow us to precisely track which image, box, and channel
   each statistic corresponds to.

+++

## Conclusion

This notebook demonstrated how to use {func}`.calculate` with custom {class}`.ImageStats` flags to perform flexible,
efficient analysis on object detection datasets.

These techniques are valuable for:

- Dataset quality assessment
- Identifying biases or artifacts
- Understanding object characteristics
- Optimizing preprocessing pipelines
- Detecting outliers or anomalies
