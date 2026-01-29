# Image statistical analysis

The image statistics features assist with understanding the dataset.
These can be used to get a big picture view of the dataset and its underlying
distribution. The {func}`.calculate` function with {class}`.ImageStats` flags creates
the data distribution that the {class}`.Outliers` class uses to identify outliers.

## What are the statistical analysis categories

DataEval provides four main categories of statistics for analyzing image datasets,
controlled via {class}`.ImageStats` flags:

- `PIXEL` - Pixel-level statistics (mean, std, variance, skewness, kurtosis, entropy, etc.)
- `VISUAL` - Visual quality statistics (brightness, contrast, darkness, sharpness, percentiles)
- `DIMENSION` - Dimension-based statistics (width, height, channels, size, aspect ratio, etc.)
- `HASH` - Hash-based statistics for duplicate detection (xxhash, phash, dhash, and D4 variants)

The information below includes what each category provides and the statistical
metrics that are available in each.

### PIXEL Statistics

The `PIXEL` flag group calculates pixel-level statistics for each image.
These statistics analyze the raw pixel value distribution and can be computed
per-channel using the `per_channel=True` parameter with {func}`.calculate`.

Available individual pixel statistics:

| Flag            | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| PIXEL_MEAN      | Average pixel value across entire image                                  |
| PIXEL_STD       | Standard deviation of pixel values across entire image                   |
| PIXEL_VAR       | Variance of pixel values across entire image                             |
| PIXEL_SKEW      | Skewness - measure of how normally distributed the data is               |
| PIXEL_KURTOSIS  | Kurtosis - measure of how normally distributed the data is               |
| PIXEL_ENTROPY   | Shannon entropy based on the histogram, $-\sum p \log p$                 |
| PIXEL_MISSING   | Total number of pixels missing a value as a percentage of total pixels   |
| PIXEL_ZEROS     | Total number of pixels with a zero value as a percentage of total pixels |
| PIXEL_HISTOGRAM | Scales pixel values between 0-1 and bins into 256 bins                   |

Convenience sub-groups:

- `PIXEL_BASIC` - Mean, std, var
- `PIXEL_DISTRIBUTION` - Skew, kurtosis, entropy, histogram

These statistics can be used in conjunction with the {class}`.Outliers` class to determine
if there are any issues with any of the images in the dataset.

### VISUAL Statistics

The `VISUAL` flag group calculates visual quality statistics for each individual image:

| Flag               | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| VISUAL_BRIGHTNESS  | Brightness measure (25th percentile)                                   |
| VISUAL_SHARPNESS   | Sharpness measure using 3x3 edge filter                                |
| VISUAL_CONTRAST    | Contrast measure (max value - min value) / mean value                  |
| VISUAL_DARKNESS    | Darkness measure (75th percentile)                                     |
| VISUAL_PERCENTILES | The 0, 25, 50, 75, and 100 percentile values of the pixel distribution |

Convenience sub-group:

- `VISUAL_BASIC` - Brightness, contrast, sharpness

These statistics can be used in conjunction with the {class}`.Outliers` class to determine
if there are any issues with any of the images in the dataset.

### DIMENSION Statistics

The `DIMENSION` flag group calculates dimension-based statistics for each individual image or bounding box:

| Flag                      | Description                                                                 |
| ------------------------- | --------------------------------------------------------------------------- |
| DIMENSION_CHANNELS        | Number of color channels in the image                                       |
| DIMENSION_HEIGHT          | Height of the image or bounding box in pixels                               |
| DIMENSION_WIDTH           | Width of the image or bounding box in pixels                                |
| DIMENSION_SIZE            | Area of the image or bounding box in pixels                                 |
| DIMENSION_ASPECT_RATIO    | Width divided by height                                                     |
| DIMENSION_DEPTH           | Automatic calculation of the bit depth based on max and min values          |
| DIMENSION_OFFSET_X        | The x value (in pixels) of the top left corner of the bounding box          |
| DIMENSION_OFFSET_Y        | The y value (in pixels) of the top left corner of the bounding box          |
| DIMENSION_CENTER          | The x and y value (in pixels) of the center of the image or bounding box    |
| DIMENSION_DISTANCE_CENTER | Distance between the center of the image and the center of the bounding box |
| DIMENSION_DISTANCE_EDGE   | Distance from the bounding box to the nearest image edge                    |
| DIMENSION_INVALID_BOX     | Whether the box is out of bounds or has no area                             |

Convenience sub-groups:

- `DIMENSION_BASIC` - Width, height, channels
- `DIMENSION_OFFSET` - Offset X and Y
- `DIMENSION_POSITION` - Center, distance to center, distance to edge

Images are expected in CxHxW format, which is used to populate the width, height, and channels metrics.

These statistics can be used in conjunction with the {class}`.Outliers` class to determine
if there are any issues with any of the images in the dataset.

### HASH Statistics

The `HASH` flag group calculates hash values for duplicate detection:

| Flag          | Description                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------ |
| HASH_XXHASH   | [xxHash](https://github.com/Cyan4973/xxHash) for exact image matching                            |
| HASH_PHASH    | [Perceptual hash](https://en.wikipedia.org/wiki/Perceptual_hashing) for near-duplicate detection |
| HASH_DHASH    | Difference/gradient hash for near-duplicate detection                                            |
| HASH_PHASH_D4 | Perceptual hash with D4 symmetry (rotation/flip invariant)                                       |
| HASH_DHASH_D4 | Difference/gradient hash with D4 symmetry (rotation/flip invariant)                              |

Convenience sub-groups:

- `HASH_DUPLICATES_BASIC` - Standard duplicate detection (xxhash + phash + dhash)
- `HASH_DUPLICATES_D4` - Rotation/flip-invariant detection (xxhash + phash_d4 + dhash_d4)

These hashes can be used in conjunction with the {class}`.Duplicates` class to identify duplicate images.
The D4 variants detect duplicates regardless of image orientation (90°/180°/270° rotations and flips).

Use `ImageStats.HASH` to compute both hash sets and distinguish between same-orientation
duplicates (matched by both basic and D4 hashes) vs rotated/flipped duplicates (matched only by D4 hashes).
The `NearDuplicateGroup.orientation` field is automatically set to `"same"` or `"rotated"` when
both hash types are computed.

## When to use calculate with ImageStats

The {func}`.calculate` function is automatically called when using `Outliers.evaluate` on data.
Therefore, you don't usually need to call {func}`.calculate` directly.
However, there are a few scenarios where using {func}`.calculate` independently is beneficial:

- When multiple sets of data as well as the combined set are to be analyzed,
  it can be easier to run {func}`.calculate` on each individual set of data
  and then pass the outputs to the {class}`.Outliers` class in each of the desired
  data combinations for analysis.
- When comparing the resulting data distribution between two or more datasets
  to determine how similar the datasets are.
- When you need specific statistics for custom analysis or visualization.
- When using the {func}`.calculate_ratios` function to compute ratios between
  bounding box statistics and image statistics.

## Example usage

Example code for calculating all statistics for images:

```python
# Import the calculate function and ImageStats flags
from dataeval.core import calculate
from dataeval.flags import ImageStats
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2

# Loading in the PASCAL VOC 2011 dataset for this example
to_tensor = v2.ToImage()
ds = VOCDetection(
    "./data",
    year="2011",
    image_set="train",
    download=True,
    transform=to_tensor,
)

# Calculate all statistics for the images
# Note: Images should be in (C,H,W) format
result = calculate(ds, stats=ImageStats.ALL)

# Access the computed statistics
print(f"Processed {result['image_count']} images")
print(f"Available statistics: {list(result['stats'].keys())}")
```

Example code for calculating specific statistics:

```python
from dataeval.core import calculate
from dataeval.flags import ImageStats

# Calculate only pixel and visual statistics
result = calculate(
    ds,
    stats=ImageStats.PIXEL | ImageStats.VISUAL
)

# Calculate only basic pixel statistics with per-channel breakdown
result = calculate(
    ds,
    stats=ImageStats.PIXEL_BASIC,
    per_channel=True
)

# Calculate dimension statistics for both full images and bounding boxes
result = calculate(
    ds,
    stats=ImageStats.DIMENSION,
    per_image=True,
    per_box=True
)
```

### Analyzing the results

The `calculate` function returns a `CalculationResult` dictionary containing:

- `source_index`: Sequence of `SourceIndex` objects tracking which image, box, and channel each statistic corresponds to
- `object_count`: Number of objects (bounding boxes) per image
- `invalid_box_count`: Number of invalid boxes per image
- `image_count`: Total number of images processed
- `stats`: Dictionary mapping statistic names to NumPy arrays of computed values

You can analyze the distribution of statistics to identify potential issues:

```python
import numpy as np

# Get mean pixel values across all images
mean_values = result['stats']['mean']

# Identify outliers (values beyond 3 standard deviations)
mean_std = np.std(mean_values)
mean_avg = np.mean(mean_values)
outliers = np.where(np.abs(mean_values - mean_avg) > 3 * mean_std)[0]

print(f"Found {len(outliers)} outlier images based on mean pixel value")
```

When analyzing distributions, look for:

- **Uniform distribution**: Check if any areas are significantly shorter or taller than the rest
- **Normal distribution**: Look at the edges of the bell curve for raised values or gaps
- **Per-channel analysis**: Compare shapes across channels to detect processing errors or channel bias

### Using with Outliers and Duplicates

The statistics from {func}`.calculate` are used internally by the {class}`.Outliers` and {class}`.Duplicates` classes:

```python
from dataeval import Outliers, Duplicates
from dataeval.flags import ImageStats

# Outliers automatically calls calculate with appropriate stats
outliers = Outliers()
outlier_results = outliers.evaluate(ds)

# Duplicates uses hash statistics (default: HASH_DUPLICATES_BASIC)
duplicates = Duplicates()
duplicate_results = duplicates.evaluate(ds)

# For rotation/flip-invariant duplicate detection
duplicates_d4 = Duplicates(flags=ImageStats.HASH_DUPLICATES_D4)
duplicate_results = duplicates_d4.evaluate(ds)

# To distinguish same-orientation vs rotated/flipped duplicates
duplicates_full = Duplicates(flags=ImageStats.HASH)
result = duplicates_full.evaluate(ds)
for group in result.items.near or []:
    if group.orientation == "rotated":
        print(f"Rotated/flipped: {group.indices}")
    elif group.orientation == "same":
        print(f"Same orientation: {group.indices}")
```

## Performance Overview

The following performance data was collected using both small images
(CIFAR-10, 3x32x32) and medium images (VOCDetection2012, ~3x375x500)
across different computational configurations.

### Statistics Categories Benchmarked

- **DIMENSION**: Image dimension analysis
- **HASH**: Hash-based similarity detection
- **VISUAL**: Visual properties analysis (brightness, contrast, etc.)
- **PIXEL**: Pixel-level statistical analysis (mean, std, histograms)
- **ALL**: Combined pixel and visual and dimension statistics
- **Per-channel mode**: Per-channel analysis with additional overhead

## Small Images Performance (CIFAR-10)

The following chart shows execution times for processing CIFAR-10 images with 16
processes across different dataset sizes (10K, 30K, 50K images).

```{raw} html
:file: ../_static/charts/small_images_16_processes.html
```

Key observations:

- Excellent linear scaling with image count for most statistics
- **DIMENSION** and **HASH** show the best performance and efficiency
- **VISUAL** provides good performance for comprehensive visual analysis
- **PIXEL** has moderate computational cost for detailed pixel analysis
- **Per-channel mode** shows expected overhead for per-channel breakdowns

## Medium Images Performance (VOC Detection 2012)

Performance characteristics change significantly with larger images, as shown
below for VOCDetection (2012) dataset processing (1K, 3K, 5K images).

```{raw} html
:file: ../_static/charts/medium_images_16_processes.html
```

Notable differences:

- **HASH** and **DIMENSION** remain relatively efficient regardless of
  image size
- **VISUAL** maintains good performance characteristics across image sizes
- **PIXEL** shows higher computational cost with larger images due to
  increased pixel data
- **Per-channel mode** demonstrates significant overhead scaling with
  image complexity

## Process Scaling Analysis

### Small Images Process Scaling

```{raw} html
:file: ../_static/charts/small_images_process_scaling.html
```

### Medium Images Process Scaling

```{raw} html
:file: ../_static/charts/medium_images_process_scaling.html
```

## Performance Recommendations

Based on the benchmark results:

1. **For fast dataset profiling**: Use `DIMENSION | HASH` for
   rapid analysis
2. **For visual quality assessment**: `VISUAL` provides good
   performance-to-insight ratio
3. **For detailed analysis**: `PIXEL` offers comprehensive metrics with
   moderate overhead
4. **For complete analysis**: `ALL` combines all statistics efficiently
5. **Memory-constrained environments**: Avoid `per_channel=True` for large datasets
6. **Process scaling**: Multi-processing (configured via `dataeval.config`) provides optimal performance for most workloads

## Key Performance Insights

- **Diminishing returns**: Increasing process count offers diminishing returns
- **Statistic selection**: Choose the minimal set of statistics needed for your analysis using specific `ImageStats` flags
- **Per-channel overhead**: Only use `per_channel=True` when channel-specific insights are required
- **Linear scaling**: All statistics scale linearly with image count, size, and process count

## Technical Notes

- Benchmarks conducted using multiprocessing with shared memory optimization
- Times measured include I/O overhead and result aggregation
- Process scaling shows diminishing returns beyond optimal core count
- Memory usage scales proportionally with per-channel analysis depth
- Tests performed on an Intel Core i9-14900HX w/ 64GB DDR5 on Windows 11/Ubuntu
  22.04 (WSL2) with dataset loaded on local storage
- Performance applies to the {func}`.calculate` function with various `ImageStats` flag combinations
