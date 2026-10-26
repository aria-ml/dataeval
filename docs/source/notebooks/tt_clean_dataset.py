# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dataeval (3.14.2)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Clean a dataset
#
# Learn how to clean a dataset with DataEval.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [ML Engineer](../getting-started/roles/ml_engineer.md)

# %% [markdown]
# ## What you will do
#
# - You will use DataEval's cleaners to assess the 2012 VOC dataset.
# - You will analyze the results with plots and tables.

# %% [markdown]
# ## What you will learn
#
# - You will learn how to assess a dataset for extreme or redundant data points.
# - You will learn which questions to ask to decide when to remove or collect additional data.

# %% [markdown]
# ## What you will need
#
# - Environment requirements
#   - `dataeval` or `dataeval[all]`

# %% [markdown]
# ## Introduction
#
# Exploratory data analysis (EDA) summarizes the main characteristics of a dataset and identifies inconsistencies in the
# data. Before you move on to machine learning or statistical modeling, you need to understand the data you are working
# with. EDA helps you find patterns, detect anomalies, check assumptions and determine relationships in the data.
#
# [Data cleaning](../concepts/DataIntegrity.md) is one of the most important parts of EDA. DataEval can identify
# duplicates and [outliers](../concepts/DataIntegrity.md#outlier-detection-image-statistics-and-embeddings), as well as
# data points that have missing values or too many extreme values. These techniques help you include only high quality
# data in your projects.

# %% [markdown]
# ## Setup
#
# Import the libraries that the example needs.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
import numpy as np
import polars as pl
from dataeval_plots import plot
from IPython.display import display
from maite_datasets.object_detection import VOCDetection

from dataeval import Metadata

# Load the classes from DataEval that are helpful for EDA
from dataeval.config import set_max_processes
from dataeval.core import image_stats, label_stats
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, Outliers

# Print all rows of dataframes
_ = pl.Config.set_tbl_rows(-1)

# Set the random value
rng = np.random.default_rng(213)

# Set multiprocessing for DataEval stats
set_max_processes(4)


# %%
# Helper method to plot sample images by class
def plot_sample_images_by_class(dataset, image_indices_per_class) -> None:
    import matplotlib.pyplot as plt

    # Plot random images from each category
    _, axs = plt.subplots(5, 4, figsize=(8, 10))

    for ax, (category, indices) in zip(axs.flat, image_indices_per_class.items(), strict=False):
        # Randomly select an index from the list of indices
        ax.imshow(dataset[rng.choice(indices)][0].transpose(1, 2, 0))
        ax.set_title(dataset.metadata["index2label"][category])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Helper method to plot images of interest
def plot_sample_outlier_images_by_metric(dataset, outlier_result, metric, layout) -> None:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Filter outliers DataFrame for the specific metric
    metric_outliers = outlier_result.data().filter(pl.col("metric_name") == metric)

    if metric_outliers.is_empty():
        print(f"No images flagged for metric: {metric}")
        return

    # Get all metric values from stored calculation results to understand the distribution
    all_metric_values = outlier_result.calculation_results["stats"][metric]
    quantiles = np.quantile(all_metric_values, [0, 0.25, 0.5, 0.75, 1])
    median = quantiles[2]

    # Compute distance from median and sort by most extreme first
    ranked = (
        metric_outliers
        .unique(subset=["item_index"])
        .with_columns((pl.col("metric_value") - median).abs().alias("_distance"))
        .sort("_distance", descending=True)
        .drop("_distance")
    )

    # Determine number of samples to plot
    n_samples = min(int(np.prod(layout)), ranked.height)

    # Create figure with space for colorbar
    fig = plt.figure(figsize=(12, layout[0] * 4))
    gs = gridspec.GridSpec(
        layout[0],
        layout[1] + 1,
        width_ratios=[1] * layout[1] + [0.05],
        hspace=0.4,
        wspace=0.1,
        left=0.05,
        right=0.92,
        top=0.92,
        bottom=0.02,
    )

    # Create colormap normalization based on full metric distribution
    vmin, vmax = quantiles[0], quantiles[4]
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    # Plot images
    for i in range(n_samples):
        row = ranked.row(i, named=True)
        img_id = row["item_index"]
        metric_value = row["metric_value"]
        color = cmap(norm(metric_value))

        ax = fig.add_subplot(gs[i // layout[1], i % layout[1]])
        ax.imshow(dataset[img_id][0].transpose(1, 2, 0))
        ax.set_xlabel(f"index: {img_id}\n{metric}: {np.round(metric_value, 3)}", fontsize=9, color="black")
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colored border to indicate extremeness
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)
            spine.set_visible(True)

    # Add colorbar with quantile markers
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([])

    quantile_labels = ["Min (Q0)", "Q1 (25%)", "Median (Q2)", "Q3 (75%)", "Max (Q4)"]
    for q_val, q_label in zip(quantiles, quantile_labels, strict=False):
        cbar.ax.axhline(q_val, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        cbar.ax.text(
            1.5,
            q_val,
            f"{q_label}\n{np.round(q_val, 2)}",
            va="center",
            fontsize=8,
            transform=cbar.ax.get_yaxis_transform(),
        )

    fig.suptitle(f'Outlier Images for "{metric}" (sorted by distance from median)', fontsize=12, y=0.99)
    plt.show()


# %% [markdown]
# ## Understand the data

# %% [markdown]
# ### Load the data
#
# You are going to work with the PASCAL VOC 2012 dataset, a small curated dataset that was used for a computer vision
# competition. Its images were used for classification, object detection, and segmentation. The dataset has multiple
# classes and images with a variety of sizes and objects, which is why this tutorial uses it.
#
# If this data is already on your computer, you can change the file location from `"./data"` to wherever the data is
# stored. Also change the download value from `True` to `False`.
#
# To keep this tutorial fast on most computers, you are going to analyze only the training set of the data, which is a
# little under 6000 images.

# %%
# Download the data and then load it as a torch Tensor
ds = VOCDetection("./data", image_set="train", year="2012", download=True)
print(ds)

# %% [markdown]
# ### Inspect the data
#
# Because this data was used for a computer vision competition, it will most likely have very few issues, but it is
# always worth checking. Many of the large webscraped datasets available for use contain image issues. Verifying at the
# beginning that you have a high quality dataset is always easier than finding out later that you trained a model on a
# dataset with erroneous images or a set of splits with leakage.

# %%
# Create the Metadata object
md = Metadata(ds)

lstats = label_stats(md.class_labels, md.item_indices, ds.index2label, image_count=len(ds))

# View per_class data as a DataFrame
pl.DataFrame({
    "class_name": list(ds.index2label.values()),
    "label_count": [lstats["label_counts_per_class"][k] for k in ds.index2label],
    "image_count": [lstats["image_counts_per_class"][k] for k in ds.index2label],
})

# %% [markdown]
# The above table shows that this dataset has a total of 20 classes.
#
# Of the classes, `person` is the class with the highest total object count followed by `chair` and `car`, while `person`,
# `chair`, and `dog` are the classes with the highest number of images.
#
# `bus`, `train`, and `cow` are the classes with the fewest objects, while `cow`, `sheep`, and `bus` are the classes with
# the fewest images.
#
# This table shows wide variation in the number of classes per image, the number of objects per image and the number of
# objects of each class per image.
#
# These differences point to an important concept,
# [class balance](../concepts/DatasetBias.md#correcting-for-imbalance-classbalance). A dataset that is imbalanced can
# result in a model that chooses the more prominent class more often only because there are more samples in that class.
# To explore this concept further, see the bias tutorial in the [Next steps](#next-steps) section at the end of this
# tutorial.
#
# Now that you have examined the metadata, inspect random images to get an idea of the variety of backgrounds, the range
# of colors, the locations of objects in images, and how often an image shows a single object versus multiple objects.

# %%
plot_sample_images_by_class(ds, lstats["image_indices_per_class"])

# %% [markdown]
# The plot shows the variety in the images, including image sizes, image brightness, object sizes, backgrounds, the
# number of objects in the image, and the lack of color in a few black and white images.
#
# DataEval helps you make sense of the many aspects that affect building representative datasets and robust models.

# %% [markdown]
# Besides making sure that you understand the structure of the labels and have visualized some of the images from the
# dataset, you can visualize the data distribution across statistics such as the image size or the pixel mean. To view
# these distributions, use DataEval's stat functions and plot the results.
#
# Now you can identify which images differ statistically from the rest of the images.

# %% [markdown]
# ## Identify any outlying data points

# %% [markdown]
# ### Extreme/missing values
#
# Here you will identify the images with extreme values from DataEval's stat functions. To detect these extreme values,
# you will use the {class}`.Outliers` class. The `Outliers` class has multiple methods to determine the extreme values,
# which are discussed in the [Data Cleaning explanation](../concepts/DataIntegrity.md). For this guide, you will use the
# "zscore" because the Z score defines outliers in a normal distribution.
#
# The output of the `Outliers` class contains a DataFrame with columns for image_id, metric_name, and metric_value for
# each flagged outlier.

# %%
# This cell takes about 1-5 minutes to run depending on your hardware

# Initialize the Outliers class
outliers = Outliers(outlier_threshold="zscore")

# Find the extreme images
outlier_imgs = outliers.evaluate(ds, per_target=False)

# View the number of extreme images
print(f"Number of images with extreme values: {len(outlier_imgs)}")

# %% [markdown]
# This class can flag a lot of images, depending on how varied the dataset is and which method you use to define extreme
# values. Switching the method can give different results.

# %%
# List the metrics with an extreme value
outlier_imgs.aggregate_by_metric()

# %% [markdown]
# Digging into the flagged images and organizing them by category shows that the metric with the most extreme values is
# "size" while "sharpness" has the fewest.
#
# `Outliers` flags any image on the edge of each metric's data distribution. Some images are flagged as an outlier by
# multiple metrics, while others are flagged by only a single metric. It is then up to you to sift through the
# information in the result from `Outliers`.
#
# Exploring the results also includes displaying how the flagged images are spread across the 20 classes.

# %%
# List the outliers by class label
outlier_imgs.aggregate_by_class(md)

# %% [markdown]
# The table above splits the issues by class and metric. It shows these trends:
#
# - An image with an unusual aspect ratio is most likely to contain a bottle or person.
# - An image with an issue in brightness is most likely to contain an aeroplane.
# - An image with an issue in darkness is most likely to be a person.
# - Images with high contrast are likely to fall within 1 of 4 classes: bottle, cat, chair, person.
# - Images with low entropy (images with constant pixels) are likely to fall within 1 of 4 classes: aeroplane, bird,
#   bottle, person.
# - Unusual skew and kurtosis images follow a similar trend to entropy.
# - Every class has images with size issues.
#
# Remember that the number of images differs by class and that effective use of this tool requires understanding the
# dataset in question. For example, 36 low entropy images out of the 2000 for person might be outliers, while 28 low
# entropy images out of 300 for aeroplane might not be. Low entropy might be an inherent characteristic of the aeroplane
# class.

# %% [markdown]
# {meth}`.OutliersOutput.classwise` answers that caution directly. It fits the thresholds within each class, so an image
# is flagged only when it is unusual for its own class. A class with fewer than 20 rows has too few to fit a threshold
# and is not judged.

# %%
# Fit the thresholds within each class instead of across the whole dataset
outlier_imgs_classwise = outlier_imgs.classwise(md)

print(f"Number of images flagged globally  : {len(outlier_imgs)}")
print(f"Number of images flagged per class : {len(outlier_imgs_classwise)}")

# %% [markdown]
# The two counts differ because each class sets its own thresholds. The classwise result supports the same aggregations
# as the global one, such as `aggregate_by_class(md)`.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Verify the class counts that the text reads from the label statistics table
objects = {ds.index2label[k]: lstats["label_counts_per_class"][k] for k in ds.index2label}
images = {ds.index2label[k]: lstats["image_counts_per_class"][k] for k in ds.index2label}
assert 5000 < len(ds) < 6000
assert len(objects) == 20
assert sorted(objects, key=lambda name: -objects[name])[:3] == ["person", "chair", "car"]
assert sorted(images, key=lambda name: -images[name])[:3] == ["person", "chair", "dog"]
assert sorted(objects, key=lambda name: objects[name])[:3] == ["bus", "train", "cow"]
assert sorted(images, key=lambda name: images[name])[:3] == ["cow", "sheep", "bus"]

assert len(outlier_imgs_classwise) != len(outlier_imgs)
assert not outlier_imgs_classwise.populations["judged"].all()

# %% [markdown]
# To understand the table above, you will plot sample images for these metrics:
#
# - entropy
# - size
# - zeros
# - sharpness
#
# Entropy, variance, standard deviation, kurtosis, and skew all measure (in different ways) how much change there is
# across the pixels in the image, and entropy is the easiest to understand.
#
# Size, width, height, and aspect ratio are interrelated, and size has the most extreme images among those.
#
# Zeros is a category of its own, but it is closely related to brightness, contrast, darkness, and mean. Zeros measures
# the percentage of pixels with a zero value compared to the average image.
#
# Sharpness is also a category of its own. It measures the perceived edges in an image.

# %% [markdown]
# (questions)=
#
# #### Questions
#
# When you look at these images, consider these questions:
#
# - Does this image represent something that would be expected in operation?
# - Is there commonality to the objects in the images?
# - Is there commonality to the backgrounds of the images?
# - Is there commonality to the class of objects in the images?
#
# Asking these questions helps you notice problems such as all objects being located on the left side of the image or
# all the images of a specific class having a specific background. Training a model with data that has commonalities can
# cause your model to develop biases or limit your model's ability to generalize to non-training data.

# %% [markdown]
# ### Entropy

# %%
# Plot images flagged for "entropy"
plot_sample_outlier_images_by_metric(ds, outlier_imgs, "entropy", (1, 4))

# %% [markdown]
# When you examine the flagged images for entropy, look for patterns in the content of the images. Many of these images
# may feature backgrounds with very little variation, such as water or sky. Others might have darker backgrounds than
# usual.
#
# For example, in an operational setting, water or sky backgrounds may or may not appear frequently, depending on the
# expected use case. Similarly, darker images may indicate low-light conditions, which could suggest either operational
# relevance (e.g., night operations) or anomalies that need to be addressed.
#
# To refine your dataset, decide whether these flagged images represent scenarios that match your goals. If they do,
# consider collecting more data with similar characteristics to balance your dataset. If not, these images may be
# excluded as outliers.

# %% [markdown]
# ### Aspect ratio

# %%
# Plot images flagged for "aspect_ratio"
plot_sample_outlier_images_by_metric(ds, outlier_imgs, "aspect_ratio", (1, 4))

# %% [markdown]
# Flagged images for aspect ratio often include examples where the objects in the image are unusually wide or tall
# relative to the rest of the dataset. For instance, bottle images might be cropped tall.
#
# If your workflow involves preprocessing images to a uniform size, verify that resizing does not distort important
# details. For example, cropping could remove relevant parts of the image, while resizing could stretch or compress
# objects. Alternatively, if you plan to filter images based on size, ensure that this does not introduce bias, for
# example by disproportionately excluding images of certain classes or contexts.
#
# After evaluating the flagged images, you may notice that dimensional discrepancies are common across multiple classes,
# as shown in the earlier table. This observation suggests that these issues are a general feature of the dataset, and
# dropping all size outliers might be an appropriate step. However, be cautious and verify whether this action creates any
# imbalances.

# %% [markdown]
# ### Zeros

# %%
# Plot images flagged for "zeros"
plot_sample_outlier_images_by_metric(ds, outlier_imgs, "zeros", (1, 4))

# %% [markdown]
# Images flagged for zeros typically feature large regions of completely black or gray pixels. Some of these may also
# appear in grayscale. These characteristics could indicate issues like underexposed photos, scanning errors, or specific
# use cases.
#
# Grayscale images, in particular, might stand out if the rest of your dataset is primarily in color. Check whether
# grayscale images are relevant to your operational scenario or whether they are artifacts of the data collection process.
#
# For instance, if grayscale images are operationally irrelevant, consider removing them. However, if grayscale scenarios
# are possible, ensure that you have sufficient representation of these types of images to train a robust model.
# Similarly, dark images with many zero-value pixels may indicate rare but valid scenarios (e.g., nighttime operations) or
# irrelevant anomalies.

# %% [markdown]
# ### Sharpness

# %%
# Plot images flagged for "sharpness"
plot_sample_outlier_images_by_metric(ds, outlier_imgs, "sharpness", (1, 2))

# %% [markdown]
# Sharpness measures the clarity of edges in an image. Flagged images often include those with unusually crisp or blurry
# details. For instance, you might notice a close-up shot of leaves or grass, where the texture stands out significantly
# compared to other images in the dataset.
#
# Evaluate whether these highly detailed images are typical of your use case. If they are uncommon in your operational
# scenario, they might skew your model's ability to generalize. In such cases, consider excluding these images.
# Conversely, if they are operationally relevant, ensure that similar images are sufficiently represented in your dataset
# to prevent biases.

# %% [markdown]
# ## Cleaning summary
#
# The Outliers class identifies images that deviate significantly from the dataset's overall distribution. While it
# cannot determine operational relevance, it shows patterns that may require further investigation.
#
# For example, flagged images might reflect real-world scenarios underrepresented in your dataset, such as night
# operations or objects photographed from unusual angles. Alternatively, they may reveal anomalies, such as artifacts from
# the data collection process.
#
# By reviewing flagged images for multiple metrics and examples, you can better understand how the Outliers class
# identifies extremes. This hands-on exploration helps you decide whether to include or exclude specific images based on
# your dataset's intended use.

# %% [markdown]
# ## Identify duplicate data

# %% [markdown]
# ### Duplicates
#
# Now that you know how to identify poor quality images in your dataset, another important part of data cleaning is
# detecting and removing duplicates.
#
# The `Duplicates` class identifies both exact duplicates and potential (near) duplicates. Potential duplicates can occur
# in a variety of ways:
#
# - Intentional perturbations
#   - Images with varying brightness
#   - Translating the image
#   - Padding the image
#   - Cropping the image
# - Unintentional changes
#   - Copying the image from one format to another (png->jpeg)
#   - Using the same image with two different filenames
#   - Duplicate frames from video extraction
#   - Oversight in the data collection process

# %%
# Initialize the Duplicates class
dups = Duplicates(ImageStats.HASH)

# Find the duplicates
results = dups.evaluate(ds, per_target=False)

# %%
# View duplicate groups
display(results)

# %% [markdown]
# As expected, there are no exact duplicate images in this dataset, since it was curated for a specific competition.
# There are, however, near duplicates detected by phash and dhash.

# %%
_ = plot(ds, figsize=(12, 6), indices=(1548, 1561))

# %% [markdown]
# The image is a near duplicate: image 1561 is a grayscale version of image 1548.

# %% [markdown]
# To show what the `Duplicates` class can do, you will add some duplicates to the dataset and then rerun the
# `Duplicates` class.

# %%
# Create exact and duplicate images

# Copy images 23 and 46 to create exact duplicates
# Copy and crop images 5 and 4376 to create near duplicates
# Rotate image 100 by 90 degrees to create a rotated duplicate
# Mirror and rotate image 200 to create a mirrored+rotated duplicate
dupes = [
    ds[23][0],  # exact duplicate
    ds[46][0],  # exact duplicate
    ds[5][0][:, 2:-2, 2:-2],  # cropped near duplicate
    ds[4376][0][:, :-5, 5:],  # cropped near duplicate
    np.rot90(ds[100][0], k=1, axes=(1, 2)),  # 90° rotation
    np.flip(np.rot90(ds[200][0], k=2, axes=(1, 2)), axis=2),  # 180° rotation + horizontal flip
]

dupes_stats = image_stats(dupes, stats=ImageStats.HASH)

# %%
# Find the duplicates appended to the dataset
duplicates = dups.from_stats([dups.stats, dupes_stats])

# View all duplicate groups
display(duplicates)

# %% [markdown]
# The `Duplicates` class identified all images from the second dataset as exact or near duplicates. It returns the
# results as a DataFrame where each row is a duplicate group with columns for group_id, level, dup_type (exact/near),
# item_indices, methods, orientation, and dataset_indices (for cross-dataset results).
#
# - Images 23 and 46 from dataset 0 are identified as exact duplicates of images 0 and 1 from dataset 1 respectively.
# - Images 5 and 4376 from dataset 0 are identified as near duplicates, with the same orientation, of images 2 and 3
#   from dataset 1 (cropped versions). Both basic hashes (phash, dhash) and D4 hashes detect them.
# - Images 100 and 200 from dataset 0 are identified as rotated or flipped duplicates of images 4 and 5 from dataset 1
#   (rotated and mirrored+rotated versions). Only D4 hashes (phash_d4, dhash_d4) detect them, because the basic
#   perceptual hashes are orientation-sensitive. The orientation column shows "rotated" for these groups.
#
# By using `ImageStats.HASH` (which computes both basic and D4 hashes), you can distinguish between same-orientation
# duplicates and rotated/flipped duplicates by checking the methods and orientation columns. This is useful when you want
# to:
#
# - Keep one version of each rotated duplicate set
# - Identify images that may have been augmented with rotations
# - Detect unintentional orientation variations in your dataset

# %% [markdown]
# ## Conclusion
#
# You used DataEval's `Outliers` class to identify and analyze images that deviate from the overall distribution of your
# dataset, and DataEval's `Duplicates` class to identify exact and near duplicates. By examining the images flagged by
# the different metrics, you learned about potential issues within your dataset.

# %% [markdown]
# ## What you learned
#
# - Some classes are underrepresented and may require additional data collection.
# - Image characteristics such as brightness, sharpness, or size can be inconsistent, which could affect model
#   performance.
# - Duplicate data can affect model performance.

# %% [markdown]
# ## Next steps
#
# - [Data Integrity](../concepts/DataIntegrity.md) — Understand core principles of dataset integrity, duplicates, and outliers.
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — Evaluate coverage, bias, and dimensionality across dataset splits.
# - [How to identify duplicates](./h2_deduplicate.py) — Remove exact, duplicate, or near-duplicate images from datasets.

# %% [markdown]
# ## On your own
#
# Now that you have completed a tutorial on exploring a dataset, repeat it with the test set, the full dataset or your
# own dataset. When you check other sets of data, observe how the stats of each grouping of data change or stay the
# same.
#
# You can also try the different statistical methods that the `Outliers` class uses to see how the method affects the
# number and type of issues detected.
