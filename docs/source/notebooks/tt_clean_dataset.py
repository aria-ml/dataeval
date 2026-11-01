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
# Learn how to clean a dataset with DataEval. You will check a set of images that was assembled from other sources for
# bad data before you train on it.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [ML Engineer](../getting-started/roles/ml_engineer.md)

# %% [markdown]
# ## What you will do
#
# - You will use DataEval's cleaners to assess a dataset of military vehicle images.
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
# You are about to train a model on a set of images that someone assembled from other sources. Before you do, you need
# to know whether the images are what you expect: photographs of the right subjects, each stored once.
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
from maite_datasets.image_classification import MilitaryVehicles

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
    _, axs = plt.subplots(6, 4, figsize=(8, 12))

    for ax, (category, indices) in zip(axs.flat, image_indices_per_class.items(), strict=False):
        # Randomly select an index from the list of indices
        ax.imshow(dataset[rng.choice(indices)][0].transpose(1, 2, 0))
        ax.set_title(dataset.index2label[category])
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
# You are going to work with `MilitaryVehicles`, a dataset of images of 24 types of military vehicle and equipment. The
# images were assembled from other sources, so they vary in size, setting and kind of image, as a collected dataset
# does.
#
# If this data is already on your computer, you can change the file location from `"./data"` to wherever the data is
# stored. Also change the download value from `True` to `False`.
#
# To keep this tutorial fast on most computers, you are going to analyze only the train split of the data, which is a
# little under 8000 images.

# %%
# Download the data and then load it as a torch Tensor
ds = MilitaryVehicles("./data", image_set="train", download=True)
print(ds)

# %% [markdown]
# ### Inspect the data
#
# A dataset assembled from other sources is likely to have issues, so check it before you use it. Many of the large
# webscraped datasets available for use contain image issues. Verifying at the beginning that you have a high quality
# dataset is always easier than finding out later that you trained a model on a dataset with erroneous images or a set
# of splits with leakage.

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
# The above table shows that this dataset has a total of 24 classes. Each image has one label, so the label count and
# the image count are equal.
#
# `T-72`, `BTR-80`, and `BMP-1` are the classes with the most images, while `30N6E`, `BTR-70`, and `Iskander` are the
# classes with the fewest images. The largest class has about 3.5 times the images of the smallest.
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
# The plot shows the variety in the images, including image sizes, brightness, backgrounds and the kind of image. Some
# are photographs of a vehicle outdoors. Others are 3D renders, a plastic scale model, a model kit box, a drawing and a
# black and white photograph.
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
# Digging into the flagged images and organizing them by category shows that the metrics with the most extreme values
# are "width" and "size", while "mean" and "sharpness" have the fewest.
#
# `Outliers` flags any image on the edge of each metric's data distribution. Some images are flagged as an outlier by
# multiple metrics, while others are flagged by only a single metric. It is then up to you to sift through the
# information in the result from `Outliers`.
#
# Exploring the results also includes displaying how the flagged images are spread across the 24 classes.

# %%
# List the outliers by class label
outlier_imgs.aggregate_by_class(md)

# %% [markdown]
# The table above splits the issues by class and metric. It shows these trends:
#
# - An image with an unusual aspect ratio is most likely to be a `D-30`, followed by a `2S19 MSTA` or an `Rs-24`.
# - An image with an issue in brightness is most likely to be a `BTR-60`, followed by a `T-14` or a `T-64`.
# - Images with low entropy (images with constant pixels) are most likely to be a `BMP-2`, followed by a `BTR-60`, a
#   `T-14` or a `T-80`.
# - Images with many zero-valued pixels are most likely to be a `BMP-2` or a `TOS-1`, followed by a `BMP-T15`.
# - Images with high contrast are most likely to be an `Rs-24`.
# - Most classes have 16 to 24 images with a size issue. `30N6E` has 6, and `T-80`, `T-62`, `BTR-70` and `T-90` have
#   none.
#
# Remember that the number of images differs by class and that effective use of this tool requires understanding the
# dataset in question. For example, 15 low entropy images out of the 390 for `BMP-2` is a smaller share than 14 low
# entropy images out of the 299 for `T-80`, even though the count is higher.

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
images = {ds.index2label[k]: lstats["image_counts_per_class"][k] for k in ds.index2label}
assert (len(ds), len(images)) == (7823, 24)
assert lstats["label_counts_per_class"] == lstats["image_counts_per_class"]
assert sorted(images, key=lambda name: -images[name])[:3] == ["T-72", "BTR-80", "BMP-1"]
assert sorted(images, key=lambda name: images[name])[:3] == ["30N6E", "BTR-70", "Iskander"]
assert 3 < max(images.values()) / min(images.values()) < 4  # "about 3.5 times"

# Verify the metric and class counts that the text reads from the outlier tables
by_metric = dict(zip(*outlier_imgs.aggregate_by_metric().select("metric_name", "Total"), strict=True))
ranked = sorted(by_metric, key=lambda metric: -by_metric[metric])
assert ranked[:2] == ["width", "size"]
assert set(ranked[-2:]) == {"mean", "sharpness"}

by_class = {row["class_name"]: row for row in outlier_imgs.aggregate_by_class(md).iter_rows(named=True)}
by_class.pop("Total", None)


def top(metric: str, n: int) -> list[str]:
    return sorted(by_class, key=lambda name: -by_class[name][metric])[:n]


assert top("aspect_ratio", 3) == ["D-30", "2S19 MSTA", "Rs-24"]
assert top("brightness", 3) == ["BTR-60", "T-14", "T-64"]
assert top("entropy", 1) == ["BMP-2"]
assert all(by_class[name]["entropy"] == 14 for name in ("BTR-60", "T-14", "T-80"))
assert set(top("zeros", 2)) == {"BMP-2", "TOS-1"}
assert top("zeros", 3)[2] == "BMP-T15"
assert top("contrast", 1) == ["Rs-24"]
no_size_issue = {"T-80", "T-62", "BTR-70", "T-90"}
assert all(by_class[name]["size"] == 0 for name in no_size_issue)
assert by_class["30N6E"]["size"] == 6
assert all(16 <= row["size"] <= 24 for name, row in by_class.items() if name not in no_size_issue | {"30N6E"})
assert (by_class["BMP-2"]["entropy"], images["BMP-2"], by_class["T-80"]["entropy"], images["T-80"]) == (
    15,
    390,
    14,
    299,
)

assert len(outlier_imgs_classwise) != len(outlier_imgs)

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
# When you examine the flagged images for entropy, look for patterns in the content of the images. The images flagged
# here are not photographs of a scene. They are a warning sign, a vehicle on a white page and two line drawings from a
# reference sheet, so most of each image is one flat color.
#
# For example, in an operational setting, a model sees photographs or sensor imagery and not reference sheets. A
# reference drawing teaches the model what a drawing looks like and little about what the vehicle looks like in the
# field. A sign teaches it nothing about the vehicle at all.
#
# To refine your dataset, decide whether these flagged images represent scenarios that match your goals. If they do,
# for example because the model must also recognize vehicles in manuals, consider collecting more data with similar
# characteristics to balance your dataset. If not, these images may be excluded as outliers.

# %% [markdown]
# ### Aspect ratio

# %%
# Plot images flagged for "aspect_ratio"
plot_sample_outlier_images_by_metric(ds, outlier_imgs, "aspect_ratio", (1, 4))

# %% [markdown]
# The images flagged for aspect ratio are unusually tall. Three are composites: a model kit box with a column of parts,
# a stack of photographs and a social media post with text. The fourth is a photograph of a scale model on a work
# surface.
#
# If your workflow involves preprocessing images to a uniform size, verify that resizing does not distort important
# details. For example, cropping could remove relevant parts of the image, while resizing could stretch or compress
# objects. A tall composite resized to a square squeezes every panel, so the vehicles in it no longer look like the
# vehicles in your other images. Alternatively, if you plan to filter images based on size, ensure that this does not
# introduce bias, for example by disproportionately excluding images of certain classes or contexts.
#
# After evaluating the flagged images, you may notice that dimensional discrepancies are common across multiple classes,
# as shown in the earlier table. This observation suggests that these issues are a general feature of the dataset, and
# dropping all aspect ratio outliers might be an appropriate step. However, be cautious and verify whether this action
# creates any imbalances.

# %% [markdown]
# ### Zeros

# %%
# Plot images flagged for "zeros"
plot_sample_outlier_images_by_metric(ds, outlier_imgs, "zeros", (1, 4))

# %% [markdown]
# Images flagged for zeros typically feature large regions of completely black pixels. Here they are not underexposed
# photographs. They are line art and 3D renders of a vehicle on a black background.
#
# A black background can become a cue that a model uses in place of the vehicle. Check whether these images are relevant
# to your operational scenario or whether they are artifacts of how the dataset was assembled.
#
# For instance, if renders are operationally irrelevant, consider removing them. However, if the model will also see
# rendered or synthetic imagery, ensure that you have enough of it, and that its background does not line up with a
# single class. In the earlier table, `BMP-2`, `TOS-1` and `BMP-T15` have the most images flagged for zeros.

# %% [markdown]
# ### Sharpness

# %%
# Plot images flagged for "sharpness"
plot_sample_outlier_images_by_metric(ds, outlier_imgs, "sharpness", (1, 2))

# %% [markdown]
# Sharpness measures the clarity of edges in an image. The two images plotted here are the least sharp of the flagged
# images: an aerial view of a smoke plume and a rocket launch in smoke. Smoke and haze soften edges.
#
# Evaluate whether these soft images are typical of your use case. Unlike the diagrams and renders, they look like
# operational scenes, so they may be rare events that your model needs. If they are, ensure that similar images are
# sufficiently represented in your dataset to prevent biases. If they are not, consider excluding them.

# %% [markdown]
# ## Cleaning summary
#
# The Outliers class identifies images that deviate significantly from the dataset's overall distribution. While it
# cannot determine operational relevance, it shows patterns that may require further investigation.
#
# For example, flagged images might reflect real-world scenarios underrepresented in your dataset, such as a launch in
# smoke. Alternatively, they may reveal content that is not a photograph of the vehicle, such as drawings, renders and
# composites, which come from how the dataset was assembled.
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
# The `Duplicates` class identifies both exact duplicates and potential (near) duplicates. Potential duplicates can
# occur in a variety of ways:
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
# Initialize the Duplicates class. A hash radius of 0 counts two images as duplicates only when their hashes are
# identical
dups = Duplicates(ImageStats.HASH, hash_radius=0)

# Find the duplicates
results = dups.evaluate(ds, per_target=False)

# %%
# Count the duplicate groups by type, then view the first few
groups = results.data()
display(
    groups.group_by("dup_type", maintain_order=True).agg(
        pl.len().alias("groups"), pl.col("item_indices").list.len().sum().alias("images")
    )
)
display(groups.head(3))

# %% [markdown]
# The dataset has both exact duplicates, which are the same file stored twice, and near duplicates, whose perceptual
# hashes match (phash and dhash). Each row is one group of images. Plot the first exact group and one near group.

# %%
plot(ds, figsize=(12, 8), indices=(478, 6118, 1618, 7217), images_per_row=2)


# %% [markdown]
# The first pair is an exact duplicate. In the second pair, image 1618 has the watermark blurred out and image 7217
# keeps it. The files differ, but the hashes still match.
#
# Duplicates cost more than storage when the dataset has labels. Look up the labels of the images in each group.


# %%
def label_of(i: int) -> str:
    """Return the class name that the dataset gives to image `i`."""
    return ds.index2label[int(np.argmax(ds[i][1]))]


for kind in ("exact", "near"):
    found = groups.filter(pl.col("dup_type") == kind)["item_indices"].to_list()
    mixed = [group for group in found if len({label_of(i) for i in group}) > 1]
    print(f"{kind}: {len(mixed)} of {len(found)} groups hold images with different labels")

# %% [markdown]
# Every exact group and most of the near groups hold the same image under different labels. A model cannot learn both
# labels for one image, and when the copies land in different splits, the model sees the test image during training.
# Keep one copy of each group and decide which label is correct. Check a near group by eye before you drop either image,
# because the match is by perceptual hash and not by file contents.

# %% [markdown]
# To show what the `Duplicates` class can do on copies you control, you will make six images from the dataset's own
# images and then compare them with the dataset.

# %%
# Create exact and duplicate images

# Copy images 23 and 46 to create exact duplicates
# Copy and crop images 5 and 589 by 2 pixels to create near duplicates
# Rotate image 100 by 90 degrees to create a rotated duplicate
# Mirror and rotate image 200 to create a mirrored+rotated duplicate
dupes = [
    ds[23][0],  # exact duplicate
    ds[46][0],  # exact duplicate
    ds[5][0][:, 2:-2, 2:-2],  # cropped near duplicate
    ds[589][0][:, :-2, 2:],  # cropped near duplicate
    np.rot90(ds[100][0], k=1, axes=(1, 2)),  # 90° rotation
    np.flip(np.rot90(ds[200][0], k=2, axes=(1, 2)), axis=2),  # 180° rotation + horizontal flip
]

dupes_stats = image_stats(dupes, stats=ImageStats.HASH)

# %%
# Find the duplicates appended to the dataset
duplicates = dups.from_stats([dups.stats, dupes_stats])

# View the groups that contain one of the added images. The other groups are the duplicates within the dataset.
display(duplicates.data().filter(pl.col("dataset_indices").list.contains(1)))

# %% [markdown]
# The `Duplicates` class identified all six added images as exact or near duplicates. It returns the results as a
# DataFrame where each row is a duplicate group with columns for group_id, level, dup_type (exact/near), item_indices,
# methods, orientation, and dataset_indices (for cross-dataset results).
#
# - Images 23 and 46 from dataset 0 are identified as exact duplicates of images 0 and 1 from dataset 1 respectively.
# - Images 5 and 589 from dataset 0 are identified as near duplicates, with the same orientation, of images 2 and 3
#   from dataset 1 (cropped versions). The basic hashes (phash, dhash) detect them. Here dhash finds both crops and
#   phash finds only one, so a copy can escape one hash and still be caught by another.
# - Images 100 and 200 from dataset 0 are identified as rotated or flipped duplicates of images 4 and 5 from dataset 1
#   (rotated and mirrored+rotated versions). Only D4 hashes (phash_d4, dhash_d4) detect them, because the basic
#   perceptual hashes are orientation-sensitive. The orientation column shows "rotated" for these groups.
#
# A perceptual hash catches small changes only. These crops are 2 pixels, and a 5-pixel crop of image 589 is not found
# at this setting. A clean result means that no copies this similar exist, and not that the dataset has no near copies.
#
# Plot the two rotated copies next to the images that they came from. The basic hashes miss them because those hashes
# change with orientation, and the D4 hashes match them.

# %%
import matplotlib.pyplot as plt

# Each row holds an image from the dataset and the copy that was added to it
fig, axs = plt.subplots(2, 2, figsize=(10, 7))
rotated = [(100, dupes[4], "rotated 90°"), (200, dupes[5], "rotated 180° and mirrored")]
for row, (source, copy, change) in zip(axs, rotated, strict=True):
    row[0].imshow(ds[source][0].transpose(1, 2, 0))
    row[0].set_title(f"Dataset image {source}")
    row[1].imshow(copy.transpose(1, 2, 0))
    row[1].set_title(f"Added copy, {change}")
for ax in axs.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# By using `ImageStats.HASH` (which computes both basic and D4 hashes), you can distinguish between same-orientation
# duplicates and rotated/flipped duplicates by checking the methods and orientation columns. This is useful when you
# want to:
#
# - Keep one version of each rotated duplicate set
# - Identify images that may have been augmented with rotations
# - Detect unintentional orientation variations in your dataset

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Verify what the text says about the plotted images
flagged = {
    metric: set(outlier_imgs.data().filter(pl.col("metric_name") == metric)["item_index"].to_list())
    for metric in ("entropy", "aspect_ratio", "zeros", "sharpness")
}
assert {3753, 1536, 2777, 2792} <= flagged["entropy"]  # a sign, a vehicle on a white page, line drawings
assert {2094, 2662, 7572, 7292} <= flagged["aspect_ratio"]  # composites and a scale model
assert {6627, 6788, 7233, 3052} <= flagged["zeros"]  # line art and renders on black
least_sharp = outlier_imgs.data().filter(pl.col("metric_name") == "sharpness").sort("metric_value")["item_index"]
assert least_sharp.to_list()[:2] == [3674, 7655]  # the smoke plume and the launch

# Verify what the text says about the duplicate groups and their labels
exact_groups = groups.filter(pl.col("dup_type") == "exact")["item_indices"].to_list()
near_groups = groups.filter(pl.col("dup_type") == "near")["item_indices"].to_list()
assert exact_groups[0] == [478, 6118]  # the first pair plotted
assert [1618, 7217] in near_groups  # the second pair plotted
assert all(len({label_of(i) for i in group}) > 1 for group in exact_groups)  # "every exact group"
mixed_near = sum(len({label_of(i) for i in group}) > 1 for group in near_groups)
assert 0.5 < mixed_near / len(near_groups) < 0.95  # "most of the near groups"

# Verify what the text says about the six added images
added = {
    tuple(row["item_indices"]): row
    for row in duplicates.data().filter(pl.col("dataset_indices").list.contains(1)).iter_rows(named=True)
}
assert len(added) == 6
assert added[(23, 0)]["dup_type"] == added[(46, 1)]["dup_type"] == "exact"
for pair in ((5, 2), (589, 3)):
    assert added[pair]["dup_type"] == "near"
    assert added[pair]["orientation"] == "same"
assert set(added[(5, 2)]["methods"]) == {"phash", "dhash"}  # dhash finds both crops, phash only one
assert set(added[(589, 3)]["methods"]) == {"dhash"}
for pair in ((100, 4), (200, 5)):
    assert added[pair]["orientation"] == "rotated"
    assert set(added[pair]["methods"]) == {"phash_d4", "dhash_d4"}
larger_crop = image_stats([ds[589][0][:, :-5, 5:]], stats=ImageStats.HASH)
assert dups.from_stats([dups.stats, larger_crop]).data().filter(pl.col("dataset_indices").list.contains(1)).is_empty()

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
# - Images that are not photographs of the subject, such as drawings, renders and composites, show up as outliers.
# - Duplicate data can affect model performance, and the same image can carry different labels.

# %% [markdown]
# ## Next steps
#
# - [Data integrity](../concepts/DataIntegrity.md) — Understand core principles of dataset integrity, duplicates, and
#   outliers.
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — Find duplicates, outliers and gaps in coverage without
#   labels.
# - [Identify duplicates](./h2_deduplicate.py) — Remove exact, duplicate, or near-duplicate images from datasets.

# %% [markdown]
# ## On your own
#
# Now that you have completed a tutorial on exploring a dataset, repeat it with the test set, the full dataset or your
# own dataset. When you check other sets of data, observe how the stats of each grouping of data change or stay the
# same.
#
# You can also try the different statistical methods that the `Outliers` class uses to see how the method affects the
# number and type of issues detected.
