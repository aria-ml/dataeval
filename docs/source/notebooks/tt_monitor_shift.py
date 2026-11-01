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
#     display_name: dataeval
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Monitor shifts in operational data
#
# Learn the basics of monitoring data shifts after deployment. You will compare drone imagery from the platform that a
# model was built on with imagery from a second platform.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Monitoring](../getting-started/roles/ML_Lifecycle.md#monitoring)
#
# Relevant personas: [ML Engineer](../getting-started/roles/ml_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)
#
# ## What you will do
#
# - Select reference and operational frames from a dataset with a dataset view
# - Construct [embeddings](../concepts/Embeddings.md) by inferencing images through a pretrained network
# - Compare different drift detectors and understand their strengths
# - Combine the detectors with a majority vote
# - Inspect detector-specific outputs for root-cause analysis
# - Use chunked drift detection to monitor drift across data segments
#
# ## What you will learn
#
# - Learn the strengths and trade-offs of each drift detector
# - Learn why one detector is not enough and how a majority vote helps
# - Learn how to analyze embeddings for operational drift
# - Learn that an embedding encodes the whole image, so drift shows that the images changed and not which factor changed
# - Learn to tell covariate shift from a change in the class mix, and why embedding detectors cannot see concept drift
# - Learn how to inspect per-feature and per-detector statistics
# - Learn how to use chunked drift detection for temporal monitoring
#
# ## What you will need
#
# - Knowledge of Python
# - Beginner knowledge of PyTorch or neural networks

# %% [markdown]
# ## Introduction
#
# Monitoring is a step in the [AI/ML lifecycle](../getting-started/roles/ML_Lifecycle.md). When a model is deployed,
# data can, and generally will, [drift](../concepts/DistributionShift.md) from the distribution on which the model was
# originally trained. Detecting changes in the operational distribution is part of AI T&E, so that you can address them
# early. Some change might not affect performance, but significant deviation is often associated with model degradation.
#
# In this tutorial, you will use the [SeaDrone](https://seadronessee.cs.uni-tuebingen.de/) maritime dataset, which holds
# frames from three drones. Suppose that a model was built on imagery from one of them, and that the operator then adds
# a second drone that flies higher and produces frames of different sizes. The frames from the first drone are your
# reference data. You will first check held-out frames from the same drone, which should not drift. You will then check
# frames from the second drone, which should.

# %% [markdown]
# ## Setup
#
# Begin by importing the libraries that this tutorial uses.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # %pip install -q dataeval dataeval-plots[plotly] maite-datasets
except Exception:
    pass

# %%
import dataeval_plots as dep
import numpy as np
import plotly.express as px
import plotly.io as pio
import polars as pl
import torch
from IPython.display import display
from maite_datasets.object_detection import SeaDrone
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings
from dataeval.config import set_device, set_seed
from dataeval.data import Filter, Indices, Limit, Shuffle, View
from dataeval.extractors import TorchExtractor
from dataeval.shift import ChunkedDrift, DriftDomainClassifier, DriftKNeighbors, DriftMMD, DriftUnivariate

# Set a random seed
rng = np.random.default_rng(213)

# Set default device for notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_device(device)

# Seed NumPy and torch so the drift detectors are reproducible
set_seed(213, all_generators=True)

# Use plotly to render plots
dep.set_default_backend("plotly")

# Use the notebook renderer so JS is embedded
pio.renderers.default = "notebook"

# %% [markdown]
# :::{note}
# The device is set above because later steps use it. The device is the piece of hardware where the model, data, and
# other related objects are stored in memory. If a GPU is available, this notebook uses it and not the CPU. To force
# running only on the CPU, change `device` to `"cpu"`. For more information, see the
# [PyTorch device page](https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html).
# :::

# %% [markdown]
# ## Construct embeddings
#
# Rather than compare raw pixels, you will work in [embedding](../concepts/Embeddings.md) space: a compact feature
# vector per image that keeps drift detection fast and memory-light without sacrificing accuracy.
#
# ### Load the dataset
#
# Load the `val` split of SeaDrone. Set `lazy=True` so that an image is decoded only when it is used, which keeps the
# metadata steps below fast.

# %%
# Load the dataset without decoding the images
ds = SeaDrone("./data", image_set="val", download=True, lazy=True)
print(ds)

# %% [markdown]
# Compare the three drones with the metadata of each frame. SeaDrone writes `-1` where the telemetry was not recorded,
# so the median altitude uses only the frames with a recorded altitude.

# %%
# One row per frame, read from the metadata that each datum carries. A lazy dataset does not decode the image for this.
frames = pl.DataFrame([ds[i][2] for i in range(len(ds))]).select("drone", "altitude", "width")

by_drone = frames.group_by("drone").agg(
    pl.len().alias("frames"),
    pl.col("altitude").filter(pl.col("altitude") > 0).median().alias("median_altitude"),
    pl.col("altitude").filter(pl.col("altitude") > 0).len().alias("frames_with_altitude"),
    pl.col("width").min().alias("min_width"),
    pl.col("width").max().alias("max_width"),
)
by_drone.sort("frames", descending=True)

# %% [markdown]
# The `mavic` drone has the most frames. All of its frames are 3840 pixels wide and its median altitude is about 37 m.
# The `trinity` drone flies higher, with a median altitude of about 110 m where it was recorded, and its frames vary in
# size from about 1230 to 5456 pixels wide. These are the differences that you will monitor. The altitude of most
# `trinity` frames was not recorded, so treat it as a rough guide.

# %% [markdown]
# ### Select the reference and operational frames
#
# A deployed model sees the data that its operating environment produces, and that is what you monitor. Here you will
# select three sets of frames with a {class}`.View`, which wraps a dataset without copying it, and a {class}`.Filter`,
# which keeps the frames that match a condition on a metadata factor.
#
# - The reference is 250 `mavic` frames. They stand in for the data that the model was built on.
# - The operational data is 100 other `mavic` frames, held out from the reference. They come from the same drone, so
#   they should not drift.
# - The new drone is 150 `trinity` frames.
#
# :::{note}
# The filter selects frames with a metadata factor, but the embeddings are computed from the whole image. Drift in the
# embeddings shows that the images changed, and not which factor changed. The two drones differ in altitude and image
# size together, so a detector cannot tell you which of them matters. Keep this in mind whenever you monitor
# embeddings: the vector reflects everything the camera saw.
# :::
#
# > For other ways to reshape a dataset in place, see [Build dataset views](./h2_build_dataset_views.py).

# %%
# Shuffle the frames of the first drone, then split them into the reference and the held-out operational frames
mavic = View(ds, operations=[Filter(where=pl.col("drone") == "mavic"), Shuffle(seed=0)])
reference_view = View(mavic, operations=[Indices(range(250))])
operational_view = View(mavic, operations=[Indices(range(250, 350))])

# Select frames from the second drone
trinity_view = View(ds, operations=[Filter(where=pl.col("drone") == "trinity"), Shuffle(seed=0), Limit(150)])

print(f"reference:   {len(reference_view)} mavic frames")
print(f"operational: {len(operational_view)} held-out mavic frames")
print(f"new drone:   {len(trinity_view)} trinity frames")

# %% [markdown]
# Before you embed the frames, look at the first three frames of the reference, then of the new drone.

# %%
# Plot the first three frames of each drone, labeled with the drone and the width of the frame in pixels. The frames are
# large, so draw them with matplotlib, which renders each figure as a small image, and not with plotly, which embeds
# every frame at full resolution in the page.
for view in (reference_view, trinity_view):
    fig = dep.plot(
        view,
        indices=[0, 1, 2],
        images_per_row=3,
        figsize=(10, 3.5),
        show_metadata=["drone", "width"],
        backend="matplotlib",
    )
    display(fig)

# %% [markdown]
# Both drones photograph open water with small boats and swimmers, so the difference between them is hard to see by eye.
# The green boxes mark the annotated objects, which you will count later. A drift detector compares the whole set of
# frames, which is more than a glance at a few images allows.

# %% [markdown]
# ### Extract embeddings
#
# Reduce each image to a feature vector with a
# [pretrained ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) from
# Torchvision. Point a {class}`.TorchExtractor` at its penultimate `avgpool` layer (its learned features, not the
# 1000-class logits), and give it the standard resize-and-normalize preprocessing of the model as `transforms`. Then run
# each set of frames through {class}`.Embeddings`.
#
# For other ways to build embeddings, see [Create embeddings](./h2_create_embeddings.py).

# %%
resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
transforms = ResNet18_Weights.DEFAULT.transforms()
extractor = TorchExtractor(resnet, transforms=transforms, layer_name="avgpool")

# Create embeddings for the reference, the held-out operational frames and the frames from the new drone
reference_embs = Embeddings(reference_view, extractor=extractor, batch_size=64)
operational_embs = Embeddings(operational_view, extractor=extractor, batch_size=64)
shifted_embs = Embeddings(trinity_view, extractor=extractor, batch_size=64)

# %% [markdown]
# Each image is now a single feature vector instead of a full-resolution image, which speeds up the drift algorithms
# below without affecting the accuracy of the results.

# %%
print(f"({len(reference_embs)}, {reference_embs[0].shape})")  # (250, shape)
print(f"({len(operational_embs)}, {operational_embs[0].shape})")  # (100, shape)
print(f"({len(shifted_embs)}, {shifted_embs[0].shape})")  # (150, shape)

# %% [markdown]
# ## Understanding drift detectors
#
# Before you test for drift, it helps to understand the approaches available. Each detector has strengths that suit
# certain scenarios. This tutorial uses four detectors that take different strategies to detect distributional change.
#
# | Detector                        | Approach                                         | Strengths                                   | Best for                              |
# | ------------------------------- | ------------------------------------------------ | ------------------------------------------- | ------------------------------------- |
# | {class}`.DriftUnivariate` (CVM) | Statistical test per feature                     | Fast, interpretable per-feature results     | Identifying which features drifted    |
# | {class}`.DriftMMD`              | Kernel-based multivariate test                   | Captures feature dependencies               | High-dimensional data, complex shifts |
# | {class}`.DriftDomainClassifier` | Trains a classifier to distinguish distributions | Feature importances for root-cause analysis | Understanding why drift occurred      |
# | {class}`.DriftKNeighbors`       | Compares k-NN distances                          | Lightweight and fast                        | Quick monitoring checks               |
#
# :::{note}
# `DriftUnivariate` supports several statistical tests beyond CVM, including Kolmogorov-Smirnov (`ks`), Mann-Whitney U
# (`mwu`), Anderson-Darling (`anderson`), and Baumgartner-Weiss-Schindler (`bws`). Each has different sensitivity
# characteristics. See the [drift concept page](../concepts/DistributionShift.md#drift-detection) for details.
# :::

# %% [markdown]
# ## Test for drift
#
# In this step, you will check for drift between the reference embeddings and the embeddings from before. If drift is
# detected, retrain a model that was trained on this reference data with new operational data. This can help mitigate
# performance degradation in a deployed model. See the [About Drift](../concepts/DistributionShift.md) page to learn
# more.
#
# ### Drift detectors
#
# DataEval offers several drift detectors. This tutorial uses four that each take a different approach:
# {class}`.DriftUnivariate`, {class}`.DriftMMD`, {class}`.DriftDomainClassifier`, and {class}`.DriftKNeighbors`.
#
# Since each detector outputs a binary decision on whether drift is detected, you can use a **majority vote** to decide
# whether the data has drifted.\
# To learn more about these algorithms, see the
# [theory behind drift detection](../concepts/DistributionShift.md#taxonomy-of-shift) concept page.
#
# ### Fit the detectors
#
# Each drift detector needs a reference set that the operational set will be compared against. In the following code,
# you will set the reference data to the reference embeddings.

# %%
# A type alias for all of the drift detectors
DriftDetector = DriftUnivariate | DriftMMD | DriftDomainClassifier | DriftKNeighbors

# Create a mapping for the detectors to iterate over
detectors: dict[str, DriftDetector] = {
    "CVM": DriftUnivariate(method="cvm").fit(reference_embs),
    "MMD": DriftMMD().fit(reference_embs),
    "MVDC": DriftDomainClassifier().fit(reference_embs),
    "KNN": DriftKNeighbors().fit(reference_embs),
}

# %% [markdown]
# ### Make predictions
#
# Now that the detectors are set up, you can make predictions. Start with the held-out frames from the reference drone.

# %%
# Iterate and print the name of the detector class and its boolean drift prediction
clean_results = {name: detector.predict(operational_embs) for name, detector in detectors.items()}

print("\n".join(f"{res[0]} detected drift? {res[1].drifted}" for res in clean_results.items()))

alerts = sum(result.drifted for result in clean_results.values())
print(f"Detectors that detected drift: {alerts} of {len(clean_results)}")

# %% [markdown]
# The held-out frames come from the same drone as the reference, so you expect no drift. Three of the four detectors
# agree, but `CVM` reports drift. That is a false alarm, and one detector can raise it on its own. Do not act on a
# single detector. Instead, use a **majority vote** and report drift only when more than half of the detectors report
# it. Here one detector in four reports drift, so the vote says that there is no drift.
#
# Now check that the detectors can detect a real change. Run them on the frames from the second drone.

# %%
# Iterate and print the name of the detector class and its boolean drift prediction
shifted_results = {name: detector.predict(shifted_embs) for name, detector in detectors.items()}

print("\n".join(f"{res[0]} detected drift? {res[1].drifted}" for res in shifted_results.items()))

alerts = sum(result.drifted for result in shifted_results.values())
print(f"Detectors that detected drift: {alerts} of {len(shifted_results)}")

# %% [markdown]
# All four detectors detect drift, so the majority vote does too. This is the behavior that you want from a monitor: it
# stays quiet on frames like the reference and alerts when the drone changes. The detectors do not say whether the
# altitude, the image size or something else caused the change.

# %% [markdown]
# ### Identify the kind of shift
#
# Detecting drift tells you that the data changed. What you do next depends on what changed. The
# [taxonomy of shift](../concepts/DistributionShift.md#taxonomy-of-shift) separates three kinds, and your data shows
# two of them.
#
# - **Covariate shift** is a change in the inputs, here the images, while the task stays the same. The detectors above
#   measure it, because they compare embeddings of the images.
# - **Label shift** is a change in how often each class occurs.
# - **Concept drift** is a change in what a label means for the same input.
#
# To see covariate shift, project the embeddings of the three sets of frames onto their two largest directions of
# variation with `dep.project`, and color each point by its set of frames.

# %%
# Project all of the embeddings onto two dimensions and color the points by set of frames
set_names = {0: "reference", 1: "held-out", 2: "trinity"}
all_embs = np.concatenate([reference_embs, operational_embs, shifted_embs])
set_ids = np.repeat(list(set_names), [len(reference_embs), len(operational_embs), len(shifted_embs)])
dep.project(all_embs, method="pca", labels=set_ids, label_names=set_names, figsize=(7, 5.5))

# %% [markdown]
# The held-out frames land on top of the reference frames. The `trinity` frames gather in one region that the reference
# covers sparsely. The inputs moved, which is covariate shift. Two dimensions show only part of the picture, so the
# detectors, which use all 512 dimensions, separate the sets better than the plot does.
#
# The images are not the only thing that can change. Compare how often each class occurs among the annotated objects in
# each set of frames.

# %%
# Count the annotated objects of each class in each set of frames, skipping the unused class at index 0
views = {"reference": reference_view, "held-out": operational_view, "trinity": trinity_view}
class_names = [ds.index2label[i] for i in range(1, len(ds.index2label))]
class_mix, objects_per_frame = {}, {}
for name, view in views.items():
    labels = np.concatenate([np.asarray(view[i][1].labels) for i in range(len(view))])
    counts = np.bincount(labels, minlength=len(ds.index2label))[1:]
    class_mix[name] = counts / counts.sum()
    objects_per_frame[name] = counts.sum() / len(view)
    print(f"{name:10s} {counts.sum()} objects in {len(view)} frames, {objects_per_frame[name]:.1f} per frame")

# Plot the share of each class as grouped bars
mix = pl.DataFrame([
    {"frames": name, "class": class_name.replace("_", " "), "share": 100 * share}
    for name, shares in class_mix.items()
    for class_name, share in zip(class_names, shares, strict=True)
])
(
    px
    .bar(mix, x="class", y="share", color="frames", barmode="group", text_auto=True, height=400)
    .update_traces(texttemplate="%{y:.0f}")
    .update_layout(yaxis_title="Share of objects (%)", legend_title_text="Frames")
)

# %% [markdown]
# The held-out frames match the reference. The `trinity` frames hold far more boats (46% of the objects, against 27%)
# and fewer swimmers (43%, against 57%), and each frame holds fewer objects (4.2, against 6.8). The class mix changed
# together with the images, which is label shift on top of covariate shift. Real shifts are often a mix like this, and
# the embedding detectors report that the frames changed without saying which part. To compare label distributions
# directly, see [Measure label independence](./h2_measure_label_independence.py).
#
# Concept drift is different. The images can stay the same while a label changes meaning, for example when a labeling
# standard changes. An embedding comparison cannot detect this, because embeddings hold no labels. It also cannot tell
# you whether a model performs worse on the new frames. A drift alert is a warning that the data changed, and measuring
# the effect on a model takes labels or model outputs. To monitor with model outputs, see
# [Detect distribution shift from precomputed model outputs](./h2_detect_drift_from_model_outputs.py).

# %% [markdown]
# ### Inspecting detector outputs
#
# Each detector reports whether drift occurred and also provides statistics that reveal different things about the
# drift. The next cells show what each detector tells you.

# %% [markdown]
# #### DriftUnivariate: per-feature analysis
#
# The univariate detector tests each feature independently and reports which features drifted and their p-values. This
# is useful for identifying which dimensions of the embedding space shifted.

# %%
cvm_result = shifted_results["CVM"]
cvm_details = cvm_result.details

n_drifted = sum(cvm_details["feature_drift"])
n_features = len(cvm_details["feature_drift"])
print(f"Features drifted: {n_drifted}/{n_features}")
print(f"Corrected p-value threshold: {cvm_details['feature_threshold']:.6f}")
print(f"Min feature p-value: {min(cvm_details['p_vals']):.6f}")
print(f"Max feature p-value: {max(cvm_details['p_vals']):.6f}")

# %% [markdown]
# #### DriftDomainClassifier: feature importances
#
# The domain classifier trains a model to distinguish reference from test data and reports how important each feature
# was in making that distinction. High AUROC means the distributions are easily separable, which is a strong signal of
# drift.

# %%
mvdc_result = shifted_results["MVDC"]
mvdc_details = mvdc_result.details

print(f"AUROC: {mvdc_result.distance:.4f} (threshold: {mvdc_result.threshold})")
print(f"Per-fold AUROCs: {[round(a, 4) for a in mvdc_details['fold_aurocs']]}")

# Show top 5 most important features
importances = np.array(mvdc_details["feature_importances"])
top_indices = np.argsort(importances)[::-1][:5]
print("\nTop 5 features driving drift:")
print("\n".join(f"  Feature {idx}: importance = {importances[idx]:.4f}" for idx in top_indices))

# %% [markdown]
# #### DriftKNeighbors: distance comparison
#
# The k-NN detector compares how far test samples are from their nearest neighbors in the reference set versus the
# expected baseline distance. A large increase signals that test data occupies different regions of feature space.

# %%
knn_result = shifted_results["KNN"]
knn_details = knn_result.details

print(f"Mean reference k-NN distance: {knn_details['mean_ref_distance']:.4f}")
print(f"Mean test k-NN distance:      {knn_details['mean_test_distance']:.4f}")
print(f"Distance increase:             {knn_details['mean_test_distance'] - knn_details['mean_ref_distance']:.4f}")
print(f"P-value:                       {knn_details['p_val']:.6f}")

# %% [markdown]
# #### DriftMMD: multivariate distribution distance
#
# MMD measures the overall distance between two distributions in a kernel feature space. It captures both marginal and
# joint distributional changes that univariate tests might miss.

# %%
mmd_result = shifted_results["MMD"]
mmd_details = mmd_result.details

print(f"MMD² distance:   {mmd_result.distance:.6f}")
print(f"MMD² threshold:  {mmd_details['distance_threshold']:.6f}")
print(f"P-value:         {mmd_details['p_val']:.6f}")

# %% [markdown]
# Each detector reveals a different aspect of drift. The univariate detector pinpoints which features changed, the
# domain classifier shows which features matter most for distinguishing the distributions, the k-NN detector quantifies
# how far the data moved, and MMD provides a single multivariate distance between the distributions.

# %% [markdown]
# ### Choosing the right detector
#
# The best detector depends on what you need to know:
#
# - To see which features drifted, use {class}`.DriftUnivariate`. It provides per-feature p-values and drift flags.
# - To see why drift occurred, use {class}`.DriftDomainClassifier`. Its feature importances show what drives the shift.
# - For sensitivity to multivariate changes, use {class}`.DriftMMD`. It captures complex dependencies between features.
# - For fast, lightweight checks, use {class}`.DriftKNeighbors`. It is a simple distance comparison with minimal
#   overhead.
# - For more reliable detection, use several detectors with a majority vote to reduce false positives, as the check on
#   the held-out frames showed.

# %% [markdown]
# ### Monitor drift over time with chunking
#
# In real deployments, operational data arrives in batches over time. Rather than comparing all operational data at
# once, you can use **chunking** to split the data into segments and monitor how drift evolves across each chunk. This
# helps identify when drift begins to appear.
#
# DataEval's drift detectors support chunking through the `chunk_count` or `chunk_size` parameters on `fit()`. During
# fitting, the detector establishes a baseline by computing the metric across chunks of the reference data. During
# prediction, each chunk of test data is compared against this baseline, returning a {class}`.DriftOutput` with a
# `polars.DataFrame` in the `details` field containing per-chunk results.
#
# #### Simulate the second drone coming online
#
# To illustrate how chunking reveals when drift begins, you will build a stream in which the first 40% of the frames
# come from the reference drone and the remaining 60% come from the second drone. This simulates a deployment in which a
# new platform starts to contribute imagery partway through a monitoring window.

# %%
# Build a combined array: first the held-out frames from the reference drone, then the frames from the new drone
combined_embs = np.concatenate([operational_embs, shifted_embs])
print(
    f"Combined shape: {combined_embs.shape} (reference drone: {len(operational_embs)}, new drone: {len(shifted_embs)})"
)

# %% [markdown]
# #### Fit detectors with chunking

# %%
# Re-fit detectors with chunking enabled (5 chunks each)
chunked_detectors: dict[str, ChunkedDrift] = {
    "CVM": DriftUnivariate(method="cvm").chunked(chunk_count=5).fit(reference_embs),
    "MMD": DriftMMD().chunked(chunk_count=5).fit(reference_embs),
    "MVDC": DriftDomainClassifier(threshold=(0.45, 0.65)).chunked(chunk_count=5).fit(reference_embs),
    "KNN": DriftKNeighbors().chunked(chunk_count=5).fit(reference_embs),
}

# %% [markdown]
# #### Predict on combined data and plot chunk results

# %%
chunked_results = {}

for name, detector in chunked_detectors.items():
    result = detector.predict(combined_embs)
    chunked_results[name] = result
    print(f"\n{name} - Overall drift detected: {result.drifted} (metric: {result.metric_name})")
    if isinstance(result.details, pl.DataFrame):
        display(dep.plot(result, figsize=(8, 3)))

# %% [markdown]
# Each plot shows one detector. The gray points are the reference chunks, which the detector scored when you fitted it,
# and the blue points are the chunks of the stream. The green band is the range in which a chunk counts as no drift,
# and a red diamond marks a chunk that the detector flagged. The reference chunks are numbered back from -1, and the
# label of the axis gives the number of frames in a chunk. From the third chunk, where the frames of the new drone
# begin, every detector leaves the range.

# %% [markdown]
# #### Combine the chunk results with a majority vote
#
# The first two chunks hold only frames from the reference drone, and the last three hold frames from the new drone.
# Count how many detectors alert on each chunk, and report drift for a chunk when more than half of them do. Draw one
# row for each detector and a last row for the vote, and fill a cell where that detector, or the vote, reports drift.

# %%
names = list(chunked_results)
chunk_alerts = pl.DataFrame({name: result.details["drifted"] for name, result in chunked_results.items()})
chunk_votes = (
    chunk_alerts
    .with_columns(pl.sum_horizontal(names).alias("alerts"))
    .with_columns((pl.col("alerts") > len(names) / 2).alias("majority"))
    .with_row_index("chunk")
)

# Draw a slim grid of the detectors and the vote against the chunks, with drift in red
grid = chunk_votes.select(*names, pl.col("majority").alias("Majority vote"))
(
    px
    .imshow(
        grid.to_numpy().T.astype(int),
        x=[str(i) for i in range(len(grid))],
        y=grid.columns,
        color_continuous_scale=["#eef1f4", "#d9534f"],
        zmin=0,
        zmax=1,
        labels={"x": "Chunk"},
        aspect="auto",
        height=190,
    )
    .update_coloraxes(showscale=False)
    .update_traces(xgap=4, ygap=4)
    .update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0})
)

# %% [markdown]
# A single detector can alert on a chunk of reference frames, so one alert is not enough. The majority vote stays quiet
# until the third chunk, which is where the frames from the new drone begin. This chunk-level view makes it easy to
# pinpoint when in a data stream drift begins.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
from sklearn.decomposition import PCA

# Lock the claims this tutorial makes in prose, so a dependency or data change cannot
# silently invert them while the narrative keeps asserting the old result.

# The drones that the prose describes
drones = {row["drone"]: row for row in by_drone.iter_rows(named=True)}
mavic_row, trinity_row, m210_row = drones["mavic"], drones["trinity"], drones["m210"]
assert (mavic_row["frames"], mavic_row["min_width"], mavic_row["max_width"]) == (937, 3840, 3840)  # "All of its frames"
assert round(mavic_row["median_altitude"]) == 37  # "about 37 m"
assert (round(trinity_row["median_altitude"]), trinity_row["max_width"]) == (110, 5456)  # "about 110 m", "to 5456"
assert 1200 < trinity_row["min_width"] < 1300  # "about 1230"
assert trinity_row["frames_with_altitude"] < trinity_row["frames"] / 2  # "most trinity frames"
assert (m210_row["min_width"], m210_row["max_width"], round(m210_row["median_altitude"])) == (
    1920,
    3840,
    9,
)  # "On your own"

# The frames that the prose quotes
assert (len(reference_view), len(operational_view), len(trinity_view)) == (250, 100, 150)

# "Three of the four detectors agree, but `CVM` reports drift" and "the vote says that there is no drift"
assert {name: result.drifted for name, result in clean_results.items()} == {
    "CVM": True,
    "MMD": False,
    "MVDC": False,
    "KNN": False,
}

# "All four detectors detect drift, so the majority vote does too"
for name, result in shifted_results.items():
    assert result.drifted, f"{name} failed to detect drift on frames from the second drone"

# "every detector leaves the range" from the third chunk, with a gray reference segment of chunks counted back from -1
for name, result in chunked_results.items():
    assert all(result.details["drifted"][2:]), f"{name} did not flag every chunk from the new drone"
    assert len(result.baseline) == 5, name
    assert result.details["end_index"][0] - result.details["start_index"][0] + 1 == 50, name  # "frames in a chunk"

# "A single detector can alert on a chunk of reference frames" and "the majority vote stays quiet until the third chunk"
assert list(chunk_votes["majority"]) == [False, False, True, True, True]
assert max(chunk_votes["alerts"][:2]) == 1
assert min(chunk_votes["alerts"][2:]) >= 3

# "the detectors, which use all 512 dimensions"
assert reference_embs[0].shape == (512,)

# "The held-out frames land on top of the reference frames. The trinity frames gather in one region"
projected = PCA(n_components=2).fit_transform(all_embs)
centers = {name: projected[set_ids == i].mean(axis=0) for i, name in set_names.items()}
assert np.linalg.norm(centers["trinity"] - centers["reference"]) > 3 * np.linalg.norm(
    centers["held-out"] - centers["reference"]
)

# "The held-out frames match the reference", and the shares and counts that the prose quotes
boat, swimmer = class_names.index("boat"), class_names.index("swimmer")
assert np.abs(class_mix["held-out"] - class_mix["reference"]).max() < 0.05
assert [round(100 * class_mix[v][boat]) for v in ("reference", "trinity")] == [27, 46]  # "46% ... against 27%"
assert [round(100 * class_mix[v][swimmer]) for v in ("reference", "trinity")] == [57, 43]  # "43%, against 57%"
assert [round(objects_per_frame[v], 1) for v in ("reference", "trinity")] == [6.8, 4.2]  # "4.2, against 6.8"

# "On your own": the m210 frames are mostly swimmers
m210_labels = np.concatenate([
    np.asarray(ds[int(i)][1].labels) for i in np.flatnonzero(frames["drone"].to_numpy() == "m210")
])
assert np.mean(m210_labels == class_names.index("swimmer") + 1) > 0.9

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you created embeddings from SeaDrone imagery from two drones, compared drift detectors and their
# outputs, combined them with a majority vote, and used chunked monitoring to identify when drift begins.
#
# You learned these points:
#
# - One detector can raise a false alarm, so combine several detectors with a majority vote
# - Embedding detectors show covariate shift, so check the class mix for label shift as well
# - Embedding detectors cannot see concept drift or the effect of drift on a model, which take labels or model outputs
# - `DriftUnivariate` reveals which features drifted through per-feature statistical tests
# - `DriftDomainClassifier` explains why drift occurred through feature importances
# - `DriftMMD` provides a single multivariate distance that captures complex distributional changes
# - `DriftKNeighbors` offers fast, lightweight detection based on distance comparisons
# - Chunked monitoring helps pinpoint when drift begins in a data stream
#
# Drift can affect the ability of a model to achieve the performance recorded during model training. When you detect
# data drift, consider retraining the model and incorporating operational data into the dataset.

# %% [markdown]
# ## Next steps
#
# - [Distribution shift](../concepts/DistributionShift.md) — Read about covariate shift, concept drift, and label parity
#   in deployed ML models.
# - [Identify out-of-distribution samples](./tt_identify_ood_samples.py) — Compare how many images autoencoder and k-NN
#   detectors flag as unlike the training data.
# - [Measure label independence](./h2_measure_label_independence.py) — Compare label distributions between two
#   datasets to assess class representation.

# %% [markdown]
# ## On your own
#
# Once you are familiar with DataEval and data monitoring, run this analysis using your own reference and operational
# datasets.
#
# Experiment with these changes:
#
# - Try different embeddings for KNN: ResNet, ViT, CLIP, or domain-specific pretrained models
# - Design custom architectures for your specific data type, and not generic examples
# - Test different drift scenarios on your own data, with varying difficulty levels
# - Change the operational drone. Use `m210`, which has a median altitude of about 9 m and frames 1920 or 3840 pixels
#   wide, as the operational data, and compare how many detectors report drift. Then compare its class mix, which is
#   mostly swimmers.
# - Define the reference and the operational frames by altitude and not by drone, and see whether the detectors still
#   agree.
