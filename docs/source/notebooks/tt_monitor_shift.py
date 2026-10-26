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
# Learn the basics of monitoring data shifts after deployment.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Monitoring](../getting-started/roles/ML_Lifecycle.md#monitoring)
#
# Relevant personas: [ML Engineer](../getting-started/roles/ml_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)
#
# ## What you will do
#
# - Narrow a dataset to the classes your deployment monitors with a dataset view
# - Construct [embeddings](../concepts/Embeddings.md) by inferencing images through a pretrained network
# - Compare different drift detectors and understand their strengths
# - Inspect detector-specific outputs for root-cause analysis
# - Use chunked drift detection to monitor drift across data segments
#
# ## What you will learn
#
# - Learn the strengths and trade-offs of each drift detector
# - Learn how to analyze embeddings for operational drift
# - Learn that an embedding encodes the whole image, not only the classes you kept
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
# In this tutorial, you will use the [2012 VOC](https://huggingface.co/datasets/HuggingFaceM4/pascal_voc/tree/main)
# computer vision dataset to detect drift between the image distribution of the `train` split and the `val` split, which
# represents an operational dataset. Both splits are narrowed to the indoor concepts that a deployed model would see,
# which mirrors how you monitor the slice your model serves and not an entire benchmark.

# %% [markdown]
# ## Setup
#
# Begin by importing the libraries that this tutorial uses.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
import numpy as np
import polars as pl
import torch
from IPython.display import display
from maite_datasets.object_detection import VOCDetection
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.v2 import GaussianNoise

from dataeval import Embeddings
from dataeval.config import set_device, set_seed
from dataeval.data import Relabel, View
from dataeval.extractors import TorchExtractor
from dataeval.shift import ChunkedDrift, DriftDomainClassifier, DriftKNeighbors, DriftMMD, DriftUnivariate

# Set a random seed
rng = np.random.default_rng(213)

# Set default device for notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_device(device)

# Seed NumPy and torch so the added Gaussian noise (and drift detectors) are reproducible
set_seed(213, all_generators=True)

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
# ### Download VOC dataset
#
# Download the `train` and `val` splits of the 2012 VOC dataset. The `val` split stands in for the operational data that
# you would see after deployment.

# %%
# Load the training dataset
train_ds = VOCDetection("./data", year="2012", image_set="train", download=True)
print(train_ds)
print(f"Image 0 shape: {train_ds[0][0].shape}")

# %%
# Load the "operational" dataset
operational_ds = VOCDetection("./data", year="2012", image_set="val", download=True)
print(operational_ds)
print(f"Image 0 shape: {operational_ds[0][0].shape}")

# %% [markdown]
# ### Narrow both splits to the monitored classes
#
# A deployed model rarely sees an entire benchmark. It sees the slice its operating environment contains, and that
# slice is what you monitor. Here you will narrow both splits to four indoor concepts with a {class}`.View`, which
# wraps a dataset without copying it. Working on a smaller, focused slice also keeps this tutorial quick to run.
#
# {class}`.Relabel` conforms a dataset to a target vocabulary. The four concepts below are kept, every other VOC class
# is out-of-vocabulary and gets dropped, and any image left with no labels drops along with them.
#
# :::{note}
# The view narrows the labels, not the pixels. A kept image is one that contains indoor furniture. Most of them also
# contain people, bottles, or pets, and those objects are still in the frame. Embeddings are computed from the whole
# image, so each vector encodes the entire scene and not only the four classes you kept. Keep this in mind whenever you
# monitor embeddings: the vector reflects everything the camera saw, and not only what you labeled.
# :::
#
# > For other ways to reshape a dataset in place, see [Build dataset views](./h2_build_dataset_views.py).

# %%
# The indoor concepts this deployment monitors
FURNITURE = ("chair", "diningtable", "sofa", "tvmonitor")

# Keep these four concepts as-is; every other VOC class is dropped as out-of-vocabulary
furniture_only = Relabel({name: name for name in FURNITURE}, FURNITURE)

train_view = View(train_ds, operations=[furniture_only])
operational_view = View(operational_ds, operations=[furniture_only])

print(f"train:       {len(train_ds)} -> {len(train_view)} images")
print(f"operational: {len(operational_ds)} -> {len(operational_view)} images")
print(f"vocabulary:  {train_view.metadata.get('index2label')}")

# %% [markdown]
# ### Extract embeddings
#
# Reduce each image to a feature vector with a
# [pretrained ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) from
# Torchvision. Point a {class}`.TorchExtractor` at its penultimate `avgpool` layer (its learned features, not the
# 1000-class logits), and give it the standard resize-and-normalize preprocessing of the model as `transforms`. Then run
# each split through {class}`.Embeddings`. Keeping the `transforms` in a variable lets you reuse them further down.
#
# For other ways to build embeddings, see [How to create embeddings](./h2_create_embeddings.py).

# %%
resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
transforms = ResNet18_Weights.DEFAULT.transforms()
extractor = TorchExtractor(resnet, transforms=transforms, layer_name="avgpool")

# Create embeddings for the train and operational splits
train_embs = Embeddings(train_view, extractor=extractor, batch_size=64)
operational_embs = Embeddings(operational_view, extractor=extractor, batch_size=64)

# %% [markdown]
# Each image is now a single feature vector instead of a full-resolution image, which speeds up the drift algorithms
# below without affecting the accuracy of the results.

# %%
print(f"({len(train_embs)}, {train_embs[0].shape})")  # (1163, shape)
print(f"({len(operational_embs)}, {operational_embs[0].shape})")  # (1173, shape)

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
# In this step, you will check for drift between the training embeddings and the operational embeddings from before. If
# drift is detected, retrain a model that was trained on this training data with new operational data. This can help
# mitigate performance degradation in a deployed model. See the [About Drift](../concepts/DistributionShift.md) page to
# learn more.
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
# you will set the reference data to the training embeddings.

# %%
# A type alias for all of the drift detectors
DriftDetector = DriftUnivariate | DriftMMD | DriftDomainClassifier | DriftKNeighbors

# Create a mapping for the detectors to iterate over
detectors: dict[str, DriftDetector] = {
    "CVM": DriftUnivariate(method="cvm").fit(train_embs),
    "MMD": DriftMMD().fit(train_embs),
    "MVDC": DriftDomainClassifier().fit(train_embs),
    "KNN": DriftKNeighbors().fit(train_embs),
}

# %% [markdown]
# ### Make predictions
#
# Now that the detectors are set up, you can make predictions against the operational embeddings you made earlier.

# %%
# Iterate and print the name of the detector class and its boolean drift prediction
clean_results = {name: detector.predict(operational_embs) for name, detector in detectors.items()}

print("\n".join(f"{res[0]} detected drift? {res[1].drifted}" for res in clean_results.items()))

# %% [markdown]
# There is no drift detected between the train and operational embeddings because they come from very similar
# distributions.\
# Ideally, your training data and your validation data, which this tutorial uses as operational data, come from the same
# distribution. This is the purpose of
# [data splitters](https://scikit-learn.org/stable/api/sklearn.model_selection.html#splitters).
#
# To check that the detectors can detect drift, add some random Gaussian noise to the operational embeddings.

# %%
# Define transform with added gaussian noise
noisy_transforms = [transforms, GaussianNoise()]

# Create extractor with noisy transforms
noisy_extractor = TorchExtractor(resnet, transforms=noisy_transforms, layer_name="avgpool")

# Applies gaussian noise to images before processing
noisy_embs = Embeddings(operational_view, extractor=noisy_extractor, batch_size=64)

# %%
# Iterate and print the name of the detector class and its boolean drift prediction
print("\n".join(f"{det[0]} detected drift? {det[1].predict(noisy_embs).drifted}" for det in detectors.items()))

# %% [markdown]
# Now drift is detected!
#
# Adding Gaussian noise was enough to cause a noticeable change in the drift detectors, but this is not always the case.
# Data can experience many [types of drift](../concepts/DistributionShift.md#taxonomy-of-shift).

# %% [markdown]
# ### Inspecting detector outputs
#
# Each detector reports whether drift occurred and also provides statistics that reveal different things about the
# drift. The next cells show what each detector tells you.

# %%
# Store results for inspection
results = {name: detector.predict(noisy_embs) for name, detector in detectors.items()}

# %% [markdown]
# #### DriftUnivariate: per-feature analysis
#
# The univariate detector tests each feature independently and reports which features drifted and their p-values. This
# is useful for identifying which dimensions of the embedding space shifted.

# %%
cvm_result = results["CVM"]
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
mvdc_result = results["MVDC"]
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
knn_result = results["KNN"]
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
mmd_result = results["MMD"]
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
# - For more reliable detection, use several detectors with a majority vote to reduce false positives.

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
# #### Simulate gradual drift onset
#
# To illustrate how chunking reveals when drift begins, you will build a combined dataset where the first 40% of samples
# are clean operational embeddings and the remaining 60% are noisy. This simulates a scenario where data quality
# degrades partway through a monitoring window.

# %%
# Build a combined array: first 40% clean, last 60% noisy
n_operational = len(operational_embs)
split_idx = int(n_operational * 0.4)

combined_embs = np.concatenate([operational_embs[:split_idx], noisy_embs[split_idx:]])
print(f"Combined shape: {combined_embs.shape} (clean: {split_idx}, noisy: {n_operational - split_idx})")

# %% [markdown]
# #### Fit detectors with chunking

# %%
# Re-fit detectors with chunking enabled (5 chunks each)
chunked_detectors: dict[str, ChunkedDrift] = {
    "CVM": DriftUnivariate(method="cvm").chunked(chunk_count=5).fit(train_embs),
    "MMD": DriftMMD().chunked(chunk_count=5).fit(train_embs),
    "MVDC": DriftDomainClassifier(threshold=(0.45, 0.65)).chunked(chunk_count=5).fit(train_embs),
    "KNN": DriftKNeighbors().chunked(chunk_count=5).fit(train_embs),
}

# %% [markdown]
# #### Predict on combined data and display chunk results

# %%
chunked_results = {}

for name, detector in chunked_detectors.items():
    result = detector.predict(combined_embs)
    chunked_results[name] = result
    print(f"\n{name} - Overall drift detected: {result.drifted} (metric: {result.metric_name})")
    if isinstance(result.details, pl.DataFrame):
        display(result.details)

# %% [markdown]
# The first two chunks (covering the clean 40%) should show no drift, while the later chunks (covering the noisy 60%)
# should trigger drift alerts. This chunk-level view makes it easy to pinpoint when in a data stream drift begins.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Lock the claims this tutorial makes in prose, so a dependency or data change cannot
# silently invert them while the narrative keeps asserting the old result.

# The monitored slice is the size the prose and inline comments quote
assert len(train_view) == 1163, f"train view size changed: {len(train_view)}"
assert len(operational_view) == 1173, f"operational view size changed: {len(operational_view)}"
assert tuple(train_view.metadata.get("index2label", {}).values()) == FURNITURE

# "There is no drift detected between the train and operational embeddings"
for name, result in clean_results.items():
    assert not result.drifted, f"{name} reported drift on clean operational data"

# "Now drift is detected!" -- every detector must fire once noise is added
for name, result in results.items():
    assert result.drifted, f"{name} failed to detect drift on noisy data"

# "The first two chunks (covering the clean 40%) should show no drift, while the
# later chunks (covering the noisy 60%) should trigger drift alerts."
for name, result in chunked_results.items():
    assert list(result.details["drifted"]) == [False, False, True, True, True], (
        f"{name} chunk pattern changed: {list(result.details['drifted'])}"
    )

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you created embeddings from the VOC dataset, compared drift detectors and their outputs, and used
# chunked monitoring to identify when drift begins.
#
# You learned these points:
#
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
# - [Distribution Shift](../concepts/DistributionShift.md) — Read about covariate shift, concept drift, and label parity
#   in deployed ML models.
# - [Identify out-of-distribution samples](./tt_identify_ood_samples.py) — Compare how many images autoencoder and k-NN
#   detectors flag as unlike the training data.
# - [How to measure label independence](./h2_measure_label_independence.py) — Compare label distributions between two
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
# - Widen the operational domain. This tutorial monitors a narrow slice, which keeps the reference distribution tight
#   and makes injected noise easy to spot. Expect weaker, noisier signals as the monitored domain broadens. Drop the
#   view above to run the same analysis across all 20 VOC classes and compare.
