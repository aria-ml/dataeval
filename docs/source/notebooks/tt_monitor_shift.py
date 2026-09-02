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
# This guide provides a beginner friendly introduction on monitoring post deployment data shifts.
#
# Estimated time to complete: 5 minutes
#
# Relevant ML stages: [Monitoring](../getting-started/roles/ML_Lifecycle.md#monitoring)
#
# Relevant personas: [ML Engineer](../getting-started/roles/ml_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)
#
# ## What you'll do
#
# - Narrow a dataset to the classes your deployment monitors with a dataset view
# - Construct [embeddings](../concepts/Embeddings.md) by inferencing images through a pretrained network
# - Compare different drift detectors and understand their strengths
# - Inspect detector-specific outputs for root-cause analysis
# - Use chunked drift detection to monitor drift across data segments
# - Compare the label distributions between a training and operational set, and see the check fail on a broken one
#
# ## What you'll learn
#
# - Learn the strengths and trade-offs of each drift detector
# - Learn how to analyze embeddings for operational drift
# - Learn that an embedding encodes the whole image, not only the classes you kept
# - Learn how to inspect per-feature and per-detector statistics
# - Learn how to use chunked drift detection for temporal monitoring
# - Learn how to analyze label distributions, and what a parity failure looks like
#
# ## What you'll need
#
# - Knowledge of Python
# - Beginner knowledge of PyTorch or neural networks

# %% [markdown]
# ## Introduction
#
# Monitoring is a critical step in the [AI/ML lifecycle](../getting-started/roles/ML_Lifecycle.md). When a model is
# deployed, data can, and generally will, [drift](../concepts/DistributionShift.md) from the distribution on which the
# model was originally trained. One critical step in AI T&E is the detection of changes in the operational distribution so
# that they may be proactively addressed. While some change might not affect performance, significant deviation is often
# associated with model degradation.
#
# For this tutorial, you will use the popular
# [2012 VOC](https://huggingface.co/datasets/HuggingFaceM4/pascal_voc/tree/main) computer vision dataset to detect drift
# between the image distribution of the `train` split and the `val` split, which will represent an operational dataset in
# this guide. Both splits are narrowed to the indoor concepts a deployed model would actually see, mirroring how you
# monitor the slice your model serves rather than an entire benchmark. You will then determine if the labels within
# these two datasets has high [parity](../concepts/DistributionShift.md#label-parity), or equivalent label
# distributions.

# %% [markdown]
# ## Setup
#
# You'll begin by importing the necessary libraries for this tutorial.

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

from dataeval import Embeddings, Metadata
from dataeval.config import set_device, set_seed
from dataeval.core import label_parity
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
# > **More on device**
# >
# > The device is set above as it will be used in subsequent steps. The device is the piece of hardware where the model,
# > data, and other related objects are stored in memory. If a GPU is available, this notebook will use that hardware
# > rather than the CPU. To force running only on the CPU, change `device` to `"cpu"` For more information, see the
# > [PyTorch device page](https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html).

# %% [markdown]
# ## Construct embeddings
#
# Rather than compare raw pixels, you'll work in [embedding](../concepts/Embeddings.md) space: a compact feature vector
# per image that keeps drift detection fast and memory-light without sacrificing accuracy.
#
# ### Download VOC dataset
#
# Download the `train` and `val` splits of the 2012 VOC dataset — `val` stands in for the operational data you'd see
# after deployment.

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
# > **The view narrows labels, not pixels**
# >
# > A kept image is one that *contains* indoor furniture. Most of them also contain people, bottles, or pets, and
# > those objects are still in the frame. Embeddings are computed from the whole image, so each vector encodes the
# > entire scene rather than only the four classes you kept. Keep this in mind whenever you monitor embeddings: the
# > vector reflects everything the camera saw, not just what you labeled.
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
# Torchvision: point a {class}`.TorchExtractor` at its penultimate `avgpool` layer (its learned features, not the
# 1000-class logits) and run each split through {class}`.Embeddings`, which applies the model's standard
# resize-and-normalize preprocessing for you. Keeping the `transforms` in a variable lets you reuse them further down.
#
# > For other ways to build embeddings, see [Encode images with an ONNX model](./h2_encode_with_onnx.py) (a
# > framework-agnostic ONNX extractor) and [Embed object detection crops](./h2_embed_detection_crops.py) (one embedding
# > per object box).

# %%
resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
transforms = ResNet18_Weights.DEFAULT.transforms()
extractor = TorchExtractor(resnet, transforms=transforms, layer_name="avgpool")

# Create embeddings for the train and operational splits
train_embs = Embeddings(train_view, extractor=extractor, batch_size=64)
operational_embs = Embeddings(operational_view, extractor=extractor, batch_size=64)

# %% [markdown]
# Each image is now a single feature vector instead of a full-resolution image, which speeds up the drift algorithms
# below without impacting the accuracy of the results.

# %%
print(f"({len(train_embs)}, {train_embs[0].shape})")  # (1163, shape)
print(f"({len(operational_embs)}, {operational_embs[0].shape})")  # (1173, shape)

# %% [markdown]
# ## Understanding drift detectors
#
# Before testing for drift, it helps to understand the different approaches available. Each detector has distinct
# strengths that make it better suited for certain scenarios. This tutorial uses four detectors that represent
# fundamentally different strategies for detecting distributional change.
#
# | Detector                        | Approach                                         | Strengths                                   | Best For                              |
# | ------------------------------- | ------------------------------------------------ | ------------------------------------------- | ------------------------------------- |
# | {class}`.DriftUnivariate` (CVM) | Statistical test per feature                     | Fast, interpretable per-feature results     | Identifying _which_ features drifted  |
# | {class}`.DriftMMD`              | Kernel-based multivariate test                   | Captures feature dependencies               | High-dimensional data, complex shifts |
# | {class}`.DriftDomainClassifier` | Trains a classifier to distinguish distributions | Feature importances for root-cause analysis | Understanding _why_ drift occurred    |
# | {class}`.DriftKNeighbors`       | Compares k-NN distances                          | Lightweight and fast                        | Quick monitoring checks               |
#
# > **Other univariate methods**
# >
# > `DriftUnivariate` supports several statistical tests beyond CVM, including Kolmogorov-Smirnov (`ks`), Mann-Whitney U
# > (`mwu`), Anderson-Darling (`anderson`), and Baumgartner-Weiss-Schindler (`bws`). Each has different sensitivity
# > characteristics — see the [drift concept page](../concepts/DistributionShift.md#drift-detection) for details.

# %% [markdown]
# ## Test for drift
#
# In this step, you will be checking for drift between the training embeddings and the operational embeddings from before.
# If drift is detected, a model trained on this training data should be retrained with new operational data. This can help
# mitigate performance degradation in a deployed model. Visit our [About Drift](../concepts/DistributionShift.md) page to
# learn more.
#
# ### Drift detectors
#
# DataEval offers several drift detectors. This tutorial demonstrates four that each take a different approach:
# {class}`.DriftUnivariate`, {class}`.DriftMMD`, {class}`.DriftDomainClassifier`, and {class}`.DriftKNeighbors`.
#
# Since each detector outputs a binary decision on whether drift is detected, a **majority vote** can be used to make the
# determination of drift.\
# To learn more about these algorithms, see the
# [theory behind drift detection](../concepts/DistributionShift.md#taxonomy-of-shift) concept page.
#
# ### Fit the detectors
#
# Each drift detector needs a reference set that the operational set will be compared against. In the following code, you
# will set the reference data to the training embeddings.

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
# Now that the detectors are setup, predictions can be made against the operational embeddings you made earlier.

# %%
# Iterate and print the name of the detector class and its boolean drift prediction
clean_results = {name: detector.predict(operational_embs) for name, detector in detectors.items()}

print("\n".join(f"{res[0]} detected drift? {res[1].drifted}" for res in clean_results.items()))

# %% [markdown]
# Did you expect these results?
#
# There is no drift detected between the train and operational embeddings because they come from very similar
# distributions.\
# Ideally, your training data and your validation data, which we used as operational, come from the same distribution.
# This is the purpose of [data splitters](https://scikit-learn.org/stable/api/sklearn.model_selection.html#splitters).
#
# So how do we know if the detectors can detect drift?
#
# Well, add some random Gaussian noise to the operational embeddings and find out.

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
# There are many [types of drift](../concepts/DistributionShift.md#taxonomy-of-shift) that data can and will experience.

# %% [markdown]
# ### Inspecting detector outputs
#
# Each detector doesn't just report whether drift occurred — it provides statistics that reveal _different things_ about
# the drift. Let's look at what each detector tells us.

# %%
# Store results for inspection
results = {name: detector.predict(noisy_embs) for name, detector in detectors.items()}

# %% [markdown]
# #### DriftUnivariate: per-feature analysis
#
# The univariate detector tests each feature independently and reports which features drifted and their p-values. This is
# useful for identifying _which_ dimensions of the embedding space shifted.

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
# The domain classifier trains a model to distinguish reference from test data and reports how important each feature was
# in making that distinction. High AUROC means the distributions are easily separable — a strong signal of drift.

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
# Each detector reveals a different facet of drift: the univariate detector pinpoints _which_ features changed, the domain
# classifier shows _which features matter most_ for distinguishing the distributions, the k-NN detector quantifies _how
# far_ the data moved, and MMD provides a single _multivariate distance_ between the distributions.

# %% [markdown]
# ### Choosing the right detector
#
# The best detector depends on what you need to know:
#
# - **Which features drifted?** Use {class}`.DriftUnivariate` — it provides per-feature p-values and drift flags
# - **Why did drift occur?** Use {class}`.DriftDomainClassifier` — its feature importances show what drives the shift
# - **How sensitive to multivariate changes?** Use {class}`.DriftMMD` — it captures complex dependencies between features
# - **Need fast, lightweight checks?** Use {class}`.DriftKNeighbors` — simple distance comparison with minimal overhead
# - **Want robust detection?** Use multiple detectors with a **majority vote** to reduce false positives

# %% [markdown]
# ### Monitor drift over time with chunking
#
# In real deployments, operational data arrives in batches over time. Rather than comparing all operational data at once,
# you can use **chunking** to split the data into segments and monitor how drift evolves across each chunk. This helps
# identify _when_ drift begins to appear.
#
# DataEval's drift detectors support chunking through the `chunk_count` or `chunk_size` parameters on `fit()`. During
# fitting, the detector establishes a baseline by computing the metric across chunks of the reference data. During
# prediction, each chunk of test data is compared against this baseline, returning a {class}`.DriftOutput` with a
# `polars.DataFrame` in the `details` field containing per-chunk results.
#
# #### Simulate gradual drift onset
#
# To illustrate how chunking reveals _when_ drift begins, you will build a combined dataset where the first 40% of samples
# are clean operational embeddings and the remaining 60% are noisy. This simulates a scenario where data quality degrades
# partway through a monitoring window.

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
# should trigger drift alerts. This chunk-level view makes it easy to pinpoint _when_ in a data stream drift begins.
#
# Next you will look at the labels' distributions.

# %% [markdown]
# ## Evaluate parity

# %% [markdown]
# Instead of looking at the images, you can compare the distributions of the labels using a method called
# [label parity](../concepts/DistributionShift.md#label-parity).\
# There is parity between two sets of labels if the label frequencies are approximately equal.
#
# You will now compare the label distributions using the `label_parity` function.

# %%
# Get the metadata for each view
train_md = Metadata(train_view)
operational_md = Metadata(operational_view)

# The views expose the four monitored classes
label_parity(train_md.class_labels, operational_md.class_labels, num_classes=len(FURNITURE))["p_value"]

# %% [markdown]
# From the {func}`.label_parity` function, you can see that it calculated a p_value of ~**0.96**. Since this is close to
# 1.0, it can be said that the two distributions **have** class label parity, or similar distributions.

# %% [markdown]
# ### What a parity failure looks like
#
# A passing check is hard to interpret on its own, so it helps to see the alarm actually go off. You will build a
# deliberately broken operational set and re-run the same check.
#
# Imagine an upstream annotation policy change: partway through deployment, the labeling team decides `sofa` is close
# enough to `chair` and stops distinguishing them. Another {class}`.Relabel` expresses exactly that — two source
# classes collapsing into one target concept.
#
# The important part is what does **not** change. Every image is still in the set and every box is still annotated;
# only the names attached to them move. The drift detectors from earlier in this tutorial would see nothing at all
# here, because the pixels feeding the embeddings are identical. Label parity is what catches this class of problem.

# %%
# An upstream policy stops distinguishing sofas from chairs -- the images are untouched
policy_change = Relabel(
    {"chair": "chair", "diningtable": "diningtable", "sofa": "chair", "tvmonitor": "tvmonitor"},
    FURNITURE,
)

# Nest the change on the operational view rather than rebuilding it from the raw split
poor_parity_view = View(operational_view, operations=[policy_change])
poor_parity_md = Metadata(poor_parity_view)

# Gather the counts of each label in the training and broken operational sets
train_label_counts = np.bincount(np.asarray(train_md.class_labels), minlength=len(FURNITURE))
poor_parity_label_counts = np.bincount(np.asarray(poor_parity_md.class_labels), minlength=len(FURNITURE))

print(f"images:                   {len(operational_view)} -> {len(poor_parity_view)} (unchanged)")
print(f"train label counts:       {train_label_counts}")
print(f"poor parity label counts: {poor_parity_label_counts}")

# %%
label_parity(train_md.class_labels, poor_parity_md.class_labels, num_classes=len(FURNITURE))["p_value"]

# %% [markdown]
# The p_value is now effectively **zero**. Every `sofa` has been counted as a `chair`, so one class is empty while
# another is inflated, and the two distributions **lack** parity.
#
# In a real deployment this is the signal to go looking for a cause before retraining on the operational data: an
# annotation policy that changed underneath you, a class your model never sees anymore, or an operational mix that has
# genuinely shifted. Each needs a different response, and each would leave the images themselves looking perfectly
# normal.

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

# "a p_value of ~0.96 ... close to 1.0"
parity_p_value = label_parity(train_md.class_labels, operational_md.class_labels, num_classes=len(FURNITURE))["p_value"]
assert parity_p_value > 0.9, f"label parity dropped to {parity_p_value}"

# "The p_value is now effectively zero" -- and the contrived set must change labels only
assert len(poor_parity_view) == len(operational_view), "the policy change should not drop images"
assert len(poor_parity_md.class_labels) == len(operational_md.class_labels), "it should not drop annotations"
poor_parity_result = label_parity(train_md.class_labels, poor_parity_md.class_labels, num_classes=len(FURNITURE))
poor_parity_p_value = poor_parity_result["p_value"]
assert poor_parity_p_value < 0.01, f"contrived poor-parity set no longer trips the check: {poor_parity_p_value}"

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you have learned to create embeddings from the VOC dataset, compare different drift detectors and
# their unique outputs, use chunked monitoring to identify when drift begins, and calculate the parity of label
# distributions for both a healthy and a deliberately broken operational set.
#
# Key takeaways:
#
# - **DriftUnivariate** reveals _which_ features drifted through per-feature statistical tests
# - **DriftDomainClassifier** explains _why_ drift occurred through feature importances
# - **DriftMMD** provides a single multivariate distance that captures complex distributional changes
# - **DriftKNeighbors** offers fast, lightweight detection based on distance comparisons
# - **Chunked monitoring** helps pinpoint _when_ drift begins in a data stream
# - **Label parity** catches shifts the image detectors cannot see — the pixels can be identical while the labels move
#
# These are important steps when monitoring data, as drift and lack of parity can affect a model's ability to achieve
# performance recorded during model training. When data drift is detected or the label distributions lack parity, it is a
# good idea to consider retraining the model and incorporating operational data into the dataset.

# %% [markdown]
# ## Next steps
#
# - [Distribution shift concepts](../concepts/DistributionShift.md) — Read about covariate shift, concept drift, and label parity in deployed ML models.
# - [Identify out-of-distribution samples tutorial](./tt_identify_ood_samples.py) — Detect anomalous and out-of-distribution samples using autoencoders and k-NN.
# - [Display data distributions](./h2_measure_divergence.py) — Measure statistical divergence and plot distribution overlays between reference and target datasets.
# - [Compare label distributions](./h2_measure_label_independence.py) — Compare label distributions between two datasets to assess class representation.
# - [Configure global DataEval defaults](./h2_configure_defaults.py) — Set global default configuration parameters like batch size and execution device.
# - [Build dataset views](./h2_build_dataset_views.py) — Filter, reshape, and transform datasets without copying underlying media files.
# - [Encode with an ONNX model](./h2_encode_with_onnx.py) — Extract image embeddings and predictions using an ONNX runtime inference session.
# - [Embed object detection crops](./h2_embed_detection_crops.py) — Extract feature embeddings for individual bounding box targets in detection datasets.

# %% [markdown]
# ## On your own
#
# Once you are familiar with DataEval and data monitoring, run this analysis using your own reference and operational
# datasets.
#
# Experiment with:
#
# - **Different embeddings for KNN**: ResNet, ViT, CLIP, or domain-specific pretrained models
# - **Custom architectures**: Design models for your specific data type (not generic examples)
# - **Different drift scenarios**: Test on your own data with varying difficulty levels
# - **Wider operational domains**: This tutorial monitors a narrow slice, which keeps the reference distribution tight
#   and makes injected noise easy to spot. Expect weaker, noisier signals as the monitored domain broadens — drop the
#   view above to run the same analysis across all 20 VOC classes and compare.
