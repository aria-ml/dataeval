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
# # How to detect distribution shift from precomputed model outputs
#
# In many production monitoring environments you have no access to the model, its weights or the original training data, only
# logs of what the model output over time: class probabilities, logits, embeddings or hard prediction labels. DataEval's
# drift detectors accept raw arrays, so you can pass those outputs to `.fit()` and `.predict()` without wrapping a model in a
# `FeatureExtractor`.
#
# Use this guide when you want to detect different types of shift from precomputed predictions alone.
#
# You need `dataeval`. No extra dependencies, such as `torch` or `maite-datasets`, are required.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots[plotly] maite-datasets
except Exception:
    pass

# %%
import numpy as np
from scipy.stats import entropy

from dataeval.config import set_seed
from dataeval.shift import DriftDomainClassifier, DriftUnivariate

# Set random seed for reproducibility
set_seed(0, all_generators=True)
rng = np.random.default_rng(0)

# %% [markdown]
# ## Detect confidence drift with entropy
#
# If a deployed model encounters data outside its training distribution, its predictions typically become
# less confident (more uniform). You can calculate the {term}`Shannon entropy<Shannon Entropy>` of the predicted
# probabilities and test for an upward shift in that distribution.
#
# First, simulate some historical probabilities (confident, peaked distributions) and some drifted probabilities (less
# confident, flatter distributions).

# %%
num_samples = 1000
num_classes = 5

# Baseline: Model is confident (one class dominates)
prev_logits = rng.normal(loc=0, scale=1.0, size=(num_samples, num_classes))
prev_logits[:, 0] += 3.0  # Make class 0 the dominant, confident prediction
prev_probs = np.exp(prev_logits) / np.sum(np.exp(prev_logits), axis=1, keepdims=True)

# Drifted: Model is confused (probabilities are closer to uniform)
curr_logits = rng.normal(loc=0, scale=1.0, size=(num_samples, num_classes))
curr_probs = np.exp(curr_logits) / np.sum(np.exp(curr_logits), axis=1, keepdims=True)

# %% [markdown]
# Now convert the probabilities into one normalized entropy value per prediction with `scipy.stats.entropy`, and use
# {class}`.DriftUnivariate` to check whether the new entropy distribution differs significantly from the baseline.

# %%
# Divide by the maximum entropy so that 1.0 means a uniform prediction
prev_entropy = np.asarray(entropy(prev_probs, axis=1))[:, None] / np.log(num_classes)
curr_entropy = np.asarray(entropy(curr_probs, axis=1))[:, None] / np.log(num_classes)

# Use Kolmogorov-Smirnov (ks) test
detector = DriftUnivariate(method="ks").fit(prev_entropy)
result = detector.predict(curr_entropy)

print(f"Confidence Shift Detected: {result.drifted}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert result.drifted

# %% [markdown]
# ## Detect marginal class probability shift
#
# Sometimes the model remains confident, but the frequency with which it predicts certain classes changes.
# You can pass the raw probability arrays `(N, C)` directly into the univariate detector. It will
# apply the statistical test independently to each class's probability distribution and automatically apply a
# multiple-testing correction (Bonferroni by default) to control false positives.

# %%
# Simulate a shift where class 2 becomes the dominant prediction instead of class 0
drifted_class_logits = rng.normal(loc=0, scale=1.0, size=(num_samples, num_classes))
drifted_class_logits[:, 2] += 3.0
curr_probs_marginal = np.exp(drifted_class_logits) / np.sum(np.exp(drifted_class_logits), axis=1, keepdims=True)

# Use Cramér-von Mises (cvm), which is highly sensitive to overall distributional differences
detector_marginal = DriftUnivariate(method="cvm").fit(prev_probs)
result_marginal = detector_marginal.predict(curr_probs_marginal)

print(f"Marginal Shift Detected: {result_marginal.drifted}")
drifted_classes = np.where(result_marginal.details["feature_drift"])[0]
print(f"Which classes drifted? {drifted_classes.tolist()}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert result_marginal.drifted
assert result_marginal.details["feature_drift"].any()

# %% [markdown]
# ## Detect joint probability shift
#
# If the relationships or correlations between the predicted classes shift, univariate tests might miss it.
# The {class}`.DriftDomainClassifier` treats the model outputs as features and trains a LightGBM classifier
# to separate the baseline predictions from the incoming predictions. If the classifier achieves a high AUROC
# (e.g., > 0.55), the distributions are distinct enough to flag drift.

# %%
detector_multivariate = DriftDomainClassifier().fit(prev_probs)
result_multivariate = detector_multivariate.predict(curr_probs_marginal)

print(f"Multivariate Shift Detected: {result_multivariate.drifted} (AUROC: {result_multivariate.distance:.4f})")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert result_multivariate.drifted

# %% [markdown]
# ## Detect label shift from hard predictions
#
# If you do not have probabilities, only discrete integer labels output by the model, you can still detect label shift.
# One-hot encode the predictions to create an `(N, C)` array where each row has a single `1` and the rest are `0`. The
# univariate detector will then compare the predicted class proportions.

# %%
# Convert our probabilities to hard predictions (argmax)
prev_labels = np.argmax(prev_probs, axis=1)
curr_labels = np.argmax(curr_probs_marginal, axis=1)

# One-hot encode
prev_onehot = np.eye(num_classes)[prev_labels].astype(np.float32)
curr_onehot = np.eye(num_classes)[curr_labels].astype(np.float32)

detector_label = DriftUnivariate(method="cvm").fit(prev_onehot)
result_label = detector_label.predict(curr_onehot)

print(f"Label Shift Detected: {result_label.drifted}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert result_label.drifted

# %% [markdown]
# ## Next steps
#
# - [Distribution Shift](../concepts/DistributionShift.md) — Explore types of distribution shift and strategies for detecting drift.
# - [Monitor shifts in operational data](./tt_monitor_shift.py) — Track dataset shift and covariate drift in operational streams over time.
# - [How to detect uncertainty drift with a MAITE model](./h2_detect_uncertainty_drift.py) — Calculate uncertainty metrics to monitor distribution changes.
