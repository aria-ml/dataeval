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
# # How to choose where a transform belongs

# %% [markdown]
# ## Problem statement
#
# A transform has two possible homes, and choosing the wrong one fails silently.
#
# Put model preprocessing in the **view** and every evaluator sees it: {class}`.Outliers`
# reports brightness and contrast on ImageNet-normalized tensors, which are not numbers about
# your data. Put dataset-defining manipulation in the **extractor** and only the embeddings
# see it: if 512x512 tiles are what ships, your statistics still describe source-resolution
# imagery that does not.
#
# Neither mistake raises. This guide gives you the rule that tells the two apart, and shows
# you what to do when DataEval warns that a statistic has stopped describing your data.

# %% [markdown]
# ### When to use
#
# Read this before you write your first transform. Come back to it whenever you hit a
# {class}`~dataeval.exceptions.StatsInvalidatedWarning`.

# %% [markdown]
# ### What you will need
#
# 1. Any dataset implementing {class}`~dataeval.protocols.AnnotatedDataset`.
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `maite-datasets`

# %% [markdown]
# ## Getting started
#
# First import the required libraries needed to set up the example, including the {class}`.View` container and the {class}`.Limit` operation.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
import warnings

import numpy as np
import torch
from maite_datasets.image_classification import MNIST

from dataeval.data import Crop, Limit, Resize, SelectChannels, View
from dataeval.extractors import TorchExtractor
from dataeval.flags import ImageStats
from dataeval.quality import Outliers

# %%
mnist = View(MNIST("./data", image_set="test", download=True), [Limit(500)])
print("source dataset size:", len(mnist))
print("source image shape:", np.asarray(mnist[0][0]).shape)

# %% [markdown]
# ## Decide where a transform belongs
#
# "Transforms" covers three unrelated jobs. Two questions tell them apart.
#
# **Does an evaluator that never touches your model need to see it?**
# This separates view-level from extractor-level.
#
# **Is it deterministic, with an unambiguous rewrite of the targets?**
# This separates the curated operations from the escape hatch.
#
# | Kind | Example | Home |
# | --- | --- | --- |
# | Model-input adaptation | resize-to-224, ImageNet normalize, dtype cast | extractor `transforms=` |
# | Dataset definition | crop out a HUD, downsample to the deployed resolution, band selection | curated View operations |
# | Augmentation / corruption | random flip, color jitter, blur | {class}`.TorchvisionTransform` |
#
# The load-bearing case: nobody wants {class}`.Outliers` reporting brightness on normalized
# tensors, and everybody wants it running on 512x512 tiles when 512x512 tiles are what ships.

# %% [markdown]
# ## Put dataset definition in the view
#
# Suppose your deployed system runs at 14x14. That is a fact about the data under evaluation,
# not about the model, so it belongs in the {class}`.View` with the {class}`.Resize` operation,
# where the statistics, the duplicate search, and the embeddings all see the same thing.

# %%
deployed = View(mnist, [Resize((14, 14))])
print("deployed image shape:", np.asarray(deployed[0][0]).shape)

# %% [markdown]
# The other two curated operations answer the same question. Use {class}`.Crop` to remove a
# fixed region that is never evidence: a burned-in overlay, a scan border, a dead sensor
# region. Use {class}`.SelectChannels` to narrow or combine channels, for a mono sensor stored
# as RGB, or for the bands of a multispectral cube you actually evaluate.

# %%
bordered = View(mnist, [Crop((4, 4, 24, 24))])
print("cropped image shape:", np.asarray(bordered[0][0]).shape)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert np.asarray(deployed[0][0]).shape == (1, 14, 14)
assert np.asarray(bordered[0][0]).shape == (1, 20, 20)

# %% [markdown]
# ## Respond to an invalidation warning
#
# A resize makes `width`, `height`, and `aspect_ratio` report the resize target, and makes
# `sharpness` measure the interpolation kernel. DataEval still computes them; nothing is
# blocked. But they are now facts about the transform.
#
# Every operation declares what it invalidates, and the quality evaluators intersect that
# against the {class}`.ImageStats` statistics you asked for. Where the two overlap you get a
# {class}`~dataeval.exceptions.StatsInvalidatedWarning`.

# %%
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Outliers(flags=ImageStats.DIMENSION | ImageStats.VISUAL).evaluate(deployed)

invalidated = [w for w in caught if w.category.__name__ == "StatsInvalidatedWarning"]
print(str(invalidated[0].message))

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(invalidated) == 1
assert "sharpness" in str(invalidated[0].message)

# %% [markdown]
# ### Three ways to respond
#
# **1. Move it to the extractor.** If the resize was there to feed a model, it was model
# preprocessing all along. Move it to `transforms=` and the warning has done its job.
#
# **2. Narrow `flags=`.** If the resize really is part of the dataset definition, drop the
# statistics it invalidates and keep the rest. Resize leaves channel count alone, and it
# leaves hashes alone: resize-then-phash is a *better* near-duplicate check across
# heterogeneous source resolutions, not a worse one.

# %%
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Outliers(flags=ImageStats.PIXEL).evaluate(deployed)

print("warnings:", [w.category.__name__ for w in caught if "Invalidated" in w.category.__name__])

# %% [markdown]
# **3. Assert with `invalidates=`.** Every operation takes an override, for when you know
# something the default cannot. Defaults are deliberately conservative:
# {class}`.TorchvisionTransform` declares `ImageStats.ALL`, because an arbitrary transform can
# move anything. So you reach for the override mainly to *narrow* a declaration to what you
# can vouch for.
#
# Say you have settled that the deployed resolution **is** your dataset, and you no longer
# need to hear about the dimension statistics. You do still want to hear about `sharpness`,
# because bilinear downsampling smooths and nothing about your source resolution changes that.

# %%
asserted = View(mnist, [Resize((14, 14), invalidates=ImageStats.VISUAL_SHARPNESS)])

warned = {}
for name, flags in [("DIMENSION", ImageStats.DIMENSION), ("VISUAL", ImageStats.VISUAL)]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Outliers(flags=flags).evaluate(asserted)
    warned[name] = any(w.category.__name__ == "StatsInvalidatedWarning" for w in caught)
    print(f"{name:>9}: {'warned' if warned[name] else 'silent'}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert warned == {"DIMENSION": False, "VISUAL": True}

# %% [markdown]
# `ImageStats.NONE` is available too, but it claims the operation changes nothing about any
# statistic, and a resize still smooths and still promotes the dtype. Narrow to the set you
# can defend instead.
#
# Either way, use the override deliberately. It silences a warning about *your* data, so it
# is a claim you are making, not a way to quiet the output.

# %% [markdown]
# ## Layer the two
#
# The two homes are not alternatives. A realistic pipeline uses both: the deployed resolution
# defines the dataset, and the model's normalization adapts that dataset to the model.
#
# Below, the resize sits in the view, so {class}`.Outliers` and {class}`.Duplicates` see
# 14x14 imagery. The normalization sits in the {class}`.TorchExtractor` (or extractor), so only the embeddings see it, and
# `brightness` stays a statistic about pixels rather than about a mean and standard deviation
# someone chose.

# %%
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(14 * 14, 16))


def normalize(image: torch.Tensor) -> torch.Tensor:
    """Model preprocessing: scale to [0, 1], then standardize."""
    return (image / 255.0 - 0.1307) / 0.3081


extractor = TorchExtractor(model, transforms=normalize)
embedding = extractor(np.asarray(deployed[0][0])[None])

print("view sees:", np.asarray(deployed[0][0]).shape, "-- raw pixel values")
print("extractor produces:", tuple(np.asarray(embedding).shape), "-- normalized, model-side only")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert np.asarray(embedding).shape[-1] == 16

# %% [markdown]
# The same split settles a case that looks like a view operation but is not. A model wanting
# three channels from a 1-channel dataset needs `SelectChannels("rgb")` in `transforms=`,
# because that is model-input adaptation. Put it in the view and you have asserted that your
# grayscale dataset is an RGB dataset, and `channels` now describes the model's input layer.

# %%
print("as a claim about the data:", np.asarray(View(mnist, [SelectChannels("rgb")])[0][0]).shape)
print("...but ask whether the data really has three channels")

# %% [markdown]
# ## Reach for the escape hatch last
#
# {class}`.TorchvisionTransform` runs any torchvision v2 transform over a view, carrying boxes
# and dropping out-of-frame detections with them. Use it for augmentation and corruption
# applied dataset-wide, such as evaluating a corrupted view against the source to probe
# robustness, and for pipelines you already have.
#
# Know two things before you reach for it.
#
# Random transforms are seeded per datum from the datum's `id`, so the view is stable: the
# statistics pass, the embedding pass, and the duplicates pass all see the same pixels, and
# iterating twice is byte-identical.
#
# **A view built through it may not be reconstructable from its sidecar.** Provenance records
# each operation's `repr`. A curated operation reprs to something you can paste back, while a
# `v2.Compose([...])` reprs multi-line, and a lambda inside one records nothing recoverable.
# That is the last reason to prefer the curated operations.

# %% [markdown]
# ## Summary
#
# - Ask **does an evaluator that never touches my model need to see it?** Yes puts it in a
#   view operation; no puts it in the extractor's `transforms=`.
# - Put dataset definition in the view: deployed resolution, fixed crops, band selection are
#   claims about your data.
# - Put model-input adaptation in the extractor: normalize, dtype cast, and channel broadcast
#   stay out of your statistics there.
# - Answer a {class}`~dataeval.exceptions.StatsInvalidatedWarning` in one of three ways: move
#   the transform to the extractor, narrow `flags=`, or assert with `invalidates=`.
# - Expect to use both homes at once. Layering them is the normal case, not a compromise.

# %% [markdown]
# ## Next steps
#
# - [Data Integrity](../concepts/DataIntegrity.md) — Learn about data corruption, missing values, and dataset quality issues.
# - [Embeddings](../concepts/Embeddings.md) — Learn how feature vectors represent data for similarity and quality analysis.
# - [How to build dataset views with View and Operation](./h2_build_dataset_views.py) — Construct and chain dataset view transformations.
# - [How to embed object detection box crops](./h2_embed_detection_crops.py) — Extract and embed cropped regions from object detection bounding boxes.
# - [How to customize calculation of image stats](./h2_custom_image_stats_object_detection.py) — Configure custom image statistics for object detection dataset analysis.
