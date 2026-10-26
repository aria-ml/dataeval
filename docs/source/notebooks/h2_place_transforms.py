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
# # How to respond to a StatsInvalidatedWarning
#
# A {class}`~dataeval.exceptions.StatsInvalidatedWarning` says that a transform in your {class}`.View` has changed what
# some of the statistics you asked for measure. Every operation declares the {class}`.ImageStats` it invalidates, and
# evaluators such as {class}`.Outliers` intersect that with your `flags=`. Nothing is blocked, but the invalidated
# statistics now describe the transform and not your data. See
# [Image Statistics](../concepts/ImageStatistics.md#transforms-and-the-statistics-they-invalidate).
#
# Use it when you see the warning, or before you put a resize in a view.
#
# You need `dataeval` and `maite-datasets`. This guide uses 500 MNIST test images.

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

from dataeval.data import Limit, Resize, View
from dataeval.extractors import TorchExtractor
from dataeval.flags import ImageStats
from dataeval.quality import Outliers

# %% [markdown]
# ## Trigger the warning
#
# A resize to the resolution your system deploys at is a fact about the data, so it belongs in the view. It also makes
# the dimension statistics, such as `width`, `height` and `aspect_ratio`, report the resize target, and makes
# `sharpness` measure the interpolation kernel. Ask an evaluator for those statistics and it warns:

# %%
mnist = View(MNIST("./data", image_set="test", download=True), [Limit(500)])
print("source dataset size:", len(mnist))
print("source image shape:", np.asarray(mnist[0][0]).shape)

# %%
deployed = View(mnist, [Resize((14, 14))])
print("deployed image shape:", np.asarray(deployed[0][0]).shape)

# %%
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Outliers(flags=ImageStats.DIMENSION | ImageStats.VISUAL).evaluate(deployed)

invalidated = [w for w in caught if w.category.__name__ == "StatsInvalidatedWarning"]
print(str(invalidated[0].message))

# %% [markdown]
# ## Move model preprocessing to the extractor
#
# If the transform was there to feed a model, it was model preprocessing all along. Give it to the extractor's
# `transforms=` and the view stays raw. Here the resize stays in the view as the deployed resolution, and only the
# normalization is model-side.

# %%
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(14 * 14, 16))


def normalize(image: torch.Tensor) -> torch.Tensor:
    """Model preprocessing: scale to [0, 1], then standardize."""
    return (image / 255.0 - 0.1307) / 0.3081


extractor = TorchExtractor(model, transforms=normalize)
embedding = extractor(np.asarray(deployed[0][0])[None])

print("view sees:", np.asarray(deployed[0][0]).shape, "-- raw pixel values")
print("extractor produces:", tuple(np.asarray(embedding).shape), "-- normalized, model-side only")

# %% [markdown]
# The view still sees raw pixel values, so `brightness` stays a statistic about pixels and not about a mean and standard
# deviation someone chose. For a whole dataset, pass the extractor to {class}`.Embeddings`, which makes the same call
# for each batch. See [Embeddings](../concepts/Embeddings.md#put-model-preprocessing-in-the-extractor).

# %% [markdown]
# ## Narrow the flags
#
# If the resize is part of the dataset definition, drop the statistics it invalidates and keep the rest. The default
# resize declares the dimension statistics, except the channel count, and sharpness as invalidated. It does not declare
# the pixel statistics or the hashes, so `flags=ImageStats.PIXEL` runs without a warning.

# %%
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Outliers(flags=ImageStats.PIXEL).evaluate(deployed)

print("warnings:", [w.category.__name__ for w in caught if "Invalidated" in w.category.__name__])

# %% [markdown]
# ## Assert what the operation invalidates
#
# Every operation takes `invalidates=` for when you know something the default cannot. The defaults are conservative, so
# use it to narrow a declaration to what you can vouch for. If the deployed resolution is your dataset you no longer
# need to hear about the dimension statistics, but bilinear downsampling still smooths, so `sharpness` should keep
# warning.

# %%
asserted = View(mnist, [Resize((14, 14), invalidates=ImageStats.VISUAL_SHARPNESS)])

warned = {}
for name, flags in [("DIMENSION", ImageStats.DIMENSION), ("VISUAL", ImageStats.VISUAL)]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Outliers(flags=flags).evaluate(asserted)
    warned[name] = any(w.category.__name__ == "StatsInvalidatedWarning" for w in caught)
    print(f"{name:>9}: {'warned' if warned[name] else 'silent'}")

# %% [markdown]
# `ImageStats.NONE` claims the operation changes nothing about any statistic, and a resize still smooths and promotes
# the dtype, so narrow to the set you can defend. The override is a claim about your data and not a way to quiet the
# output.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert np.asarray(deployed[0][0]).shape == (1, 14, 14)
assert len(invalidated) == 1
assert "sharpness" in str(invalidated[0].message)
assert np.asarray(embedding).shape[-1] == 16
assert warned == {"DIMENSION": False, "VISUAL": True}

# %% [markdown]
# ## Next steps
#
# - [Image Statistics](../concepts/ImageStatistics.md) — Understand image statistics and metrics computed across channels.
# - [Embeddings](../concepts/Embeddings.md) — Learn how feature embeddings represent dataset samples in low-dimensional space.
# - [How to build dataset views with View and Operation](./h2_build_dataset_views.py) — Construct and chain dataset view transformations.
