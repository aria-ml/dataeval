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
# # How to detect undersampled data subsets
#
# Most computer vision datasets have many images, but some subsets, such as a label or a style within a label, can be
# undersampled. [Coverage](../concepts/DatasetBias.md#measuring-coverage-geometry-in-embedding-space) analysis finds
# this regional sparsity, and the {class}`.Coverage` evaluator returns example images that have few similar instances in
# the dataset.
#
# Use it when you have many images but only a small fraction from certain regimes or labels.
#
# You need an image classification dataset, a model that turns each image into an embedding, `dataeval` and `tabulate`.
# This guide uses an autoencoder trained on MNIST, which stands in for your own model.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import base64
    import io
    import json

    import google.colab  # noqa: F401
    import torch

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
    # !export LC_ALL="en_US.UTF-8"
    # !export LD_LIBRARY_PATH="/usr/lib64-nvidia"
    # !export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
    # !ldconfig /usr/lib64-nvidia

    # Code below is to download the pretrained model weights stored on github
    # !mkdir models
    # !curl -o gitlfsbinary https://api.github.com/repos/aria-ml/dataeval/git/blobs/ad520d5589fdc49830f98d28aa5eaed0bbdfe5cb

    with open("gitlfsbinary") as f:
        rawfile = json.load(f)

    binaryfile = base64.b64decode(rawfile["content"])
    buffer = io.BytesIO(binaryfile)

    temp = torch.load(buffer, weights_only=False)
    torch.save(temp, "models/ae")

    del rawfile
    del binaryfile
    del buffer
    del temp
except Exception:
    pass

# %pip install -q tabulate

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from maite_datasets.image_classification import MNIST
from sklearn.manifold import TSNE

from dataeval import Embeddings, Metadata
from dataeval.data import Limit, View
from dataeval.extractors import TorchExtractor
from dataeval.scope import Coverage

# %% [markdown]
# ## Load the data
#
# Load the MNIST data and create the training dataset.

# %%
# Set seeds
torch.manual_seed(14)

transforms = [
    lambda x: x / 255.0,  # scale to [0, 1]
    lambda x: (x - 0.1307) / 0.3081,  # normalize
    lambda x: x.astype(np.float32),  # convert to float32
]

# MNIST with mean 0 unit variance
train_ds = MNIST(root="./data", image_set="train", transforms=transforms, download=True)

# Select a subset of the dataset
subset = View(train_ds, Limit(2000))


# %% [markdown]
# ## Create the embeddings
#
# In this how-to, you use an autoencoder to reduce the dimension of the MNIST images.
#
# The autoencoder stands in for the model that you would use on your own data. Any extractor works, such as a
# pre-trained network or an autoencoder trained on your dataset. The
# [Embeddings](../concepts/Embeddings.md#choose-an-extractor) page lists the extractors.


# %% tags=["hide-input"]
# Define model architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, 10),
            # 10
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(10, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x


# %% [markdown]
# For computational reasons, you load the trained autoencoder.

# %%
# The trained autoencoder was trained for 1000 epochs
sd = torch.load("models/ae", weights_only=True)
model = Autoencoder()
model.load_state_dict(sd)

# %% [markdown]
# For this example, you take only the first 2000 entries of the data.

# %%
# Create extractor using the autoencoder's encoder portion
extractor = TorchExtractor(model.encoder)

# Calculate the embeddings and extract the labels from the dataset
embeddings = Embeddings(subset, extractor=extractor, batch_size=64)
metadata = Metadata(subset)
labels = metadata.class_labels

# %% [markdown]
# ## Visualize the embeddings
#
# To visualize the encodings, you use TSNE on them to view separation.

# %%
# Visualize 10d as 2d with TSNE
tsne = TSNE(n_components=2)
red_dim = tsne.fit_transform(np.asarray(embeddings))

# %%
# Plot results with color being label
fig, ax = plt.subplots()
scatter = ax.scatter(
    x=red_dim[:, 0],
    y=red_dim[:, 1],
    c=labels,
    label=labels,
)
ax.legend(*scatter.legend_elements(), loc="upper right", ncols=2)
plt.show()

# %% [markdown]
# There is some good separation, but you can see a few images in the "gaps". This could be an artifact of dimension
# reduction, or suggest that the dataset has poor coverage for some covariates.

# %% [markdown]
# ## Measure the coverage
#
# Measure the coverage with {class}`.Coverage`. It rescales the embeddings to the unit interval for you, so the range of
# your extractor's output does not matter.

# %%
# Use data adaptive cutoff
cvrg = Coverage(num_observations=20, percent=0.01).evaluate(metadata, embeddings=embeddings)

# %%
# Plot the least covered 1%
f, axs = plt.subplots(4, 5, figsize=(5, 5))
axs = axs.flatten()
for count, i in enumerate(axs):
    idx = cvrg.uncovered_indices[count]
    i.imshow(np.squeeze(train_ds[idx][0]), cmap="gray")
    i.set_axis_off()
    i.title.set_text(int(labels[idx]))

# %% [markdown]
# {class}`.Coverage` identified that in this set of 2000 images, there is potential under-coverage for wonky 2s and 7s.
# Other digits have some undercovered instances, but they could be outliers.
#
# {class}`.Coverage` also reports coverage for each class. See
# [Identify gaps in a labeled dataset](./tt_identify_dataset_gaps.py).

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
wonky = sum(labels[i] == 2 or labels[i] == 7 for idx, i in enumerate(cvrg.uncovered_indices) if idx < 20)
print(wonky)
assert (wonky / 20) > 0.4

# %% [markdown]
# ## Next steps
#
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Learn about dataset bias, representation, and coverage
#   geometry.
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — Discover structure, clusters, and coverage in
#   unlabeled datasets.
# - [How to create embeddings](./h2_create_embeddings.py) — Create image embeddings with PyTorch, ONNX or a model-free
#   extractor.
