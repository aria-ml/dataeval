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
# # Speed up passes over a large dataset
#
# When a dataset decodes each item as it is read, such as large photos, every pass waits on its reads. Read a pass's
# pixels in several threads with {func}`.use_load_threads`, build {class}`.Metadata` from that read with the
# `record_reads` option of {class}`.View`, and compute statistics in several processes with
# {func}`.use_max_processes`. The first two are experimental.
#
# If the slow pass needs no pixels, such as `Metadata` or a {class}`.Filter`, use lazy images: `lazy=True` on a
# `maite-datasets` dataset, or [Defer image and video decoding](./h2_defer_decoding.py).
#
# You need `dataeval`, `maite-datasets` and `torchvision`, and about 1.1 GB of disk for the SeaDrone download.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets torchvision
except Exception:
    pass

# %%
import warnings

import numpy as np
import torch
from maite_datasets.object_detection import SeaDrone
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings, Metadata
from dataeval.config import set_device, set_seed, use_load_threads, use_max_processes
from dataeval.core import image_stats
from dataeval.data import Limit, Shuffle, View
from dataeval.exceptions import ExperimentalWarning
from dataeval.extractors import TorchExtractor

set_seed(0)  # For reproducibility
set_device("cuda" if torch.cuda.is_available() else "cpu")

# This guide hides the warnings that load threads and `record_reads` are experimental, and the ones about binning
# factors it does not use.
warnings.filterwarnings("ignore", category=ExperimentalWarning)
warnings.filterwarnings("ignore", message=".*(was dropped|binned automatically).*")

# %% tags=["hide-input"]
# A 48-image sample of SeaDrone stands in for a dataset that decodes each photo as it is read: by default its loader
# decodes the full 3840 x 2160 image on every read.
dataset = View(SeaDrone(root="./data", image_set="val", download=True), [Shuffle(seed=0), Limit(48)])

# %% [markdown]
# ## Read the pixels once, in threads
#
# Wrap the pass in {func}`.use_load_threads` and call `compute()` inside the block: {class}`.Embeddings` reads on first
# use, so a pass that runs after the block reads one item at a time. The view keeps what this pass reads for the next
# step.

# %%
weights = ResNet18_Weights.DEFAULT
extractor = TorchExtractor(
    resnet18(weights=weights, progress=False),
    transforms=weights.transforms(),
    layer_name="avgpool",
)
view = View(dataset, record_reads=True)

with use_load_threads(4):
    embeddings = Embeddings(view, extractor=extractor, batch_size=32).compute()

print(embeddings.shape)

# %% [markdown]
# The embeddings equal the ones a serial read gives. The dataset must be safe to read from several threads, the
# threads hold about four times their count of items in memory, and they can be slower when reads are short, so time a
# {class}`.Limit` sample of your data with and without them; {func}`.set_load_threads` lists the passes they reach.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
import threading


class _Counting:
    """Pass-through that records each read and the thread it ran on."""

    def __init__(self, source):
        self.source, self.metadata, self.reads, self.threads = source, source.metadata, 0, set()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        self.reads += 1
        self.threads.add(threading.get_ident())
        return self.source[index]


serial = np.asarray(Embeddings(dataset, extractor=extractor, batch_size=32).compute())
assert np.allclose(np.asarray(embeddings), serial, atol=1e-5)

counted = _Counting(dataset)
with use_load_threads(4):
    Embeddings(View(counted), extractor=extractor, batch_size=32).compute()
assert len(counted.threads) > 1

# %% [markdown]
# ## Build the metadata from the kept reads
#
# Pass {class}`.Metadata` the same view, after the pixel pass. It takes each item's targets and metadata from what the
# pass read instead of reading the dataset again.

# %%
metadata = Metadata(view)
print(metadata.level_counts)

# %% [markdown]
# The view holds each item's target and metadata in memory for as long as it lives. It holds nothing until a pass on it
# has run, and `Metadata` then reads the whole dataset again.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert metadata.dataframe.equals(Metadata(dataset).dataframe)

recorded = _Counting(dataset)
recorded_view = View(recorded, record_reads=True)
with use_load_threads(4):
    Embeddings(recorded_view, extractor=extractor, batch_size=32).compute()
before = recorded.reads
_ = Metadata(recorded_view).dataframe
assert recorded.reads - before <= 1

plain = _Counting(dataset)
plain_view = View(plain)
Embeddings(plain_view, extractor=extractor, batch_size=32).compute()
before = plain.reads
_ = Metadata(plain_view).dataframe
assert plain.reads - before >= len(dataset)

# %% [markdown]
# ## Compute statistics in several processes
#
# Load threads only read. {func}`.image_stats` computes in worker processes, which {func}`.use_max_processes` sets,
# while it reads in the load threads, and on large photos the processes are what speed it up.

# %%
with use_max_processes(4):
    stats = image_stats(view)

print(len(stats["source_index"]), "rows of", len(stats["stats"]), "statistics")

# %% [markdown]
# Each worker process receives every image it measures, so processes suit a pass whose computing outweighs that copy.
# The two are separate settings: give threads to a pass that waits on its reads and processes to one that computes.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
sample = View(dataset, Limit(4))
default = image_stats(sample)
with use_max_processes(4):
    parallel = image_stats(sample)
for name, values in default["stats"].items():
    a, b = np.asarray(values), np.asarray(parallel["stats"][name])
    assert np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), name

# %% [markdown]
# ## Next steps
#
# - [Keep metadata in sync with its dataset and cache it](./h2_sync_and_cache_metadata.py) — Save the metadata so that
#   later sessions skip the walk over the dataset.
# - [Create embeddings](./h2_create_embeddings.py) — Choose the extractor for the pixel pass.
