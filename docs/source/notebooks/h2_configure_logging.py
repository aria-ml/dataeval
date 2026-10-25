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
# # How to configure logging with DataEval
#
# DataEval logs through Python's standard `logging` module, on a `dataeval` logger and its subsystem loggers. Attach handlers to
# send the messages to the console or to a file. DataEval primarily logs at the DEBUG, INFO and WARNING levels.
#
# Use this guide when you want to see what DataEval is doing, debug an issue, or save logs for later analysis.
#
# You need `dataeval` and a basic understanding of Python's logging module. The examples run `ber_knn` and `ber_mst` on random
# blobs.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
import logging
from pathlib import Path

import sklearn.datasets as dsets

from dataeval.core import ber_knn, ber_mst

# %% tags=["hide-input"]
blobs = dsets.make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)
embeddings, labels = blobs[0], blobs[1]

# %% [markdown]
# ## Log to the console
#
# Attach a handler to the `dataeval` logger and set its level. INFO shows when each operation starts and finishes.

# %%
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))

dataeval_logger = logging.getLogger("dataeval")
dataeval_logger.setLevel(logging.INFO)
dataeval_logger.addHandler(console_handler)

result = ber_knn(embeddings, labels, k=3)

# %% [markdown]
# Set the level to DEBUG for more detail:

# %%
dataeval_logger.setLevel(logging.DEBUG)
result = ber_mst(embeddings, labels)

# %% [markdown]
# For a quick look, {func}`dataeval.log` attaches a verbose console handler in one call. Calling it again reuses that handler and
# only sets the level.

# %% [markdown]
# ## Log to a file
#
# Add a file handler. Here the console keeps showing INFO while the file receives everything down to DEBUG.

# %%
console_handler.setLevel(logging.INFO)

log_file = Path("dataeval_operations.log")
file_handler = logging.FileHandler(log_file, mode="w")  # "a" appends instead
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
dataeval_logger.addHandler(file_handler)

result = ber_knn(embeddings, labels, k=5)
print(log_file.read_text())

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert "DEBUG" in log_file.read_text()
log_file.unlink()

# %% [markdown]
# ## Narrow the output by subsystem
#
# Every DataEval log record is emitted on `dataeval` or one of its subsystem loggers, so configuring `dataeval` alone captures
# everything. To narrow the output, set a level or attach a handler on a subsystem instead:
#
# | Namespace | Covers |
# | --- | --- |
# | `dataeval.core` | The stateless numerical routines |
# | `dataeval.metadata` | `Metadata`, including automatic binning warnings |
# | `dataeval.embeddings` | `Embeddings` batching and caching |
# | `dataeval.data` | Dataset views, splits, and crops |
# | `dataeval.bias`, `dataeval.shift`, `dataeval.scope`, `dataeval.quality` | The evaluators in each subpackage |
# | `dataeval.extractors` | Feature extractors |
# | `dataeval.utils` | Preprocessing and training helpers |
#
# For example, to quiet everything except metadata handling:
#
# ```python
# logging.getLogger("dataeval").setLevel(logging.WARNING)
# logging.getLogger("dataeval.metadata").setLevel(logging.DEBUG)
# ```
#
# The names do not follow DataEval's file layout, and a few, such as `dataeval.metadata` and `dataeval.embeddings`, have no
# importable module of the same name. Each record carries its file and line whichever logger it lands on. Include
# `%(filename)s:%(lineno)s` in your formatter to see them.

# %% [markdown]
# ## Turn logging off
#
# `logging.disable` silences every logger up to a level until you re-enable it.

# %%
logging.disable(logging.CRITICAL)
result = ber_mst(embeddings, labels)  # nothing is logged

logging.disable(logging.NOTSET)

# %% [markdown]
# ## Next steps
#
# - [Clean a dataset](./tt_clean_dataset.py) — Clean and prepare datasets for model training by finding duplicates, outliers, and corrupted data.
# - [How to configure global DataEval defaults](./h2_configure_defaults.py) — Set global default device, batch size, worker processes, and seed.
