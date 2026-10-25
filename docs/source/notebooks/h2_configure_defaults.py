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
# # How to configure global DataEval defaults
#
# DataEval keeps global settings for the compute it uses and for reproducibility: the PyTorch device, the default batch size, the
# maximum number of worker processes and a seed. Set one once and every evaluator reads it. Each setting also has a context
# manager that overrides it temporarily. The `dataeval.config` API reference lists every argument.
#
# Use this guide when you want to:
#
# - run on the CPU, a GPU or a specific GPU
# - set a default batch size
# - control the number of parallel worker processes
# - make results reproducible
#
# You need `dataeval`.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
import dataeval

# %% [markdown]
# ## Set the device
#
# Pass `"cpu"`, `"cuda"` or a specific GPU such as `"cuda:1"`. See
# [`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) for more information. Pass `None` to
# reset to PyTorch's default device.

# %%
dataeval.config.set_device("cpu")
print(f"Current device for DataEval: {dataeval.config.get_device()}")

dataeval.config.set_device(None)
print(f"Current device for DataEval: {dataeval.config.get_device()}")

# %% [markdown]
# ## Set the batch size
#
# The batch size must be a positive integer. A function that requires a `batch_size` fails if you do not pass one and no global
# batch size is set.

# %%
dataeval.config.set_batch_size(64)
print(f"Current batch size: {dataeval.config.get_batch_size()}")

dataeval.config.set_batch_size(None)

# %% [markdown]
# ## Set the maximum number of worker processes
#
# DataEval follows the conventions of [`scikit-learn`](https://scikit-learn.org/stable/glossary.html#term-n_jobs) and
# [`joblib`](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html): `-1` uses all visible CPU cores and `None`
# unsets the limit.

# %%
dataeval.config.set_max_processes(4)
print(f"Max processes: {dataeval.config.get_max_processes()}")

dataeval.config.set_max_processes(-1)
print(f"Max processes: {dataeval.config.get_max_processes()}")

dataeval.config.set_max_processes(None)
print(f"Max processes: {dataeval.config.get_max_processes()}")

# %% [markdown]
# ## Set the seed
#
# DataEval uses a global seed for operations that rely on random state, such as clustering (KMeans), mutual information
# estimation, domain classification, data selection and bag-of-visual-words extraction. It passes the seed as the `random_state`
# of internal library calls such as scikit-learn's `KMeans`, `KNeighborsClassifier` and `StratifiedKFold`.
#
# - `all_generators=True` also seeds NumPy (`np.random.seed`) and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`).
# - `deterministic=True` forces PyTorch to use deterministic algorithms, which can reduce performance and raises a `RuntimeError`
#   for an operation with no deterministic implementation. Combine it with `all_generators=True` for full reproducibility.
# - `set_seed(None)` resets all generators and disables deterministic algorithms, whatever the other arguments.

# %%
dataeval.config.set_seed(42)
print(f"Current seed: {dataeval.config.get_seed()}")

dataeval.config.set_seed(42, all_generators=True, deterministic=True)

dataeval.config.set_seed(None)
print(f"Current seed: {dataeval.config.get_seed()}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
import numpy as np
import torch

dataeval.config.set_seed(42, all_generators=True)
first = (np.random.rand(3), torch.rand(3))
dataeval.config.set_seed(42, all_generators=True)
second = (np.random.rand(3), torch.rand(3))
assert np.array_equal(first[0], second[0])
assert torch.equal(first[1], second[1])
dataeval.config.set_seed(None)

# %% [markdown]
# ## Override a setting temporarily
#
# `use_device`, `use_batch_size`, `use_max_processes` and `use_seed` are context managers that restore the previous value when
# the block exits.
#
# `use_seed` swaps only the seed DataEval hands to the estimators it calls, which is what `get_seed` returns. It leaves the NumPy
# and PyTorch generators alone, because their state cannot be put back on exit. Use `set_seed(..., all_generators=True)` when a
# block needs those seeded.

# %%
dataeval.config.set_batch_size(64)
with dataeval.config.use_batch_size(16):
    print(f"Inside the context: {dataeval.config.get_batch_size()}")
print(f"After the context: {dataeval.config.get_batch_size()}")

dataeval.config.set_seed(42)
with dataeval.config.use_seed(7):
    print(f"Inside the context: {dataeval.config.get_seed()}")
print(f"After the context: {dataeval.config.get_seed()}")

# %% [markdown]
# ## Next steps
#
# - [Detect common augmentations as duplicates](./tt_augmentation_duplicates.py) — Find near-identical images created through synthetic transformations and augmentations.
# - [How to configure logging with DataEval](./h2_configure_logging.py) — Set up console and file logging to monitor DataEval execution and debug operations.
