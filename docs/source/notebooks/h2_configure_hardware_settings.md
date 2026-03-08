---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: dataeval
  language: python
  name: python3
---

# How to configure global hardware configuration defaults in DataEval

+++

## Problem statement

DataEval provides global configuration settings to control computational resources and hardware acceleration. This guide
shows how to configure the default PyTorch device, batch size, and the maximum number of worker processes.

+++

### When to use

- You need to specify GPU or CPU execution for PyTorch-based operations
- You want to set a global default batch size for data processing operations
- You want to control the number of parallel worker processes
- You need to optimize performance for your hardware configuration

+++

### What you will need

1. A Python environment with dataeval installed

+++

## Getting started

```{code-cell} ipython3
---
tags: [remove_cell]
---
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    %pip install -q dataeval
except Exception:
    pass
```

```{code-cell} ipython3
import dataeval
```

## Configuring the PyTorch device

+++

DataEval provides configuration options for setting the PyTorch device to use within DataEval. See
[`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) for more information.

+++

### Set the default device to CPU

```{code-cell} ipython3
dataeval.config.set_device("cpu")

print(f"Current device for DataEval: {dataeval.config.get_device()}")
```

### Set the default device to CUDA GPU

```{code-cell} ipython3
dataeval.config.set_device("cuda")

print(f"Current device for DataEval: {dataeval.config.get_device()}")
```

### Set the default device to a specific CUDA GPU

```{code-cell} ipython3
dataeval.config.set_device("cuda:1")

print(f"Current device for DataEval: {dataeval.config.get_device()}")
```

### Reset the device to use PyTorch's default device

```{code-cell} ipython3
dataeval.config.set_device(None)

print(f"Current device for DataEval: {dataeval.config.get_device()}")
```

## Configuring the default batch size

+++

DataEval allows setting a global default batch size for operations that process data in batches. The batch size must be
a positive integer.

Note that functions and methods that require a `batch_size` will fail if not provided and a global batch size is not
set.

+++

### Set the default batch size

```{code-cell} ipython3
dataeval.config.set_batch_size(64)

print(f"Current batch size: {dataeval.config.get_batch_size()}")
```

### Reset the batch size to unset

```{code-cell} ipython3
dataeval.config.set_batch_size(None)

# When no batch size is set, get_batch_size() requires an explicit value
print("Batch size has been unset")
```

## Configuring maximum worker processes

+++

DataEval follows the maximum worker configuration conventions used by
[`scikit-learn`](https://scikit-learn.org/stable/glossary.html#term-n_jobs) and
[`joblib`](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html).

+++

### Set the maximum number of worker processes

```{code-cell} ipython3
dataeval.config.set_max_processes(4)
print(f"Max processes: {dataeval.config.get_max_processes()}")
```

### Set the maximum number of workers to all visible cpu cores

```{code-cell} ipython3
dataeval.config.set_max_processes(-1)
print(f"Max processes: {dataeval.config.get_max_processes()}")
```

### Unset the maximum number of workers

```{code-cell} ipython3
dataeval.config.set_max_processes(None)
print(f"Max processes: {dataeval.config.get_max_processes()}")
```

## Configuring the global seed

+++

DataEval uses a global seed to control randomness in operations that rely on random state, such as clustering (KMeans),
mutual information estimation, domain classification, data selection, and bag-of-visual-words extraction. Setting a seed
ensures reproducible results across runs.

+++

### Set the global seed

This sets the seed used internally by DataEval functions that accept a `random_state` or seed parameter.

```{code-cell} ipython3
dataeval.config.set_seed(42)

print(f"Current seed: {dataeval.config.get_seed()}")
```

### Set the seed for all generators

When `all_generators=True`, the seed is also applied to NumPy (`np.random.seed`) and PyTorch (`torch.manual_seed`,
`torch.cuda.manual_seed_all`). This is useful when you need full reproducibility across all numeric libraries, not just
DataEval's internal operations.

```{code-cell} ipython3
import numpy as np
import torch

# Set the seed with all_generators to seed NumPy and PyTorch
dataeval.config.set_seed(42, all_generators=True)

# Generate random values — these will be the same every time
np_result = np.random.rand(3)
torch_result = torch.rand(3)
print(f"NumPy:   {np_result}")
print(f"PyTorch: {torch_result}")

# Reset and set the same seed again
dataeval.config.set_seed(42, all_generators=True)

# The same values are produced
np_result_2 = np.random.rand(3)
torch_result_2 = torch.rand(3)
print(f"NumPy:   {np_result_2}")
print(f"PyTorch: {torch_result_2}")

assert np.array_equal(np_result, np_result_2)
assert torch.equal(torch_result, torch_result_2)
```

Even without `all_generators`, the global seed is automatically passed as the `random_state` to internal library calls
such as scikit-learn's `KMeans`, `KNeighborsClassifier`, and `StratifiedKFold`, ensuring consistent results from
DataEval operations.

### Enable deterministic algorithms

When `deterministic=True`, PyTorch is forced to use deterministic implementations of its algorithms via
`torch.use_deterministic_algorithms(True)`. This guarantees bitwise reproducibility for PyTorch operations, but may
reduce performance since some optimized (non-deterministic) algorithm implementations will be disabled. If no
deterministic implementation exists for an operation, PyTorch will raise a `RuntimeError`.

This is most useful when combined with `all_generators=True` for full reproducibility.

```{code-cell} ipython3
dataeval.config.set_seed(42, all_generators=True, deterministic=True)

print(f"Current seed: {dataeval.config.get_seed()}")
```

### Reset the seed to unset

Setting the seed to `None` always resets all generators (NumPy, PyTorch) and disables deterministic algorithms,
regardless of the other parameters.

```{code-cell} ipython3
dataeval.config.set_seed(None)

print(f"Current seed: {dataeval.config.get_seed()}")
```

## Using temporary context managers

+++

Temporarily override configuration settings using context managers:

```{code-cell} ipython3
dataeval.config.set_batch_size(64)
print(f"Before context: {dataeval.config.get_batch_size()}")

with dataeval.config.use_batch_size(16):
    print(f"Inside context: {dataeval.config.get_batch_size()}")
    # Perform operations with batch_size=16

print(f"After context: {dataeval.config.get_batch_size()}")
```

```{code-cell} ipython3
dataeval.config.set_max_processes(8)
print(f"Before context: {dataeval.config.get_max_processes()}")

with dataeval.config.use_max_processes(2):
    print(f"Inside context: {dataeval.config.get_max_processes()}")
    # Perform operations with max_processes=2

print(f"After context: {dataeval.config.get_max_processes()}")
```

## See also

### How-to guides

- [How to configuring logging with DataEval](./h2_configure_logging.md)
