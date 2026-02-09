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

# Configuring hardware: PyTorch devices and cpu processes

+++

## Problem Statement

DataEval provides global configuration settings to control computational resources and hardware acceleration. This guide
shows how to configure the default PyTorch device and the maximum number of worker processes.

+++

### _When to use_

- You need to specify GPU or CPU execution for PyTorch-based operations
- You want to control the number of parallel worker processes
- You need to optimize performance for your hardware configuration

+++

### _What you will need_

1. A Python environment with dataeval installed

+++

## _Getting Started_

```{code-cell} ipython3
:tags: [remove_cell]

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

## Using temporary context managers

+++

Temporarily override the max processes setting using a context manager:

```{code-cell} ipython3
dataeval.config.set_max_processes(8)
print(f"Before context: {dataeval.config.get_max_processes()}")

with dataeval.config.use_max_processes(2):
    print(f"Inside context: {dataeval.config.get_max_processes()}")
    # Perform operations with max_processes=2

print(f"After context: {dataeval.config.get_max_processes()}")
```
