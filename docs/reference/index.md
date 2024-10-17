# API Reference

DataEval's API is split into several submodules which support specific goals and are detailed below.  The base module
is empty except for `__version__` by design.

## Submodules

```{eval-rst}
.. autosummary::

    dataeval.detectors
    dataeval.detectors.drift
    dataeval.detectors.linters
    dataeval.detectors.ood
    dataeval.metrics
    dataeval.metrics.bias
    dataeval.metrics.estimators
    dataeval.metrics.stats
    dataeval.utils
    dataeval.utils.tensorflow
    dataeval.utils.torch
    dataeval.workflows
```

:::{toctree}
:hidden:
:titlesonly:
:maxdepth: 1

detectors
metrics
utils
workflows
:::
