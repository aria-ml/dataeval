Reference Guide
===============

```{currentmodule} dataeval
```

Detectors
---------

:::{toctree}
:hidden:
:titlesonly:

detectors/clusterer
detectors/duplicates
detectors/linter
detectors/drift_cvm
detectors/drift_ks
detectors/drift_mmd
detectors/drift_uncertainty
detectors/gaussianrbf
detectors/lastseenupdate
detectors/reservoirsamplingupdate
detectors/ood_ae
detectors/ood_aegmm
detectors/ood_llr
detectors/ood_vae
detectors/ood_vaegmm
detectors/oodscore
:::

Detectors can determine if a dataset or individual images in a dataset are indicative of a specific issue

:::
### Data Exploration
:::

```{eval-rst}
.. autosummary::

    detectors.Clusterer
    detectors.Duplicates
    detectors.Linter
```

:::
### Data Monitoring
#### Drift
:::

```{eval-rst}
.. autosummary::

    detectors.DriftCVM
    detectors.DriftKS
    detectors.DriftUncertainty
    detectors.DriftMMD
    detectors.GaussianRBF
    detectors.LastSeenUpdate
    detectors.ReservoirSamplingUpdate
```
:::
#### Out of Distribution
:::

```{eval-rst}
.. autosummary::

    detectors.OOD_AE
    detectors.OOD_AEGMM
    detectors.OOD_LLR
    detectors.OOD_VAE
    detectors.OOD_VAEGMM
    detectors.OODScore
```

Metrics
-------

:::{toctree}
:hidden:
:titlesonly:

metrics/channelstats
metrics/imagestats
metrics/balance
metrics/coverage
metrics/divergence
metrics/diversity
metrics/parity
metrics/ber
metrics/uap
:::

Metrics are a way to measure the performance of your models or datasets that can
then be analyzed in the context of a given problem

:::
### Data Exploration
:::

```{eval-rst}
.. autosummary::

    metrics.channelstats
    metrics.imagestats
```
:::
### Metadata/Label Exploration
:::

```{eval-rst}
.. autosummary::

    metrics.balance
    metrics.balance_classwise
    metrics.coverage
    metrics.divergence
    metrics.diversity
    metrics.diversity_classwise
    metrics.parity
    metrics.parity_metadata
```

:::
### Data Performance
:::

```{eval-rst}
.. autosummary::

    metrics.ber
    metrics.uap
```

Flags
-----

:::{toctree}
:hidden:
:titlesonly:

flags/imagehash
flags/imageproperty
flags/imagestatistics
flags/imagevisuals
:::

Flags are used by the `imagestats` and `channelstats` functions, as well as the `Duplicates` and `Linter` classes

```{eval-rst}
.. autosummary::

    flags.ImageHash
    flags.ImageProperty
    flags.ImageStatistics
    flags.ImageVisuals
```

Workflows
---------

:::{toctree}
:hidden:
:titlesonly:

workflows/sufficiency
:::

Workflows perform a sequence of actions to analyze the dataset and make predictions

```{eval-rst}
.. autosummary::

    workflows.Sufficiency
```

Supported Model Backends
------------------------

The models and model trainers provided by DataEval are meant to assist users in setting up
architectures that are guaranteed to work with applicable DataEval metrics.
Below is a list of backends with available trainers and models. 

:::{toctree}
:maxdepth: 1

models/torch
models/tensorflow
:::
