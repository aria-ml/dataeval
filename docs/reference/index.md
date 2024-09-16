# API Reference

```{currentmodule} dataeval
```

## Detectors

Detectors can determine if a dataset or individual images in a dataset are indicative of a specific issue.

### Drift

[Drift](../concepts/glossary.md#drift) detectors identify if the statistical properties of the data has changed.

:::{toctree}
:hidden:
:titlesonly:
:maxdepth: 4

detectors/drift/drift_cvm
detectors/drift/drift_ks
detectors/drift/drift_mmd
detectors/drift/drift_uncertainty
:::

```{eval-rst}
.. autosummary::

    detectors.drift.DriftCVM
    detectors.drift.DriftKS
    detectors.drift.DriftUncertainty
    detectors.drift.DriftMMD
```

#### _Kernels_

Kernels are used to map non-linear data to a higher dimensional space.

:::{toctree}
:hidden:
:titlesonly:

detectors/drift/kernels/guassianrbf
:::

```{eval-rst}
.. autosummary::

    detectors.drift.kernels.GaussianRBF
```

#### _Updates_

Update strategies inform how the drift detector classes update the reference data when monitoring for drift.

:::{toctree}
:hidden:
:titlesonly:

detectors/drift/updates/lastseenupdate
detectors/drift/updates/reservoirsamplingupdate
:::

```{eval-rst}
.. autosummary::

    detectors.drift.updates.LastSeenUpdate
    detectors.drift.updates.ReservoirSamplingUpdate
```

### Linters

[Linters](../concepts/glossary.md#linter) help identify potential issues in training and test data and are an important aspect of [data cleaning](../concepts/DataCleaning.md).

:::{toctree}
:hidden:
:titlesonly:

detectors/linters/clusterer
detectors/linters/duplicates
detectors/linters/outliers
:::

```{eval-rst}
.. autosummary::

    detectors.linters.Clusterer
    detectors.linters.Duplicates
    detectors.linters.Outliers
```

### Out-of-Distribution

[Out-of-distribution](../concepts/glossary.md#out-of-distribution-ood) detectors identify data that is different from the data used to train a particular model.

:::{toctree}
:hidden:
:titlesonly:

detectors/ood/ood_ae
detectors/ood/ood_aegmm
detectors/ood/ood_llr
detectors/ood/ood_vae
detectors/ood/ood_vaegmm
detectors/ood/oodscore
:::

```{eval-rst}
.. autosummary::

    detectors.ood.OOD_AE
    detectors.ood.OOD_AEGMM
    detectors.ood.OOD_LLR
    detectors.ood.OOD_VAE
    detectors.ood.OOD_VAEGMM
    detectors.ood.OODScore
```

## Flags

Flags are used by the [`imagestats`](metrics/stats/imagestats.md) and [`channelstats`](metrics/stats/channelstats.md) functions, as well as the [`Outliers`](detectors/linters/outliers.md) and [`Duplicates`](detectors/linters/duplicates.md) classes.

:::{toctree}
:hidden:
:titlesonly:

flags/imagestat
:::

```{eval-rst}
.. autosummary::

    flags.ImageStat
```

## Metrics

Metrics are a way to measure the performance of your models or datasets that can then be analyzed in the context of a given problem.

### Bias

[Bias](../concepts/glossary.md#bias) metrics check for skewed or imbalanced datasets and incomplete feature representation which may impact model performance.

:::{toctree}
:hidden:
:titlesonly:

metrics/bias/balance
metrics/bias/coverage
metrics/bias/diversity
metrics/bias/label_parity
metrics/bias/parity
:::

```{eval-rst}
.. autosummary::

    metrics.bias.balance
    metrics.bias.coverage
    metrics.bias.diversity
    metrics.bias.label_parity
    metrics.bias.parity
```

### Estimators

Estimators calculate performance bounds and the statistical distance between datasets.

:::{toctree}
:hidden:
:titlesonly:

metrics/estimators/ber
metrics/estimators/divergence
metrics/estimators/uap
:::

```{eval-rst}
.. autosummary::

    metrics.estimators.ber
    metrics.estimators.divergence
    metrics.estimators.uap
```

### Statistics

Statistics metrics calculate a variety of image properties and pixel statistics against the image and individual channels of an image.

:::{toctree}
:hidden:
:titlesonly:

metrics/stats/channelstats
metrics/stats/imagestats
:::

```{eval-rst}
.. autosummary::

    metrics.stats.channelstats
    metrics.stats.imagestats
```

## Workflows

Workflows perform a sequence of actions to analyze the dataset and make predictions.

:::{toctree}
:hidden:
:titlesonly:

workflows/sufficiency
:::

```{eval-rst}
.. autosummary::

    workflows.Sufficiency
```

## Supported Backends

The models and model trainers provided by DataEval are meant to assist users in setting up
architectures that are guaranteed to work with applicable DataEval metrics.
Currently DataEval supports both Tensorflow and PyTorch backends. 

:::
### PyTorch
:::

DataEval uses PyTorch as its main backend for metrics that require neural networks.
While these metrics can take in custom models, DataEval provides utility classes
to create a seamless integration between custom models and DataEval's metrics.

:::
#### _Models_
:::

```{eval-rst}
.. autosummary::

    torch.models.AriaAutoencoder
    torch.models.Decoder
    torch.models.Encoder
```

:::
#### _Trainer_
:::

```{eval-rst}
.. autosummary::

    torch.trainer.AETrainer
```

:::
### Tensorflow
:::

The Tensorflow models provided are tailored for usage with the out of distribution detection metrics.
DataEval provides both basic default models through the utility function `create_model` as well as 
constructors which allow for customization of the encoder, decoder and any other applicable layers
used by the model.

:::
#### _Models_
:::

```{eval-rst}
.. autosummary::

    tensorflow.models.AE
    tensorflow.models.AEGMM
    tensorflow.models.PixelCNN
    tensorflow.models.VAE
    tensorflow.models.VAEGMM
    tensorflow.models.create_model
```

:::
#### _Reconstruction Functions_
:::

```{eval-rst}
.. autosummary::

    tensorflow.recon.eucl_cosim_features
```

:::
#### _Loss Function Classes_
:::

```{eval-rst}
.. autosummary::

    tensorflow.loss.Elbo
    tensorflow.loss.LossGMM
```

:::{toctree}
:maxdepth: 1
:hidden:

backend/torch
backend/tensorflow
:::
