# API Reference

```{currentmodule} dataeval
```

## Detectors

:::{toctree}
:hidden:
:titlesonly:

detectors/clusterer
detectors/drift_cvm
detectors/drift_ks
detectors/drift_mmd
detectors/drift_uncertainty
detectors/duplicates
detectors/gaussianrbf
detectors/lastseenupdate
detectors/linter
detectors/ood_ae
detectors/ood_aegmm
detectors/ood_llr
detectors/ood_vae
detectors/ood_vaegmm
detectors/oodscore
detectors/reservoirsamplingupdate
:::

Detectors can determine if a dataset or individual images in a dataset are indicative of a specific issue

:::
### _Data Exploration_
:::

```{eval-rst}
.. autosummary::

    detectors.Clusterer
    detectors.Duplicates
    detectors.Linter
```

:::
### _Data Monitoring_
#### Drift
:::
*Detectors*

```{eval-rst}
.. autosummary::

    detectors.DriftCVM
    detectors.DriftKS
    detectors.DriftUncertainty
    detectors.DriftMMD
```

*Kernels*

```{eval-rst}
.. autosummary::

    detectors.GaussianRBF
```

*Update Strategies*

```{eval-rst}
.. autosummary::

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

## Metrics

:::{toctree}
:hidden:
:titlesonly:

metrics/balance
metrics/ber
metrics/channelstats
metrics/coverage
metrics/divergence
metrics/diversity
metrics/imagestats
metrics/parity
metrics/parity_metadata
metrics/uap
:::

Metrics are a way to measure the performance of your models or datasets that can
then be analyzed in the context of a given problem

:::
### _Data Exploration_
:::

```{eval-rst}
.. autosummary::

    metrics.channelstats
    metrics.imagestats
```
:::
### _Metadata/Label Exploration_
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
### _Data Performance_
:::

```{eval-rst}
.. autosummary::

    metrics.ber
    metrics.uap
```

## Flags

:::{toctree}
:hidden:
:titlesonly:

flags/imagestat
:::

Flags are used by the `imagestats` and `channelstats` functions, as well as the `Linter` and `Duplicates` classes

```{eval-rst}
.. autosummary::

    flags.ImageStat
```

## Workflows

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

## Supported Model Backends

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
#### Model Trainer
:::

```{eval-rst}
.. autosummary::

    models.torch.AETrainer
```

:::
#### Models
:::

```{eval-rst}
.. autosummary::

    models.torch.AriaAutoencoder
    models.torch.Decoder
    models.torch.Encoder
```

:::
### Tensorflow
:::

The Tensorflow models provided are tailored for usage with the out of distribution detection metrics.
DataEval provides both basic default models through the utility function `create_model` as well as 
constructors which allow for customization of the encoder, decoder and any other applicable layers
used by the model.

:::
#### Models
:::

```{eval-rst}
.. autosummary::

    models.tensorflow.AE
    models.tensorflow.AEGMM
    models.tensorflow.PixelCNN
    models.tensorflow.VAE
    models.tensorflow.VAEGMM
```

:::
#### Reconstruction Functions
:::

```{eval-rst}
.. autosummary::

    models.tensorflow.eucl_cosim_features
```

:::
#### Loss Function Classes
:::

```{eval-rst}
.. autosummary::

    models.tensorflow.Elbo
    models.tensorflow.LossGMM
```

:::
#### Utility Functions
:::

```{eval-rst}
.. autosummary::

    models.tensorflow.create_model
```

:::{toctree}
:maxdepth: 1
:hidden:

models/torch
models/tensorflow
:::
