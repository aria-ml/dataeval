# API Reference

## Detectors

Detectors can determine if a dataset or individual images in a dataset are indicative of a specific issue.

### Drift

[Drift](../concepts/glossary.md#drift) detectors identify if the statistical properties of the data has changed.

```{eval-rst}
.. autosummary::
   :toctree: drift

   dataeval.detectors.drift.DriftCVM
   dataeval.detectors.drift.DriftKS
   dataeval.detectors.drift.DriftUncertainty
   dataeval.detectors.drift.DriftMMD
```

#### _Kernels_

Kernels are used to map non-linear data to a higher dimensional space.

```{eval-rst}
.. autosummary::
   :toctree: drift

   dataeval.detectors.drift.kernels.GaussianRBF
```

#### _Updates_

Update strategies inform how the drift detector classes update the reference data when monitoring for drift.

```{eval-rst}
.. autosummary::
   :toctree: drift

   dataeval.detectors.drift.updates.LastSeenUpdate
   dataeval.detectors.drift.updates.ReservoirSamplingUpdate
```

### Linters

[Linters](../concepts/glossary.md#linter) help identify potential issues in training and test data and are an important aspect of [data cleaning](../concepts/DataCleaning.md).

```{eval-rst}
.. autosummary::
   :toctree: linters

   dataeval.detectors.linters.Clusterer
   dataeval.detectors.linters.Duplicates
   dataeval.detectors.linters.Outliers
```

### Out-of-Distribution

[Out-of-distribution](../concepts/glossary.md#out-of-distribution-ood) detectors identify data that is different from the data used to train a particular model.

```{eval-rst}
.. autosummary::
   :toctree: ood

   dataeval.detectors.ood.OOD_AE
   dataeval.detectors.ood.OOD_AEGMM
   dataeval.detectors.ood.OOD_LLR
   dataeval.detectors.ood.OOD_VAE
   dataeval.detectors.ood.OOD_VAEGMM
   dataeval.detectors.ood.OODScore
```

## Flags

Flags are used by the [`imagestats`](metrics/dataeval.metrics.stats.imagestats.rst) and [`channelstats`](metrics/dataeval.metrics.stats.channelstats.rst) functions, as well as the [`Outliers`](linters/dataeval.detectors.linters.Outliers.rst) and [`Duplicates`](linters/dataeval.detectors.linters.Duplicates.rst) classes.

```{eval-rst}
.. autosummary::
   :toctree: flags

   dataeval.flags.ImageStat
```

## Metrics

Metrics are a way to measure the performance of your models or datasets that can then be analyzed in the context of a given problem.

### Bias

[Bias](../concepts/glossary.md#bias) metrics check for skewed or imbalanced datasets and incomplete feature representation which may impact model performance.

```{eval-rst}
.. autosummary::
   :toctree: bias

   dataeval.metrics.bias.balance
   dataeval.metrics.bias.coverage
   dataeval.metrics.bias.diversity
   dataeval.metrics.bias.label_parity
   dataeval.metrics.bias.parity
```

### Estimators

Estimators calculate performance bounds and the statistical distance between datasets.

```{eval-rst}
.. autosummary::
   :toctree: metrics

   dataeval.metrics.estimators.ber
   dataeval.metrics.estimators.divergence
   dataeval.metrics.estimators.uap
```

### Statistics

Statistics metrics calculate a variety of image properties and pixel statistics against the image and individual channels of an image.

```{eval-rst}
.. autosummary::
   :toctree: metrics

   dataeval.metrics.stats.channelstats
   dataeval.metrics.stats.imagestats
```

## Workflows

Workflows perform a sequence of actions to analyze the dataset and make predictions.

```{eval-rst}
.. autosummary::
   :toctree: workflows

   dataeval.workflows.Sufficiency
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
   :toctree: torch

   dataeval.torch.models.AriaAutoencoder
   dataeval.torch.models.Decoder
   dataeval.torch.models.Encoder
```

:::
#### _Trainer_
:::

```{eval-rst}
.. autosummary::
   :toctree: torch

   dataeval.torch.trainer.AETrainer
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
   :toctree: tensorflow

   dataeval.tensorflow.models.AE
   dataeval.tensorflow.models.AEGMM
   dataeval.tensorflow.models.PixelCNN
   dataeval.tensorflow.models.VAE
   dataeval.tensorflow.models.VAEGMM
   dataeval.tensorflow.models.create_model
```

:::
#### _Reconstruction Functions_
:::

```{eval-rst}
.. autosummary::
   :toctree: tensorflow

   dataeval.tensorflow.recon.eucl_cosim_features
```

:::
#### _Loss Function Classes_
:::

```{eval-rst}
.. autosummary::
   :toctree: tensorflow
   
   dataeval.tensorflow.loss.Elbo
   dataeval.tensorflow.loss.LossGMM
```

:::{toctree}
:maxdepth: 1
:hidden:

backend/torch
backend/tensorflow
:::
