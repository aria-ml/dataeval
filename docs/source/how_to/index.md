# How-to Guides

These guides help you accomplish specific tasks with DataEval. Each one addresses
a practical problem and walks you through the solution step by step.

In addition to viewing them in our documentation, these notebooks can also be
opened in Google Colab to be used interactively!

The guides are organized by where they fall in the
[machine learning life cycle](../concepts/users/ML_Lifecycle.md):

1. [Configuration](#configuration)
2. [Data engineering](#data-engineering)
3. [Model development](#model-development)
4. [Monitoring](#monitoring)

## Configuration

These guides will provide quick examples of how to configure DataEval for your
environment.

```{toctree}
:hidden:

../notebooks/h2_configure_hardware_settings.md
../notebooks/h2_configure_logging.md
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_configure_hardware_settings.md)
  - Configure global hardware settings used in DataEval
  - [![Open In Colab][colab-badge]][hdw-colab]
- - [](../notebooks/h2_configure_logging.md)
  - Configure logging with DataEval
  - [![Open In Colab][colab-badge]][log-colab]

:::

[hdw-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_configure_hardware_settings.ipynb
[log-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_configure_logging.ipynb

## Data Engineering

These guides cover tasks related to preparing, cleaning, exploring, and curating
datasets for machine learning.

```{toctree}
:hidden:

../notebooks/h2_encode_with_onnx.md
../notebooks/h2_cluster_analysis.md
../notebooks/h2_deduplicate.md
../notebooks/h2_visualize_cleaning_issues.md
../notebooks/h2_custom_image_stats_object_detection.md
../notebooks/h2_add_intrinsic_factors.md
../notebooks/h2_detect_undersampling.md
../notebooks/h2_dataset_splits.md
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_encode_with_onnx.md)
  - Encode image embeddings with an ONNX model
  - [![Open In Colab][colab-badge]][onx-colab]
- - [](../notebooks/h2_cluster_analysis.md)
  - Identify outliers and anomalies with clustering algorithms
  - [![Open In Colab][colab-badge]][clst-colab]
- - [](../notebooks/h2_deduplicate.md)
  - Identify and remove duplicates from a PyTorch Dataset
  - [![Open In Colab][colab-badge]][dupe-colab]
- - [](../notebooks/h2_visualize_cleaning_issues.md)
  - Find negatively impactful images in multiple backgrounds
  - [![Open In Colab][colab-badge]][clean-colab]
- - [](../notebooks/h2_custom_image_stats_object_detection.md)
  - Customize calculation of image stats on an object detection dataset
  - [![Open In Colab][colab-badge]][calc-colab]
- - [](../notebooks/h2_add_intrinsic_factors.md)
  - Apply DataEval's statistical outputs to
    DataEval's {class}`.Metadata` object for bias analysis
  - [![Open In Colab][colab-badge]][imd-colab]
- - [](../notebooks/h2_detect_undersampling.md)
  - Detect undersampled subsets of datasets
  - [![Open In Colab][colab-badge]][cov-colab]
- - [](../notebooks/h2_dataset_splits.md)
  - Split your dataset for training and evaluation
  -

:::

[onx-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_encode_with_onnx.ipynb
[clst-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_cluster_analysis.ipynb
[dupe-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_deduplicate.ipynb
[clean-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_visualize_cleaning_issues.ipynb
[calc-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_custom_image_stats_object_detection.ipynb
[imd-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_add_intrinsic_factors.ipynb
[cov-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_detect_undersampling.ipynb

## Model Development

These guides cover tasks related to assessing data feasibility and sufficiency
for model training.

```{toctree}
:hidden:

../notebooks/h2_measure_ic_feasibility.md
../notebooks/h2_measure_ic_sufficiency.md
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_measure_ic_feasibility.md)
  - Calculate feasibility of performance requirements on
    different datasets using {term}`Bayes Error Rate (BER)`
  - [![Open In Colab][colab-badge]][ber-colab]
- - [](../notebooks/h2_measure_ic_sufficiency.md)
  - Determine the amount of data needed to meet
    image classification performance requirements
  - [![Open In Colab][colab-badge]][suff-colab]

:::

[ber-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_measure_ic_feasibility.ipynb
[suff-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_measure_ic_sufficiency.ipynb

## Monitoring

These guides cover tasks related to comparing datasets and detecting distribution
shifts in deployed systems.

```{toctree}
:hidden:

../notebooks/h2_measure_divergence.md
../notebooks/h2_measure_label_independence.md
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_measure_divergence.md)
  - Display data distributions between 2 datasets
  - [![Open In Colab][colab-badge]][div-colab]
- - [](../notebooks/h2_measure_label_independence.md)
  - Compare label distributions between 2 datasets
  - [![Open In Colab][colab-badge]][lbl-colab]

:::

[div-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_measure_divergence.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/h2_measure_label_independence.ipynb

<!-- Google collab badge icon for all collab links -->

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
