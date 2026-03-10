# How-to Guides

These guides help you accomplish specific tasks with DataEval. Each one addresses
a practical problem and walks you through the solution step by step.

In addition to viewing them in our documentation, these notebooks can also be
opened in Google Colab to be used interactively!

The guides are organized by where they fall in the
[machine learning life cycle](../getting-started/roles/ML_Lifecycle.md):

1. [Configuration](#configuration)
2. [Data engineering](#data-engineering)
3. [Model development](#model-development)
4. [Monitoring](#monitoring)

## Configuration

These guides will provide quick examples of how to configure DataEval for your
environment.

```{toctree}
:hidden:

../notebooks/h2_configure_hardware_settings.py
../notebooks/h2_configure_logging.py
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_configure_hardware_settings.py)
  - Configure global hardware settings used in DataEval
  - [![Open In Colab][colab-badge]][hdw-colab]
- - [](../notebooks/h2_configure_logging.py)
  - Configure logging with DataEval
  - [![Open In Colab][colab-badge]][log-colab]

:::

[hdw-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_configure_hardware_settings.ipynb
[log-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_configure_logging.ipynb

## Data Engineering

These guides cover tasks related to preparing, cleaning, exploring, and curating
datasets for machine learning.

```{toctree}
:hidden:

../notebooks/h2_encode_with_onnx.py
../notebooks/h2_cluster_analysis.py
../notebooks/h2_deduplicate.py
../notebooks/h2_visualize_cleaning_issues.py
../notebooks/h2_custom_image_stats_object_detection.py
../notebooks/h2_add_intrinsic_factors.py
../notebooks/h2_detect_undersampling.py
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_encode_with_onnx.py)
  - Encode image embeddings with an ONNX model
  - [![Open In Colab][colab-badge]][onx-colab]
- - [](../notebooks/h2_cluster_analysis.py)
  - Identify outliers and anomalies with clustering algorithms
  - [![Open In Colab][colab-badge]][clst-colab]
- - [](../notebooks/h2_deduplicate.py)
  - Identify and remove duplicates from a PyTorch Dataset
  - [![Open In Colab][colab-badge]][dupe-colab]
- - [](../notebooks/h2_visualize_cleaning_issues.py)
  - Find negatively impactful images in multiple backgrounds
  - [![Open In Colab][colab-badge]][clean-colab]
- - [](../notebooks/h2_custom_image_stats_object_detection.py)
  - Customize calculation of image stats on an object detection dataset
  - [![Open In Colab][colab-badge]][calc-colab]
- - [](../notebooks/h2_add_intrinsic_factors.py)
  - Apply DataEval's statistical outputs to
    DataEval's {class}`.Metadata` object for bias analysis
  - [![Open In Colab][colab-badge]][imd-colab]
- - [](../notebooks/h2_detect_undersampling.py)
  - Detect undersampled subsets of datasets
  - [![Open In Colab][colab-badge]][cov-colab]

:::

[onx-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_encode_with_onnx.ipynb
[clst-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_cluster_analysis.ipynb
[dupe-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_deduplicate.ipynb
[clean-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_visualize_cleaning_issues.ipynb
[calc-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_custom_image_stats_object_detection.ipynb
[imd-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_add_intrinsic_factors.ipynb
[cov-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_detect_undersampling.ipynb

## Model Development

These guides cover tasks related to assessing data feasibility and sufficiency
for model training.

```{toctree}
:hidden:

../notebooks/h2_measure_ic_feasibility.py
../notebooks/h2_measure_ic_sufficiency.py
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_measure_ic_feasibility.py)
  - Calculate feasibility of performance requirements on
    different datasets using {term}`Bayes Error Rate (BER)`
  - [![Open In Colab][colab-badge]][ber-colab]
- - [](../notebooks/h2_measure_ic_sufficiency.py)
  - Determine the amount of data needed to meet
    image classification performance requirements
  - [![Open In Colab][colab-badge]][suff-colab]

:::

[ber-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_measure_ic_feasibility.ipynb
[suff-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_measure_ic_sufficiency.ipynb

## Monitoring

These guides cover tasks related to comparing datasets and detecting distribution
shifts in deployed systems.

```{toctree}
:hidden:

../notebooks/h2_measure_divergence.py
../notebooks/h2_measure_label_independence.py
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_measure_divergence.py)
  - Display data distributions between 2 datasets
  - [![Open In Colab][colab-badge]][div-colab]
- - [](../notebooks/h2_measure_label_independence.py)
  - Compare label distributions between 2 datasets
  - [![Open In Colab][colab-badge]][lbl-colab]

:::

[div-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_measure_divergence.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/h2_measure_label_independence.ipynb

<!-- Google collab badge icon for all collab links -->

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
