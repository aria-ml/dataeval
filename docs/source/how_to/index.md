# How-to Guides

:::{warning}
The How Tos are WIP and are expected to be heavily modified in the future
:::

These guides demonstrate more in-depth features and customizations of DataEval
features for more advanced users.

In addition to viewing them in our documentation, these notebooks can also be
opened in Google Colab to be used interactively!

## General Usage

These guides will provide quick examples of how to configure DataEval for your environment.

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

[hdw-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_configure_hardware_settings.ipynb
[log-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_configure_logging.ipynb

## Encoding

These guides provide examples of how to extract Embeddings using the Encoder classes.

```{toctree}
:hidden:

../notebooks/h2_encode_with_onnx.md
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_encode_with_onnx.md)
  - Encode image embeddings with an ONNX model
  - [![Open In Colab][colab-badge]][onx-colab]

:::

[onx-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_encode_with_onnx.md

## Detectors

The purpose of these tools is to identify or detect issues within a dataset.
The guides below exemplify powerful solutions to common problems in ML.

```{toctree}
:hidden:

../notebooks/h2_cluster_analysis.md
../notebooks/h2_deduplicate.md
../notebooks/h2_visualize_cleaning_issues.md
../notebooks/h2_custom_image_stats_object_detection.md
../notebooks/h2_choose_duplicate_detection_method.md
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

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
- - [](../notebooks/h2_choose_duplicate_detection_method.md)
  - Choose a duplicate detection method
  - [![Open In Colab][colab-badge]][bovw-colab]

:::

[clst-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_cluster_analysis.ipynb
[dupe-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_deduplicate.ipynb
[clean-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_visualize_cleaning_issues.ipynb
[calc-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_custom_image_stats_object_detection.ipynb
[bovw-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_choose_duplicate_detection_method.ipynb

## Metrics

Metrics are a set of tools that measure and analyze data.
The guides below show best practices when solving common ML problems.

:::{toctree}
:caption: Metrics
:hidden:

../notebooks/h2_measure_ic_feasibility.md
../notebooks/h2_measure_divergence.md
../notebooks/h2_measure_label_independence.md
../notebooks/h2_detect_undersampling.md
../notebooks/h2_add_intrinsic_factors.md
:::

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_measure_ic_feasibility.md)
  - Calculate feasibility of performance requirements on
    different datasets using {term}`Bayes Error Rate (BER)`
  - [![Open In Colab][colab-badge]][ber-colab]
- - [](../notebooks/h2_measure_divergence.md)
  - Display data distributions between 2 datasets
  - [![Open In Colab][colab-badge]][div-colab]
- - [](../notebooks/h2_measure_label_independence.md)
  - Compare label distributions between 2 datasets
  - [![Open In Colab][colab-badge]][lbl-colab]
- - [](../notebooks/h2_detect_undersampling.md)
  - Detect undersampled subsets of datasets
  - [![Open In Colab][colab-badge]][cov-colab]
- - [](../notebooks/h2_add_intrinsic_factors.md)
  - Apply DataEval's statistical outputs to
    DataEval's {class}`.Metadata` object for bias analysis
  - [![Open In Colab][colab-badge]][imd-colab]

:::

[ber-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_measure_ic_feasibility.ipynb
[div-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_measure_divergence.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_measure_label_independence.ipynb
[cov-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_detect_undersampling.ipynb
[imd-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_add_intrinsic_factors.ipynb

## Workflows

Workflows are end-to-end processes that detect, measure,
and analyze data against requirements.
The guides below help you solve common problems found across machine learning tasks.

:::{toctree}
:caption: Workflows
:hidden:

../notebooks/h2_measure_ic_sufficiency.md
../notebooks/h2_dataset_splits.md
:::

:::{list-table}
:widths: 30 50 20
:header-rows: 0
:align: center

- - [](../notebooks/h2_measure_ic_sufficiency.md)
  - Determine the amount of data needed to meet
    image classification performance requirements
  - [![Open In Colab][colab-badge]][suff-colab]

:::

[suff-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.95.0/docs/source/notebooks/h2_measure_ic_sufficiency.ipynb

<!-- Google collab badge icon for all collab links -->

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
