# How-to Guides

:::{warning}
The How Tos are WIP and are expected to be heavily modified in the future
:::

These guides demonstrate more in-depth features and customizations of DataEval
features for more advanced users.

In addition to viewing them in our documentation, these notebooks can also be
opened in Google Colab to be used interactively!

## Detectors

The purpose of these tools is to identify or detect issues within a dataset.
The guides below exemplify powerful solutions to common problems in ML.

```{toctree}
:hidden:

../notebooks/h2_cluster_analysis.ipynb
../notebooks/h2_deduplicate.ipynb
../notebooks/h2_visualize_linting_issues.ipynb
```

:::{list-table}
:widths: 20 60 20
:header-rows: 0

- - [](../notebooks/h2_cluster_analysis.ipynb)
  - Identify outliers and anomalies with clustering algorithms
  - [![Open In Colab][colab-badge]][clst-colab]
- - [](../notebooks/h2_deduplicate.ipynb)
  - Identify and remove duplicates from a PyTorch Dataset
  - [![Open In Colab][colab-badge]][dupe-colab]
- - [](../notebooks/h2_visualize_linting_issues.ipynb)
  - Find negatively impactful images in multiple backgrounds
  - [![Open In Colab][colab-badge]][lint-colab]

:::

[clst-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_cluster_analysis.ipynb
[dupe-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_deduplicate.ipynb
[lint-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_visualize_linting_issues.ipynb

## Metrics

Metrics are a set of tools that measure and analyze data.
The guides below show best practices when solving common ML problems.

:::{toctree}
:caption: Metrics
:hidden:

../notebooks/h2_measure_ic_feasibility.ipynb
../notebooks/h2_measure_divergence.ipynb
../notebooks/h2_measure_label_independence.ipynb
../notebooks/h2_detect_undersampling.ipynb
../notebooks/h2_add_intrinsic_factors.ipynb
:::

:::{list-table}
:widths: 20 60 20
:header-rows: 0

- - [](../notebooks/h2_measure_ic_feasibility.ipynb)
  - Calculate feasibility of performance requirements on
    different datasets using {term}`Bayes Error Rate (BER)`
  - [![Open In Colab][colab-badge]][ber-colab]
- - [](../notebooks/h2_measure_divergence.ipynb)
  - Display data distributions between 2 datasets
  - [![Open In Colab][colab-badge]][div-colab]
- - [](../notebooks/h2_measure_label_independence.ipynb)
  - Compare label distributions between 2 datasets
  - [![Open In Colab][colab-badge]][lbl-colab]
- - [](../notebooks/h2_detect_undersampling.ipynb)
  - Detect undersampled subsets of datasets
  - [![Open In Colab][colab-badge]][cov-colab]
- - [](../notebooks/h2_add_intrinsic_factors.ipynb)
  - Apply DataEval's statistical outputs to
    DataEval's {class}`.Metadata` object for bias analysis
  - [![Open In Colab][colab-badge]][imd-colab]

:::

[ber-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_measure_ic_feasibility.ipynb
[div-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_measure_divergence.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_measure_label_independence.ipynb
[cov-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_detect_undersampling.ipynb
[imd-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_add_intrinsic_factors.ipynb

## Workflows

Workflows are end-to-end processes that detect, measure,
and analyze data against requirements.
The guides below help you solve common problems found across machine learning tasks.

:::{toctree}
:caption: Workflows
:hidden:

../notebooks/h2_measure_ic_sufficiency.ipynb
:::

:::{list-table}
:widths: 20 60 20
:header-rows: 0

- - [](../notebooks/h2_measure_ic_sufficiency.ipynb)
  - Determine the amount of data needed to meet
    image classification performance requirements
  - [![Open In Colab][colab-badge]][suff-colab]

:::

[suff-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_measure_ic_sufficiency.ipynb

## Models

DataEval uses models during all stages of the ML Lifecycle.
The guides below show specific examples on model usage at different levels of expertise.

:::{toctree}
:caption: Models
:hidden:

../notebooks/h2_train_ae_embeddings.ipynb
:::

:::{list-table}
:widths: 20 60 20
:header-rows: 0

- - [](../notebooks/h2_train_ae_embeddings.ipynb)
  - Train and evaluate an autoencoder to generate effective
    image embeddings for downstream tasks
  - [![Open In Colab][colab-badge]][ae-colab]

:::

[ae-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.8/docs/source/notebooks/h2_train_ae_embeddings.ipynb

<!-- Google collab badge icon for all collab links -->

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
