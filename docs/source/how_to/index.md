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

nbs/h2_cluster_analysis.ipynb
nbs/h2_deduplicate.ipynb
LintingTutorial.ipynb
```

:::{list-table}
:widths: 20 60 20
:header-rows: 0

* * [](nbs/h2_cluster_analysis.ipynb)
  * Identify outliers and anomalies with clustering algorithms
  * [![Open In Colab][colab-badge]][clust-colab]
* * [](nbs/h2_deduplicate.ipynb)
  * Identify and remove duplicates from a PyTorch Dataset
  * [![Open In Colab][colab-badge]][dup-colab]
* * [](LintingTutorial.ipynb)
  * Find negatively impactful images in multiple backgrounds
  * [![Open In Colab][colab-badge]][lint-colab]

:::

[clust-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/nbs/h2_cluster_analysis.ipynb
[dup-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/nbs/h2_deduplicate.ipynb
[lint-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/LintingTutorial.ipynb

## Metrics

Metrics are a set of tools that measure and analyze data.
The guides below show best practices when solving common ML problems.

:::{toctree}
:caption: Metrics
:hidden:

BayesErrorRateEstimationTutorial.ipynb
HPDivergenceTutorial.ipynb
ClassLabelAnalysisTutorial.ipynb
CoverageTutorial.ipynb
IntrinsicMetadata.ipynb
:::

:::{list-table}
:widths: 20 60 20
:header-rows: 0

* * [](BayesErrorRateEstimationTutorial.ipynb)
  * Calculate feasibility of performance requirements on
  different datasets using {term}`Bayes Error Rate (BER)`
  * [![Open In Colab][colab-badge]][ber-colab]
* * [](HPDivergenceTutorial.ipynb)
  * Display data distributions between 2 datasets
  * [![Open In Colab][colab-badge]][div-colab]
* * [](ClassLabelAnalysisTutorial.ipynb)
  * Compare label distributions between 2 datasets
  * [![Open In Colab][colab-badge]][lbl-colab]
* * [](CoverageTutorial.ipynb)
  * Detect undersampled subsets of datasets
  * [![Open In Colab][colab-badge]][cov-colab]
* * [](IntrinsicMetadata.ipynb)
  * Apply DataEval's statistical outputs to
  DataEval's {class}`.Metadata` object for bias analysis
  * [![Open In Colab][colab-badge]][imd-colab]

:::

[ber-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/BayesErrorRateEstimationTutorial.ipynb
[div-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/HPDivergenceTutorial.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/ClassLabelAnalysisTutorial.ipynb
[cov-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/CoverageTutorial.ipynb
[imd-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/IntrinsicMetadata.ipynb

## Workflows

Workflows are end-to-end processes that detect, measure,
and analyze data against requirements.
The guides below help you solve common problems found across machine learning tasks.

:::{toctree}
:caption: Workflows
:hidden:

ClassLearningCurvesTutorial.ipynb
:::

:::{list-table}
:widths: 20 60 20
:header-rows: 0

* * [](ClassLearningCurvesTutorial.ipynb)
  * Determine the amount of data needed to meet
  image classification performance requirements
  * [![Open In Colab][colab-badge]][suff-colab]

:::

[suff-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/ClassLearningCurvesTutorial.ipynb

## Models

DataEval uses models during all stages of the ML Lifecycle.
The guides below show specific examples on model usage at different levels of expertise.

:::{toctree}
:caption: Models
:hidden:

AETrainerTutorial.ipynb
:::

:::{list-table}
:widths: 20 60 20
:header-rows: 0

* * [](AETrainerTutorial.ipynb)
  * Train and evaluate an autoencoder to generate effective
  image embeddings for downstream tasks
  * [![Open In Colab][colab-badge]][ae-colab]

:::

[ae-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.0/docs/source/how_to/AETrainerTutorial.ipynb

<!-- Google collab badge icon for all collab links -->
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
