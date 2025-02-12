# How-to Guides

These guides demonstrate more in-depth features and customizations of DataEval
features for more advanced users.

In addition to viewing them in our documentation, these notebooks can also be
opened in Google Colab to be used interactively!

## Detectors

- [How to identify outliers and/or anomalies in a dataset](ClustererTutorial.ipynb)
  [![Open In Colab][colab-badge]][clust-colab]
- [How to detect duplicates in a dataset](DuplicatesTutorial.ipynb)
  [![Open In Colab][colab-badge]][dup-colab]
- [How to identify poor quality images in a dataset](LintingTutorial.ipynb)
  [![Open In Colab][colab-badge]][lint-colab]

:::{toctree}
:caption: Detectors
:hidden:

ClustererTutorial.ipynb
DuplicatesTutorial.ipynb
LintingTutorial.ipynb
:::

## Metrics

- [How to determine if a dataset can meet performance requirements](BayesErrorRateEstimationTutorial.ipynb)
  [![Open In Colab][colab-badge]][ber-colab]
- [How to compare data distributions between 2 datasets](HPDivergenceTutorial.ipynb)
  [![Open In Colab][colab-badge]][div-colab]
- [How to compare label distributions between 2 datasets](ClassLabelAnalysisTutorial.ipynb)
  [![Open In Colab][colab-badge]][lbl-colab]
- [How to detect undersampled data subsets](CoverageTutorial.ipynb)
  [![Open In Colab][colab-badge]][cov-colab]

:::{toctree}
:caption: Metrics
:hidden:

BayesErrorRateEstimationTutorial.ipynb
HPDivergenceTutorial.ipynb
ClassLabelAnalysisTutorial.ipynb
CoverageTutorial.ipynb
:::

## Workflows

- [How to determine the amount of data needed to meet image classification performance requirements](ClassLearningCurvesTutorial.ipynb)
  [![Open In Colab][colab-badge]][suff-colab]

:::{toctree}
:caption: Workflows
:hidden:

ClassLearningCurvesTutorial.ipynb
:::

## Models

- [How to create image embeddings with an autoencoder](AETrainerTutorial.ipynb)
  [![Open In Colab][colab-badge]][ae-colab]
% - How to use the AETrainer with different model architectures (future how-to)

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[ber-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.1/docs/source/how_to/BayesErrorRateEstimationTutorial.ipynb
[suff-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.1/docs/source/how_to/ClassLearningCurvesTutorial.ipynb
[div-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.1/docs/source/how_to/HPDivergenceTutorial.ipynb
[ae-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.1/docs/source/how_to/AETrainerTutorial.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.1/docs/source/how_to/ClassLabelAnalysisTutorial.ipynb
[clust-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.1/docs/source/how_to/ClustererTutorial.ipynb
[dup-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.1/docs/source/how_to/DuplicatesTutorial.ipynb
[lint-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.1/docs/source/how_to/LintingTutorial.ipynb
[cov-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.1/docs/source/how_to/CoverageTutorial.ipynb

:::{toctree}
:caption: Models
:hidden:

AETrainerTutorial.ipynb
:::
