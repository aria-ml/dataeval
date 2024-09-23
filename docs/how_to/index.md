How-to Guides
=========

These guides demonstrate more in-depth features and customizations of DataEval features for more advanced users.

In addition to viewing them in our documentation, these notebooks can also be opened in Google Colab to be used interactively!

Detectors
---------

- [How to detect if the data distribution is changing](notebooks/DriftDetectionTutorial) [![Open In Colab][colab-badge]][drift-colab]
- [How to monitor for outliers during deployment](notebooks/OODDetectionTutorial) [![Open In Colab][colab-badge]][out-colab]
- [How to identify outliers and/or anomalies in a dataset](notebooks/ClustererTutorial) [![Open In Colab][colab-badge]][clust-colab]
- [How to detect duplicates in a dataset](notebooks/DuplicatesTutorial) [![Open In Colab][colab-badge]][dup-colab]
- [How to identify poor quality images in a dataset](notebooks/LintingTutorial) [![Open In Colab][colab-badge]][lint-colab]

:::{toctree}
:hidden:
:maxdepth: 1

notebooks/DriftDetectionTutorial.ipynb
notebooks/OODDetectionTutorial.ipynb
notebooks/ClustererTutorial.ipynb
notebooks/DuplicatesTutorial.ipynb
notebooks/LintingTutorial.ipynb
:::

Metrics
-------

- [How to determine if a dataset can meet performance requirements](notebooks/BayesErrorRateEstimationTutorial) [![Open In Colab][colab-badge]][ber-colab]
- [How to compare data distributions between 2 datasets](notebooks/HPDivergenceTutorial) [![Open In Colab][colab-badge]][div-colab]
- [How to compare label distributions between 2 datasets](notebooks/ClassLabelAnalysisTutorial) [![Open In Colab][colab-badge]][lbl-colab]
- [How to detect undersampled data subsets](notebooks/CoverageTutorial) [![Open In Colab][colab-badge]][cov-colab]

:::{toctree}
:hidden:
:maxdepth: 1

notebooks/BayesErrorRateEstimationTutorial.ipynb
notebooks/HPDivergenceTutorial.ipynb
notebooks/ClassLabelAnalysisTutorial.ipynb
notebooks/CoverageTutorial.ipynb
:::

Workflows
---------

- [How to determine the amount of data needed to meet image classification performance requirements](notebooks/ClassLearningCurvesTutorial) [![Open In Colab][colab-badge]][suff-colab]
- [How to determine the amount of data needed to meet object detection performance requirements](html/ODLearningCurvesTutorial)

:::{toctree}
:hidden:
:maxdepth: 1

notebooks/ClassLearningCurvesTutorial.ipynb
html/ODLearningCurvesTutorial.md
:::

Models
------

- [How to create image embeddings with an autoencoder](notebooks/AETrainerTutorial) [![Open In Colab][colab-badge]][ae-colab]
% - How to use the AETrainer with different model architectures (future how-to)

:::{toctree}
:hidden:
:maxdepth: 1

notebooks/AETrainerTutorial.ipynb
:::

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[ber-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/BayesErrorRateEstimationTutorial.ipynb
[suff-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/ClassLearningCurvesTutorial.ipynb
[div-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/HPDivergenceTutorial.ipynb
[drift-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/DriftDetectionTutorial.ipynb
[out-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/OODDetectionTutorial.ipynb
[ae-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/AETrainerTutorial.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/ClassLabelAnalysisTutorial.ipynb
[odlc-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/ODLearningCurvesTutorial.ipynb
[clust-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/ClustererTutorial.ipynb
[dup-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/DuplicatesTutorial.ipynb
[lint-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/LintingTutorial.ipynb
[cov-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.69.4/docs/how_to/notebooks/CoverageTutorial.ipynb