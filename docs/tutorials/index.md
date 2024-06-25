Tutorials
=========

These static jupyter notebooks demonstrate how to use DAML to perform data analysis tasks using various detectors, metrics and workflows to assess the suitability of a dataset and/or model.

In addition to viewing them in our documentation, these notebooks can also be opened in Google Colab to be used interactively!

Detectors
---------

- [Drift Detection Tutorial](notebooks/DriftDetectionTutorial) [![Open In Colab][colab-badge]][drift-colab]
- [Out-of-Distribution Detection Tutorial](notebooks/OODDetectionTutorial) [![Open In Colab][colab-badge]][out-colab]
- [Clustering Tutorial](notebooks/ClustererTutorial) [![Open In Colab][colab-badge]][clust-colab]
- [Duplicates Tutorial](notebooks/DuplicatesTutorial) [![Open In Colab][colab-badge]][dup-colab]
- [Linting Tutorial](notebooks/LintingTutorial) [![Open In Colab][colab-badge]][lint-colab]

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

- [Bayes Error Rate Estimation Tutorial](notebooks/BayesErrorRateEstimationTutorial) [![Open In Colab][colab-badge]][ber-colab]
- [HP Divergence Tutorial](notebooks/HPDivergenceTutorial) [![Open In Colab][colab-badge]][div-colab]
- [Class Label Analysis Tutorial](notebooks/ClassLabelAnalysisTutorial) [![Open In Colab][colab-badge]][lbl-colab]

:::{toctree}
:hidden:
:maxdepth: 1

notebooks/BayesErrorRateEstimationTutorial.ipynb
notebooks/HPDivergenceTutorial.ipynb
notebooks/ClassLabelAnalysisTutorial.ipynb
:::

Workflows
---------

- [Class Learning Curves Tutorial](notebooks/ClassLearningCurvesTutorial) [![Open In Colab][colab-badge]][suff-colab]
- [Dataset Sufficiency Analysis for Object Detection Tutorial](html/ODLearningCurvesTutorial.rst)

:::{toctree}
:hidden:
:maxdepth: 1

notebooks/ClassLearningCurvesTutorial.ipynb
html/ODLearningCurvesTutorial.rst
:::

Models
------

- [AE Trainer Tutorial](notebooks/AETrainerTutorial) [![Open In Colab][colab-badge]][ae-colab]

:::{toctree}
:hidden:
:maxdepth: 1

notebooks/AETrainerTutorial.ipynb
:::

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[ber-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/BayesErrorRateEstimationTutorial.ipynb
[suff-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/ClassLearningCurvesTutorial.ipynb
[div-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/HPDivergenceTutorial.ipynb
[drift-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/DriftDivergenceTutorial.ipynb
[out-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/OODDetectionTutorial.ipynb
[ae-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/AETrainerTutorial.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/ClassLabelAnalysisTutorial.ipynb
[clust-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/ClustererTutorial.ipynb
[dup-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/DuplicatesTutorial.ipynb
[lint-colab]: https://colab.research.google.com/github/aria-ml/daml/blob/main/docs/tutorials/notebooks/LintingTutorial.ipynb