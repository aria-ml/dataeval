How-to Guides
=========

These guides demonstrate more in-depth features and customizations of DataEval features for more advanced users.

In addition to viewing them in our documentation, these notebooks can also be opened in Google Colab to be used interactively!

Detectors
---------

- [How to Detect if the Data Distribution has Changed](notebooks/DriftDetectionTutorial) [![Open In Colab][colab-badge]][drift-colab]
- [How to Identify Outliers and/or Anomalies through Image Distribution Shifts](notebooks/OODDetectionTutorial) [![Open In Colab][colab-badge]][out-colab]
- [How to Identify Outliers and/or Anomalies through Clustering](notebooks/ClustererTutorial) [![Open In Colab][colab-badge]][clust-colab]
- [How to Identify Duplicates in a Dataset](notebooks/DuplicatesTutorial) [![Open In Colab][colab-badge]][dup-colab]
- [How to Identify Poor Quality Images in a Datasetl](notebooks/LintingTutorial) [![Open In Colab][colab-badge]][lint-colab]

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

- [How to Determine if a Dataset is Adequate for Training to Meet a Specific Model Requirement](notebooks/BayesErrorRateEstimationTutorial) [![Open In Colab][colab-badge]][ber-colab]
- [How to Detect if Additional Data Aligns with Previous Data](notebooks/HPDivergenceTutorial) [![Open In Colab][colab-badge]][div-colab]
- [How to Compare Label Distributions bewteen 2 Datasets](notebooks/ClassLabelAnalysisTutorial) [![Open In Colab][colab-badge]][lbl-colab]
- [How to Identify Undersampled Image Subsets](notebooks/CoverageTutorial) [![Open In Colab][colab-badge]][cov-colab]

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

- [How to Determine if there is Enough Data for Training to Meet a Specific Model Requirement for Image Classification](notebooks/ClassLearningCurvesTutorial) [![Open In Colab][colab-badge]][suff-colab]
- [How to Determine if there is Enough Data for Training to Meet a Specific Model Requirement for Object Detection](html/ODLearningCurvesTutorial) [![Open In Colab][colab-badge]][odlc-colab]

:::{toctree}
:hidden:
:maxdepth: 1

notebooks/ClassLearningCurvesTutorial.ipynb
html/ODLearningCurvesTutorial.md
:::

Models
------

- [How to Create Image Embeddings with an AutoEncoder](notebooks/AETrainerTutorial) [![Open In Colab][colab-badge]][ae-colab]
% - How to use the AETrainer with Different Model Architectures (future how-to)

:::{toctree}
:hidden:
:maxdepth: 1

notebooks/AETrainerTutorial.ipynb
:::

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[ber-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/BayesErrorRateEstimationTutorial.ipynb
[suff-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/ClassLearningCurvesTutorial.ipynb
[div-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/HPDivergenceTutorial.ipynb
[drift-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/DriftDivergenceTutorial.ipynb
[out-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/OODDetectionTutorial.ipynb
[ae-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/AETrainerTutorial.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/ClassLabelAnalysisTutorial.ipynb
[odlc-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/ODLearningCurvesTutorial.ipynb
[clust-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/ClustererTutorial.ipynb
[dup-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/DuplicatesTutorial.ipynb
[lint-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/LintingTutorial.ipynb
[cov-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/main/docs/how_to/notebooks/CoverageTutorial.ipynb
