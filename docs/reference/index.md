Reference Guide
===============

Detectors
---------

Detectors can determine if a dataset or individual images in a dataset are indicative of a specific issue

:::{toctree}
:maxdepth: 1

detectors/clusterer
detectors/drift
detectors/duplicates
detectors/linter
detectors/ood
:::

Metrics
-------

Metrics are a way to measure the performance of your models or datasets that can
then be analyzed in the context of a given problem

:::{toctree}
:maxdepth: 1

metrics/ber
metrics/divergence
metrics/parity
metrics/stats
metrics/uap
:::

Workflows
-------

Workflows perform a sequence of actions to analyze the dataset and make predictions

:::{toctree}
:maxdepth: 1

workflows/sufficiency
:::

Supported Model Backends
------------------------

The models and model trainers provided by DataEval are meant to assist users in setting up
architectures that are guaranteed to work with applicable DataEval metrics.
Below is a list of backends with available trainers and models. 

:::{toctree}
:maxdepth: 1

models/torch
models/tensorflow
:::