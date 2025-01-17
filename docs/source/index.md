# DataEval

::::{grid}
:reverse:
:gutter: 3 4 4 4
:margin: 1 2 1 2

:::{grid-item}
:columns: 12 4 4 4

:::{image} _static/DataEval_Logo.png
:width: 200px
:class: sd-m-auto

:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-fs-5

```{include} ../../README.md
:start-after: <!-- start tagline -->
:end-before: <!-- end tagline -->
```

:::
::::

## Our Mission

:::{include} ../../README.md
:start-after: <!-- start needs -->
:end-before: <!-- end needs -->
:::

:::{include} ../../README.md
:start-after: <!-- start JATIC interop -->
:end-before: <!-- end JATIC interop -->
:::

## Key Features

DataEval provides many powerful tools to assist in the following T&E tasks:

- **Model-agnostic metrics that bound real-world performance**
  - relevance/completeness/coverage
  - metafeatures (data complexity)
- **Model-specific metrics that guide model selection and training**
  - dataset sufficiency
  - data/model complexity mismatch
- **Metrics for post-deployment monitoring of data with bounds on model
  performance to guide retraining**
  - dataset-shift metrics
  - model performance bounds under covariate shift
  - guidance on sampling to assess model error and model retraining

## Acknowledgement

:::{include} ../../README.md
:start-after: <!-- start acknowledgement -->
:end-before: <!-- end acknowledgement -->
:::

<!-- markdownlint-disable MD033 -->

<!-- TOC TREE -->

:::{toctree}
:caption: Home
:hidden:

Welcome <self>
QuickStart <home/quickstart.md>
home/installation.md
home/contributing.md
Change Log <home/changelog.md>
:::

:::{toctree}
:caption: Tutorials
:hidden:

Overview <tutorials/index>
:::

:::{toctree}
:caption: How-to Guides
:hidden:

Overview <how_to/index>
:::

:::{toctree}
:caption: Explanation
:hidden:

Overview <concepts/index>
:::

:::{toctree}
:caption: Reference
:hidden:

API Reference <reference/autoapi/dataeval/index>
reference/glossary
:::
