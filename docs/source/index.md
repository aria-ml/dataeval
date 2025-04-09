# DataEval

::::{grid}
:reverse:
:gutter: 3 4 4 4
:margin: 1 2 1 2

:::{grid-item}
:columns: 12 4 4 4

:::{image} _static/images/DataEval_Logo.png
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

DataEval empowers professionals across domains with tools designed to
enhance their workflows. Explore capabilities specific to your role:

::::{tab-set}

:::{tab-item} T&E Engineer

Maximize your evaluation processes with tools designed
for accuracy and reliability:

- Robust Testing: Reduce errors with metrics that reliably work with state of
  the art image classification and object detection datasets
- Post-Deployment Monitoring: Keep models on track with
  easy-to-implement logging of {term}`Operational Drift` metrics
- Responsive metrics: Optimize evaluation with tailored guidance for error
  assessment and retraining.

📖 [Learn more about tools for Test & Evaluation Engineers.](./concepts/workflows/TE_engineer_workflow.md)

:::

:::{tab-item} ML Engineer

Accelerate model development with powerful insights for training and deployment:

- Model-Specific Metrics: Evaluate dataset {term}`Sufficiency` and detect
  data/model complexity mismatches.
- Performance Optimization: Establish bounds on real-world model performance
  for improved training strategies.
- Drift Detection: Rapidly diagnose model degradation under
  {term}`Operational Drift` to maintain model accuracy and stability.

📖 [Learn more about tools for Machine Learning Engineers.](./concepts/workflows/ML_engineer_workflow.md)

:::

:::{tab-item} Data Scientist

Drive innovation with data-focused tools that uncover hidden patterns and
complexities:

- Metafeatures: Leverage metrics to analyze data complexity and improve
  data-driven decisions based on metadata features.
- Real-World Insights: Improve dataset sampling and improve
  {term}`Balance`, {term}`Completeness`, and {term}`Coverage`.
- Error Analysis: Gain actionable feedback to refine datasets and improve
  performance.
- Complete Shift Analysis: Quantify impactful changes in data due to
  [Covariate Shift](./concepts/Drift.md#covariate-shift), {term}`Label Shift`,
  and {term}`Concept Drift` before they impact your model.

📖 [Learn more about tools for Data Scientists.](./concepts/workflows/data_scientist_workflow.md)

:::

::::

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
