<!-- markdownlint-disable MD041 -->
:hide-toc:
:hide-navigation:

# DataEval documentation

**Version**: {{ release }} | {{ date_label }}: {sub-ref}`today`

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

DataEval analyzes datasets and models to give users the ability to train and
test performant, unbiased, and reliable AI models and monitor data for
impactful shifts to deployed models.

:::
::::

DataEval is an open-source software designed to help data scientists,
developers, and T&E engineers develop and analyze computer vision datasets
and the resulting impact on models.

::::{grid} 1 1 2 2
:gutter: 2 3 4 4

:::{grid-item-card}
:text-align: center
:link: ./getting-started/quickstart
:link-type: doc
:link-alt: "Beginner's Guide Link"
:img-top: _static/images/icons/icon_quick_start.svg
:class-header: sd-fs-4

**Getting Started**
^^^
New to DataEval? Check out the Beginner's Guide.
It contains an introduction to DataEval's main concepts.

:::

:::{grid-item-card}
:text-align: center
:link: ./tutorials/index
:link-type: doc
:link-alt: "Tutorial Page Link"
:img-top: _static/images/icons/icon_tutorial.svg
:class-header: sd-fs-4

**Tutorials**
^^^
Learn DataEval through guided end-to-end examples.

:::

:::{grid-item-card}
:text-align: center
:link: ./concepts/index
:link-type: doc
:img-top: _static/images/icons/icon_concept.svg
:class-header: sd-fs-4

**Explanations**
^^^
Learn about the key concepts related to using data
in computer-vision AI applications.

:::

:::{grid-item-card}
:text-align: center
:link: ./how-to/index
:link-type: doc
:link-alt: "How To Guides Page Link"
:img-top: _static/images/icons/icon_howto.svg
:class-header: sd-fs-4

**How-to Guides**
^^^
Task-based instructions and workflow recipes.

:::

:::{grid-item-card}
:text-align: center
:link: ./reference/autoapi/dataeval/index
:link-type: doc
:img-top: _static/images/icons/icon_api.svg
:class-header: sd-fs-4

**Reference**
^^^
The reference guide contains a detailed description the DataEval API.
The reference describes how the methods work and which parameters can
be used. It assumes that you have an understanding of the key concepts.

:::

:::{grid-item-card}
:text-align: center
:link: ./getting-started/roles/index
:link-type: doc
:link-alt: "Roles Guide Link"
:img-top: _static/images/icons/icon_roles.svg
:class-header: sd-fs-4

**Roles Guide**
^^^
Not sure how DataEval fits into your daily routine?
Check out our Roles Guide which shows some of the ways
in which we use DataEval.

:::
::::

## Why DataEval?

:::{include} ../../README.md
:start-after: <!-- start needs -->
:end-before: <!-- end needs -->
:::

## Key Features

DataEval empowers professionals across domains with tools designed to
enhance their workflows. Explore capabilities specific to your role:

- Metafeatures: Leverage metrics to analyze data complexity and improve
  data-driven decisions based on metadata features.
- Real-World Insights: Improve dataset sampling and improve
  {term}`Balance`, {term}`Completeness`, and {term}`Coverage`.
- Model-Specific Metrics: Evaluate dataset {term}`Sufficiency` and detect
  data/model complexity mismatches.
- Performance Optimization: Establish bounds on real-world model performance
  for improved training strategies.
- Responsive metrics: Optimize evaluation with tailored guidance for error
  assessment and retraining.
- Robust Testing: Reduce errors with metrics that reliably work with state of
  the art image classification and object detection datasets
- Post-Deployment Monitoring: Keep models on track with
  easy-to-implement logging of {term}`Operational Drift` metrics
- Drift Detection: Rapidly diagnose model degradation under
  {term}`Operational Drift` to maintain model accuracy and stability.
- Complete Shift Analysis: Quantify impactful changes in data due to
  [Covariate Shift](./concepts/DistributionShift.md#taxonomy-of-shift), {term}`Label Shift`,
  and {term}`Concept Drift` before they impact your model.

By incorporating DataEval into their workflows, data scientists, developers,
and T&E engineers can thoroughly analyze and evaluate their datasets to maximize
generalization and performance.
<!--::::{tab-set}

:::{tab-item} T&E Engineer

Maximize your evaluation processes with tools designed
for accuracy and reliability:

- Robust Testing: Reduce errors with metrics that reliably work with state of
  the art image classification and object detection datasets
- Post-Deployment Monitoring: Keep models on track with
  easy-to-implement logging of {term}`Operational Drift` metrics
- Responsive metrics: Optimize evaluation with tailored guidance for error
  assessment and retraining.

📖 [Learn more about tools for Test & Evaluation Engineers.](./getting-started/roles/te_engineer.md)

:::

:::{tab-item} ML Engineer

Accelerate model development with powerful insights for training and deployment:

- Model-Specific Metrics: Evaluate dataset {term}`Sufficiency` and detect
  data/model complexity mismatches.
- Performance Optimization: Establish bounds on real-world model performance
  for improved training strategies.
- Drift Detection: Rapidly diagnose model degradation under
  {term}`Operational Drift` to maintain model accuracy and stability.

📖 [Learn more about tools for Machine Learning Engineers.](./getting-started/roles/ml_engineer.md)

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
  [Covariate Shift](./concepts/DistributionShift.md#taxonomy-of-shift), {term}`Label Shift`,
  and {term}`Concept Drift` before they impact your model.

📖 [Learn more about tools for Data Scientists.](./getting-started/roles/data_scientist.md)

:::

::::-->

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
:::

:::{toctree}
:caption: Getting Started
:hidden:

Beginner's Guide <getting-started/quickstart>
getting-started/installation
getting-started/where-to-go-next
getting-started/which-tool
Roles Guide <getting-started/roles/index>
:::

:::{toctree}
:caption: Tutorials
:hidden:

Overview <tutorials/index>
:::

:::{toctree}
:caption: How-to Guides
:hidden:

Overview <how-to/index>
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
Functional Overview <reference/FunctionalOverview>
reference/glossary
:::

:::{toctree}
:caption: Development
:hidden:

getting-started/contributing
Change Log <getting-started/changelog>
:::
