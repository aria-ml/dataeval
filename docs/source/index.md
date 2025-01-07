---
sd_hide_title: True
---

# Home

## Welcome to DataEval's Documentation

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

---

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

---

<!-- SECTION START | Quick, beginner friendly guides as eye catchers. Not a
part of Diataxis -->

## Before You Begin

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item-card} [Installation Guide](installation.md){.font-title}
:text-align: center
:link: installation
:link-type: doc
:link-alt: "Installation Guide Link"

Jump in and try out DataEval for yourself.

:::

:::{grid-item-card} [About DataEval](about.md){.font-title}
:text-align: center
:link: about
:link-type: doc
:link-alt: "About DataEval Page Link"

Learn what drives us here at ARiA, our role in AI T&E, and why we created
DataEval.

:::

:::{grid-item-card} [Algorithm Overview](concepts/metric_table.md){.font-title}
:margin: 4 0 auto auto
:text-align: center
:link: concepts/metric_table
:link-type: doc
:link-alt: "DataEval Algorithm Overview Chart"

Quick overview detailing each algorithm's functionality and requirements.

:::
::::

## Get Started

We are proud of our tools, so we highlighted some simple but powerful
functionality that you can try yourself!

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item-card} [Bayes Error Rate](how_to/notebooks/BayesErrorRateEstimationTutorial.ipynb){.font-title}
:text-align: center
:link: how_to/notebooks/BayesErrorRateEstimationTutorial
:link-type: doc
:img-bottom: ./_static/ber_plot_thumbnail.png

Accurately calculate the maximum performance of your dataset.

<!-- We want to show visualizations of tutorials to peak the interest of a
     potential user. Might be good to add a BER graph that a user would need
     (not necessarily from tutorial) i.e. A Graph with training accuracy curve,
     and a BER line (similar to sufficiency) -->

:::

:::{grid-item-card} [Model Sufficiency](how_to/notebooks/ClassLearningCurvesTutorial.ipynb){.font-title}
:text-align: center
:link: how_to/notebooks/ClassLearningCurvesTutorial
:link-type: doc
:img-bottom: ./_static/suff_plot_thumbnail.png

Estimate your model's performance based on the size of your dataset

<!-- We should add a datasets blobs image here with the divergence -->

:::
::::

<!-- SECTION END -->

<!-- SECTION START | "In Action" of Diataxis framework-->

## Stay Practical

These handcrafted guides created by experts here at ARiA will get you up and
running while improving your day-to-day worklife.

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item-card} [Tutorials](tutorials/index.md){.font-title}
:text-align: center
:link: tutorials/index
:link-type: doc
:link-alt: "Tutorial Page Link"

Not sure where to begin?

Try out these guides to learn the ins and outs of AI T&E using DataEval.

:::

:::{grid-item-card} [How-To's](how_to/index.md){.font-title}
:text-align: center
:link: how_to/index
:link-type: doc
:link-alt: "How To Page Link"

Already know what you're looking for?

Check out these curated guides to see how DataEval can improve your workflows.

:::
::::

<!-- SECTION END -->

<!-- SECTION START | "In cognition (theory)" of Diataxis framework -->

<!-- Split acquisition (learning) and application (practice) since multiple
     types of explanation -->

<!-- SUBSECTION START | Explanations -->

## Be Theoretical

Dive deep into the concepts that DataEval is built upon to enhance your skill
set.

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item-card} [Concepts](concepts/index.md){.font-title}
:text-align: center
:link: concepts/index
:link-type: doc
:link-alt: "Concept Page Link"

Need to understand the theory behind the math that makes DataEval so powerful?

Click through these focused guides on the research, implementation, and
tradeoffs we used to better suit your needs.

:::

:::{grid-item-card} [Workflows](workflows/index.md){.font-title}
:text-align: center
:link: workflows/index
:link-type: doc
:link-alt: "Workflows Page Link"

Want in-depth understanding with no-code explanations?

Read these role-specific guides for the data analysis tasks you will see in
your daily work.

:::
::::

<!-- SUBSECTION END -->

<!-- SUBSECTION START | Reference -->
## Get Technical

Everything you need to become an expert with DataEval.

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item-card} [API Reference](reference/index.rst){.font-title}
:text-align: center
:link: reference/index
:link-type: doc
:link-alt: "Reference Page Link"

Looking for a specific function or class?

Find all the technical details needed to understand the DataEval Ecosystem.

:::

:::{grid-item-card} [Glossary](concepts/glossary.md){.font-title}
:text-align: center
:link: concepts/glossary
:link-type: doc
:link-alt: "Glossary Page link"

Looking for a definition?

Find the word in the glossary.

:::
::::
<!-- SUBSECTION END -->

<!-- SECTION END -->

## [Contributing](contributing.md)

DataEval is an open-source software that is open for anyone to request
features, fix bugs, or reach out for help.

Follow our [contributing guide](contributing.md) to get started!

## [Changelog](changelog.md)

DataEval's development changelog.

## Attribution

:::{include} ../../README.md
:start-after: <!-- start attribution -->
:end-before: <!-- end attribution -->
:::

<!-- BELOW IS SIDEBAR TOC TREE -->

:::{toctree}
:hidden:
:maxdepth: 1

self
:::

:::{toctree}
:caption: Guides
:hidden:
:maxdepth: 2

installation
tutorials/index
how_to/index

:::

:::{toctree}
:caption: Reference
:hidden:

reference/index

:::

:::{toctree}
:caption: Explanation
:hidden:
:maxdepth: 2

concepts/index
workflows/index
concepts/glossary
:::

:::{toctree}
:hidden:
:titlesonly:

contributing
changelog
about
:::
