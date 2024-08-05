---
sd_hide_title: true
---

# Overview

::::{grid}
:reverse:
:gutter: 3 4 4 4
:margin: 1 2 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} _static/DataEval_Logo.png
:width: 200px
:class: sd-m-auto
```

:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-fs-5

```{rubric} **Welcome to DataEval's Documentation**
```

DataEval is an open-source toolkit that focuses on characterizing image data and its impact on model performance across classification and object-detection tasks.

:::

::::

----------------

Guides
-------

### [Installation](installation)

If DataEval is not installed, follow this easy [step-by-step guide](installation.md)

### [Quickstart Tutorials](tutorials/index)

We are proud of our tools, so we highlighted common workflows with links so you can try them yourself!

<!-- :doc:`Bayes Error Rate Tutorial<tutorials/notebooks/BayesErrorRateEstimationTutorial>`

We want to show visualizations of tutorials to peak the interest of a potential user
   Might be good to add a BER graph that a user would need (not necessarily from tutorial)
   i.e. A Graph with training accuracy curve, and a BER line (similar to sufficiency) -->

<!--  :doc:`Out-of-Distribution (OOD) Detection Tutorial<tutorials/notebooks/OODDetectionTutorial>`

We want to show visualizations of tutorials to peak the interest of a potential user
   We could show 3 images from a training set class next to 1 that is out-of-dist but classified the same
   Could even make a few rows (multiple classes). -->

DataEval is a powerful toolkit for any data analysis workflow, so be sure to check out the
**Quickstart Tutorials** page for a more comprehensive list of all the tools we offer.

### [How-To's](how_to/index)

For the more experienced user, or if you are just curious, these guides show different ways
that DataEval's features can be used that might fit operational use more closely

Development
------------

### [Reference](reference/index)

Looking for a specific function or class?
This reference guide has all the technical details needed to understand the DataEval Ecosystem

### [Changelog](reference/changelog)
    
DataEval's development changelog

### [About](reference/about)

For more information about why DataEval exists, this page gives an overview to DataEval's purpose in ML

:::{toctree}
:hidden:
:maxdepth: 1

self
:::

:::{toctree}
:caption: Guides
:hidden:
:maxdepth: 2

installation.md
tutorials/index.md
how_to/index
:::

:::{toctree}
:caption: Concepts
:hidden:
:maxdepth: 2

concepts/index.md
:::

:::{toctree}
:caption: Reference
:hidden:
:maxdepth: 2

reference/index.md
:::

:::{toctree}
:hidden:
:titlesonly:
:maxdepth: 1

reference/changelog
reference/about
:::