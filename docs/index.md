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

```{rubric} **Welcome to DataEval's Documentation!**
```
Where experts come to build _robust_, _reliable_, and _responsible_ models.

:::

::::

----------------



## Key Features

- Novel algorithms that characterize image data and its impact on model performance
- Works for image classification and object detection tasks
- Full integration with the JATIC Toolbox by CDAO
- Open source

-----

### **_We do the all the research, you get all the benefits_**

<!-- SECTION START | Quick, beginner friendly guides as eye catchers. Not a part of Diataxis -->

## Before You Begin
::::{grid}

:::{grid-item-card}
:columns: 12 6 6 6
:width: 100%
:text-align: center
:link: installation
:link-type: doc
:link-alt: "Installation Guide Link"

## [Installation Guide](installation)

Jump in and try out DataEval for yourself.

:::

:::{grid-item-card}
:columns: 12 6 6 6
:width: 100%
:text-align: center
:link: reference/about
:link-type: doc
:link-alt: "About DataEval Page Link"

## [About DataEval](reference/about.md)

Learn what drives us here at ARiA, our role in AI T&E, and why we created DataEval.

:::

::::


## Start Here

We are proud of our tools, so we highlighted some simple but powerful functionality that you can try yourself!

::::{grid}

:::{grid-item-card}
:text-align: left
:link: how_to/notebooks/BayesErrorRateEstimationTutorial
:link-type: doc
:img-bottom: _static/ber_plot_thumbnail.png

### [Bayes Error Rate](#bayes-error-rate)

Accurately calculate the maximum performance of your dataset.

<!-- We want to show visualizations of tutorials to peak the interest of a potential user
   Might be good to add a BER graph that a user would need (not necessarily from tutorial)
   i.e. A Graph with training accuracy curve, and a BER line (similar to sufficiency) -->

:::

:::{grid-item-card}
:text-align: left
:link: how_to/notebooks/ClassLearningCurvesTutorial
:link-type: doc
:img-bottom: _static/suff_plot_thumbnail.png

### [Model Sufficiency](#model-sufficiency)

Estimate your model's performance based on the size of your dataset

<!-- We should add a datasets blobs image here with the divergence -->

:::

::::

<!-- SECTION END -->

<!-- SECTION START | "In Action" of Diataxis framework-->

## Stay Practical

These handcrafted guides created by experts here at ARIA will get you up and running while improving your day-to-day worklife.

::::{grid}

:::{grid-item-card}
:columns: 12 6 6 6
:width: 100%
:text-align: center
:link: tutorials/index
:link-type: doc
:link-alt: "Tutorial Page Link"

## [Tutorials](tutorials/index)

Not sure where to begin?

Try out these guides to learn the ins and outs of AI T&E using DataEval.

:::

:::{grid-item-card}
:columns: 12 6 6 6
:width: 100%
:text-align: center
:link: how_to/index
:link-type: doc
:link-alt: "How To Page Link"

## [How-To's](how_to/index)

Already know what you're looking for?

Check out these curated guides to see how DataEval can improve your workflows.

::::

<!-- SECTION END -->

<!-- SECTION START | "In cognition (theory)" of Diataxis framework -->

<!-- Split acquisition (learning) and application (practice) since multiple types of explanation -->
<!-- SUBSECTION START | Explanations -->
## Be Theoretical

Dive deep into the concepts that DataEval is built upon to enhance your skillset.

::::{grid}

:::{grid-item-card}
:columns: 12 6 6 6
:width: 100%
:text-align: center
:link: concepts/index
:link-type: doc
:link-alt: "Concept Page Link"

## [Concepts](concepts/index.md)

Need to understand the theory behind the math that makes DataEval so powerful?

Click through these focused guides on the research, implementation, and tradeoffs we used to better suit your needs.

:::

:::{grid-item-card}
:columns: 12 6 6 6
:width: 100%
:text-align: center
:link: workflows/index
:link-type: doc
:link-alt: "Workflows Page Link"

## [Workflows](workflows/index.md)

Want in-depth understanding with no-code explanations?

Read these role-specific guides for the data analysis tasks you will see in your daily work.

:::

::::

<!-- SUBSECTION END -->

<!-- SUBSECTION START | Reference -->
## Get Technical

Everything you need to become an expert with DataEval.

::::{grid}

:::{grid-item-card}
:columns: 12 6 6 6
:width: 100%
:text-align: center
:link: reference/index
:link-type: doc
:link-alt: "Reference Page Link"

## [API Reference](reference/index)

Looking for a specific function or class?

Find all the technical details needed to understand the DataEval Ecosystem.

:::

:::{grid-item-card}
:columns: 12 6 6 6
:width: 100%
:text-align: center
:link: concepts/glossary
:link-type: doc
:link-alt: "Glossary Page link"

## [Glossary](concepts/glossary.md)

Looking for a definition?

Find the word in the glosssary.

::::
<!-- SUBSECTION END -->

<!-- SECTION END -->

## Contributing

### [Changelog](reference/changelog)
    
DataEval's development changelog


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

installation.md
tutorials/index.md
how_to/index

:::

:::{toctree}
:caption: Explanation
:hidden:
:maxdepth: 2

concepts/index.md
workflows/index.md
concepts/glossary.md
:::

:::{toctree}
:caption: Reference
:hidden:
:maxdepth: 1

reference/index.md
:::

:::{toctree}
:hidden:
:titlesonly:

reference/changelog
reference/about
:::