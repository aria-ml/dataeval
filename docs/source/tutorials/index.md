# DataEval Tutorials

The tutorials on this page aim to teach important concepts one can come across
as a machine learning testing and evaluation engineer through easy-to-follow
examples that are handcrafted to give you the best experience possible.

The tutorials are split into sections based on which stage they fall under in
the [machine learning life cycle](../concepts/workflows/ML_Lifecycle.md).
Currently, there are tutorials for the following stages:

1. [Data engineering](#data-engineering)
2. [Monitoring](#monitoring)

More tutorials are always in development, but you can suggest specific
tutorials by [requesting a topic](../home/contributing.md).

To view the tutorial directly in the browser, click the title.

To run the tutorial interactively in Google Colab, click the
![Open In Colab][colab-badge] icon.

## **Data engineering**

```{toctree}
:hidden:

EDA_Part1.ipynb
EDA_Part2.ipynb
EDA_Part3.ipynb
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [Data Cleaning](EDA_Part1.ipynb)
  - Learn about the impacts of unstructured, raw data and how to transform it
    into a reliable, robust dataset.
  - [![Open In Colab][colab-badge]][eda-colab]
* - [Assess an unlabeled data space](./EDA_Part2.ipynb)
  - Learn how to fix and prevent gaps in data to develop more reliable and
    robust models.
  - [![Open In Colab][colab-badge]][dataspace-colab]
* - [Identify Bias and Correlations](EDA_Part3.ipynb)
  - Learn how correlations in your data and metadata can affect model
    performance and what can be done to remove that bias.
  - [![Open In Colab][colab-badge]][bias-colab]
```

<!-- Ref links -->

<!-- markdownlint-disable MD053 -->
[eda-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.0/docs/source/tutorials/EDA_Part1.ipynb
[dataspace-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.0/docs/source/tutorials/EDA_Part2.ipynb
[bias-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.0/docs/source/tutorials/EDA_Part3.ipynb
<!-- markdownlint-enable MD053 -->

<!-- END DATA ENGINEERING -->

## **Monitoring**

```{toctree}
:hidden:
:caption: Monitoring

Data_Monitoring.ipynb
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [Data monitoring](Data_Monitoring.ipynb)
  - Learn how to analyze incoming data against training data to ensure deployed
    models stay reliable and robust.
  - [![Open In Colab][colab-badge]][monitoring-colab]
```

<!-- ref links -->

<!-- markdownlint-disable MD053 -->
[monitoring-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.77.0/docs/source/tutorials/Data_Monitoring.ipynb
<!-- markdownlint-enable MD053 -->

<!-- END MONITORING -->

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
