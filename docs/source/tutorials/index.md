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

../notebooks/tt_clean_dataset.ipynb
../notebooks/tt_assess_data_space.ipynb
../notebooks/tt_identify_bias.ipynb
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [](../notebooks/tt_clean_dataset.ipynb)
  - Learn about the impacts of unstructured, raw data and how to transform it
    into a reliable, robust dataset.
  - [![Open In Colab][colab-badge]][eda-colab]
* - [](../notebooks/tt_assess_data_space.ipynb)
  - Learn how to fix and prevent gaps in data to develop more reliable and
    robust models.
  - [![Open In Colab][colab-badge]][dataspace-colab]
* - [](../notebooks/tt_identify_bias.ipynb)
  - Learn how correlations in your data and metadata can affect model
    performance and what can be done to remove that bias.
  - [![Open In Colab][colab-badge]][bias-colab]
```

<!-- Ref links -->

<!-- markdownlint-disable MD053 -->

[eda-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.3/docs/source/notebooks/tt_clean_dataset.ipynb
[dataspace-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.3/docs/source/notebooks/tt_assess_data_space.ipynb
[bias-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.3/docs/source/notebooks/tt_identify_bias.ipynb

<!-- markdownlint-enable MD053 -->

<!-- END DATA ENGINEERING -->

## **Monitoring**

```{toctree}
:hidden:
:caption: Monitoring

../notebooks/tt_monitor_shift.ipynb
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [](../notebooks/tt_monitor_shift.ipynb)
  - Learn how to analyze incoming data against training data to ensure deployed
    models stay reliable and robust.
  - [![Open In Colab][colab-badge]][monitoring-colab]
```

<!-- ref links -->

<!-- markdownlint-disable MD053 -->

[monitoring-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/v0.86.3/docs/source/tutorials/notebooks/tt_monitor_shift.ipynb

<!-- markdownlint-enable MD053 -->

<!-- END MONITORING -->

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
