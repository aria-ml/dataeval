# DataEval Tutorials

The tutorials on this page aim to teach important concepts one can come across
as a machine learning testing and evaluation engineer through easy-to-follow
examples that are handcrafted to give you the best experience possible.

The tutorials are split into sections based on which stage they fall under in
the [machine learning life cycle](../concepts/users/ML_Lifecycle.md).
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

../notebooks/tt_clean_dataset.md
../notebooks/tt_assess_data_space.md
../notebooks/tt_identify_bias.md
../notebooks/tt_augmentation_duplicates.md
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [](../notebooks/tt_clean_dataset.md)
  - Learn about the impacts of unstructured, raw data and how to transform it
    into a reliable, robust dataset.
  - [![Open In Colab][colab-badge]][eda-colab]
* - [](../notebooks/tt_assess_data_space.md)
  - Learn how to fix and prevent gaps in data to develop more reliable and
    robust models.
  - [![Open In Colab][colab-badge]][dataspace-colab]
* - [](../notebooks/tt_identify_bias.md)
  - Learn how correlations in your data and metadata can affect model
    performance and what can be done to remove that bias.
  - [![Open In Colab][colab-badge]][bias-colab]
* - [](../notebooks/tt_augmentation_duplicates.md)
  - Learn how common torchvision augmentations like rotations, flips, and color
    jitter can be detected as near duplicates using D4 hashes and BoVW embeddings.
  - [![Open In Colab][colab-badge]][augs-colab]
```

<!-- Ref links -->

<!-- markdownlint-disable MD053 -->

[eda-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0-rc1/notebooks/tt_clean_dataset.ipynb
[dataspace-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0-rc1/notebooks/tt_assess_data_space.ipynb
[bias-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0-rc1/notebooks/tt_identify_bias.ipynb
[augs-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0-rc1/notebooks/tt_augmentation_duplicates.ipynb

<!-- markdownlint-enable MD053 -->

<!-- END DATA ENGINEERING -->

## **Monitoring**

```{toctree}
:hidden:
:caption: Monitoring

../notebooks/tt_monitor_shift.md
../notebooks/tt_error_analysis.md
../notebooks/tt_identify_ood_samples.md
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [](../notebooks/tt_monitor_shift.md)
  - Learn how to analyze incoming data against training data to ensure deployed
    models stay reliable and robust.
  - [![Open In Colab][colab-badge]][monitoring-colab]
* - [](../notebooks/tt_identify_ood_samples.md)
  - Identify out-of-distribution samples in incoming data
  - [![Open In Colab][colab-badge]][iood-colab]
```

<!-- ref links -->

<!-- markdownlint-disable MD053 -->

[monitoring-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0-rc1/notebooks/tt_monitor_shift.ipynb
[iood-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0-rc1/notebooks/tt_identify_ood_samples.ipynb

<!-- markdownlint-enable MD053 -->

<!-- END MONITORING -->

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
