# DataEval Tutorials

The tutorials on this page aim to teach important concepts one can come across
as a machine learning testing and evaluation engineer through easy-to-follow
examples that are handcrafted to give you the best experience possible.

The tutorials are split into sections based on which stage they fall under in
the [machine learning life cycle](../getting-started/roles/ML_Lifecycle.md).
Currently, there are tutorials for the following stages:

1. [Data engineering](#data-engineering)
2. [Monitoring](#monitoring)

More tutorials are always in development, but you can suggest specific
tutorials by [requesting a topic](../getting-started/contributing.md).

To view the tutorial directly in the browser, click the title.

To run the tutorial interactively in Google Colab, click the
![Open In Colab][colab-badge] icon.

See [Running notebooks locally](#running-notebooks-locally) at the
bottom of this page for how to generate runnable `.ipynb` files in your
local checkout.

## **Data engineering**

```{toctree}
:hidden:

../notebooks/tt_clean_dataset.py
../notebooks/tt_assess_data_space.py
../notebooks/tt_identify_dataset_gaps.py
../notebooks/tt_identify_bias.py
../notebooks/tt_augmentation_duplicates.py
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [](../notebooks/tt_clean_dataset.py)
  - Learn about the impacts of unstructured, raw data and how to transform it
    into a reliable, robust dataset.
  - [![Open In Colab][colab-badge]][eda-colab]
* - [](../notebooks/tt_assess_data_space.py)
  - Learn how to fix and prevent gaps in data to develop more reliable and
    robust models.
  - [![Open In Colab][colab-badge]][dataspace-colab]
* - [](../notebooks/tt_identify_dataset_gaps.py)
  - Combine an ontology, labels, and embeddings to find label-space and
    embedding-space gaps in a labeled dataset.
  - [![Open In Colab][colab-badge]][gaps-colab]
* - [](../notebooks/tt_identify_bias.py)
  - Learn how correlations in your data and metadata can affect model
    performance and what can be done to remove that bias.
  - [![Open In Colab][colab-badge]][bias-colab]
* - [](../notebooks/tt_augmentation_duplicates.py)
  - Learn how common torchvision augmentations like rotations, flips, and color
    jitter can be detected as near duplicates using D4 hashes and BoVW embeddings.
  - [![Open In Colab][colab-badge]][augs-colab]
```

<!-- Ref links -->

<!-- markdownlint-disable MD053 -->

[eda-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/tt_clean_dataset.ipynb
[dataspace-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/tt_assess_data_space.ipynb
[gaps-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/tt_identify_dataset_gaps.ipynb
[bias-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/tt_identify_bias.ipynb
[augs-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/tt_augmentation_duplicates.ipynb

<!-- markdownlint-enable MD053 -->

<!-- END DATA ENGINEERING -->

## **Monitoring**

```{toctree}
:hidden:
:caption: Monitoring

../notebooks/tt_monitor_shift.py
../notebooks/tt_detect_drift_with_uncertainty.py
../notebooks/tt_identify_ood_samples.py
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [](../notebooks/tt_monitor_shift.py)
  - Learn how to analyze incoming data against training data to ensure deployed
    models stay reliable and robust.
  - [![Open In Colab][colab-badge]][monitoring-colab]
* - [](../notebooks/tt_detect_drift_with_uncertainty.py)
  - Learn how to monitor a deployed detector for drift using its own prediction
    uncertainty as a label-free signal.
  - [![Open In Colab][colab-badge]][uncertainty-drift-colab]
* - [](../notebooks/tt_identify_ood_samples.py)
  - Identify out-of-distribution samples in incoming data
  - [![Open In Colab][colab-badge]][iood-colab]
```

<!-- ref links -->

<!-- markdownlint-disable MD053 -->

[monitoring-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/tt_monitor_shift.ipynb
[uncertainty-drift-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/tt_detect_drift_with_uncertainty.ipynb
[iood-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.0.0/notebooks/tt_identify_ood_samples.ipynb

<!-- markdownlint-enable MD053 -->

<!-- END MONITORING -->

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg

## Running notebooks locally

The notebook sources live as `py:percent` scripts in
`docs/source/notebooks/`. To get runnable `.ipynb` files for local
editing, choose one of:

- **With nox (recommended):** `nox -s docsync` — bidirectional sync of
  the `.py`/`.ipynb` pairs.
- **With jupytext directly:** `jupytext --to notebook docs/source/notebooks/*.py`

The generated `.ipynb` files are gitignored, so edits stay local to
your checkout.
