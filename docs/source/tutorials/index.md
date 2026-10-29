# DataEval tutorials

The tutorials on this page aim to teach important concepts one can come across
as a machine learning testing and evaluation engineer through easy-to-follow
examples that are handcrafted to give you the best experience possible.

The tutorials are split into sections based on which stage they fall under in
the [machine learning life cycle](../getting-started/roles/ML_Lifecycle.md).
Currently, there are tutorials for the following stages:

1. [Data engineering](#data-engineering)
2. [Model development](#model-development)
3. [Monitoring](#monitoring)

More tutorials are always in development, but you can suggest specific
tutorials by [requesting a topic](../getting-started/contributing.md).

To view the tutorial directly in the browser, click the title.

To run the tutorial interactively in Google Colab, click the
![Open In Colab][colab-badge] icon.

See [Running notebooks locally](#running-notebooks-locally) at the
bottom of this page for how to generate runnable `.ipynb` files in your
local checkout.

## Data engineering

```{toctree}
:hidden:

../notebooks/tt_clean_dataset.py
../notebooks/tt_assess_data_space.py
../notebooks/tt_identify_dataset_gaps.py
../notebooks/tt_identify_bias.py
../notebooks/tt_analyze_by_custom_class_axis.py
../notebooks/tt_analyze_across_levels.py
../notebooks/tt_augmentation_duplicates.py
../notebooks/tt_identify_duplicate_video.py
../notebooks/tt_compare_objects_to_surroundings.py
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [](../notebooks/tt_clean_dataset.py)
  - Learn how to find extreme and redundant images with {class}`.Outliers` and {class}`.Duplicates`, and how
    to decide whether to remove them or collect more data.
  - [![Open In Colab][colab-badge]][eda-colab]
* - [](../notebooks/tt_assess_data_space.py)
  - Learn how to cluster the embeddings of an unlabeled dataset to find outliers and underrepresented images,
    and how to decide whether to add or remove them.
  - [![Open In Colab][colab-badge]][dataspace-colab]
* - [](../notebooks/tt_identify_dataset_gaps.py)
  - Combine an ontology, labels, and embeddings to find label-space and
    embedding-space gaps in a labeled dataset.
  - [![Open In Colab][colab-badge]][gaps-colab]
* - [](../notebooks/tt_identify_bias.py)
  - Learn how correlations in your data and metadata can affect model
    performance and what can be done to remove that bias.
  - [![Open In Colab][colab-badge]][bias-colab]
* - [](../notebooks/tt_analyze_by_custom_class_axis.py)
  - Learn how to condition bias, diversity and splitting on a factor of your own with
    {meth}`.Metadata.classed_by`, and how to tell whether a class or an environmental
    condition drives an imbalance.
  - [![Open In Colab][colab-badge]][axis-colab]
* - [](../notebooks/tt_analyze_across_levels.py)
  - Learn why a per-image factor read from detection rows is weighted by
    detection count, and how to choose the population your question is about.
  - [![Open In Colab][colab-badge]][levels-colab]
* - [](../notebooks/tt_augmentation_duplicates.py)
  - Learn how common torchvision augmentations like rotations, flips, and color
    jitter can be detected as near duplicates using D4 hashes and BoVW embeddings.
  - [![Open In Colab][colab-badge]][augs-colab]
* - [](../notebooks/tt_identify_duplicate_video.py)
  - Learn how to find duplicate video, train-test leakage and redundant footage in a
    multi-object tracking dataset, and how to read what each detection reports.
  - [![Open In Colab][colab-badge]][vdupe-colab]
* - [](../notebooks/tt_compare_objects_to_surroundings.py)
  - Compare detections against their surroundings and against each other
    to find hard-to-see objects and background shortcuts.
  - [![Open In Colab][colab-badge]][surr-colab]
```

<!-- Ref links -->

<!-- markdownlint-disable MD053 -->

[eda-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_clean_dataset.ipynb
[dataspace-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_assess_data_space.ipynb
[gaps-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_identify_dataset_gaps.ipynb
[bias-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_identify_bias.ipynb
[axis-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_analyze_by_custom_class_axis.ipynb
[levels-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_analyze_across_levels.ipynb
[augs-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_augmentation_duplicates.ipynb
[vdupe-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_identify_duplicate_video.ipynb
[surr-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_compare_objects_to_surroundings.ipynb

<!-- markdownlint-enable MD053 -->

<!-- END DATA ENGINEERING -->

## Model development

```{toctree}
:hidden:
:caption: Model development

../notebooks/tt_assess_ic_feasibility.py
```

```{list-table}
:widths: 20 60 20
:header-rows: 0

* - [](../notebooks/tt_assess_ic_feasibility.py)
  - Learn how the experimental {class}`.Feasibility` evaluator judges whether
    military vehicle requirements are reachable, how to read the evidence
    behind each determination, and how to change a question the evidence
    cannot answer into one it can.
  - [![Open In Colab][colab-badge]][feas-colab]
```

<!-- markdownlint-disable MD053 -->

[feas-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_assess_ic_feasibility.ipynb

<!-- markdownlint-enable MD053 -->

<!-- END MODEL DEVELOPMENT -->

## Monitoring

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
  - Learn how to compare drift detectors on image embeddings, read what each one reports, and monitor drift
    across chunks of incoming data.
  - [![Open In Colab][colab-badge]][monitoring-colab]
* - [](../notebooks/tt_detect_drift_with_uncertainty.py)
  - Learn how to monitor a deployed detector for drift using its own prediction
    uncertainty as a label-free signal.
  - [![Open In Colab][colab-badge]][uncertainty-drift-colab]
* - [](../notebooks/tt_identify_ood_samples.py)
  - Learn how to fit reconstruction-based and nearest-neighbor detectors on in-distribution images and compare
    how many out-of-distribution images each one flags.
  - [![Open In Colab][colab-badge]][iood-colab]
```

<!-- ref links -->

<!-- markdownlint-disable MD053 -->

[monitoring-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_monitor_shift.ipynb
[uncertainty-drift-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_detect_drift_with_uncertainty.ipynb
[iood-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/tt_identify_ood_samples.ipynb

<!-- markdownlint-enable MD053 -->

<!-- END MONITORING -->

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg

## Running notebooks locally

The notebook sources live as `py:percent` scripts in
`docs/source/notebooks/`. To get runnable `.ipynb` files for local
editing, choose one of:

- With nox (recommended), `uv run nox -s docs` builds the
  documentation and writes a runnable `.ipynb` next to each script.
- With jupytext directly, `jupytext --to notebook docs/source/notebooks/<name>.py`
  converts one notebook, and `docs/source/notebooks/*.py` converts all of them.

The generated `.ipynb` files are gitignored, so edits stay local to
your checkout.
