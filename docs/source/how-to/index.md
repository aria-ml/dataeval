# How-to guides

These guides help you accomplish specific tasks with DataEval. Each one addresses
a practical problem and walks you through the solution step by step.

In addition to viewing them in the documentation, you can open these notebooks
in Google Colab and use them interactively.

The guides are organized by where they fall in the
[machine learning life cycle](../getting-started/roles/ML_Lifecycle.md):

1. [Configuration](#configuration)
2. [Data engineering](#data-engineering)
3. [Model development](#model-development)
4. [Monitoring](#monitoring)

See [Running notebooks locally](#running-notebooks-locally) at the bottom
of this page for how to generate runnable `.ipynb` files in your local
checkout.

## Configuration

These guides will provide quick examples of how to configure DataEval for your
environment.

```{toctree}
:hidden:

../notebooks/h2_configure_defaults.py
../notebooks/h2_configure_logging.py
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0

- - [](../notebooks/h2_configure_defaults.py)
  - Configure global DataEval defaults
  - [![Open In Colab][colab-badge]][hdw-colab]
- - [](../notebooks/h2_configure_logging.py)
  - Configure logging with DataEval
  - [![Open In Colab][colab-badge]][log-colab]

:::

[hdw-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_configure_defaults.ipynb
[log-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_configure_logging.ipynb

## Data engineering

These guides cover tasks related to preparing, cleaning, exploring, and curating
datasets for machine learning.

```{toctree}
:hidden:

../notebooks/h2_wrap_dataframe_dataset.py
../notebooks/h2_wrap_video_files.py
../notebooks/h2_metadata_from_dataframe.py
../notebooks/h2_build_dataset_views.py
../notebooks/h2_create_embeddings.py
../notebooks/h2_embed_detection_crops.py
../notebooks/h2_place_transforms.py
../notebooks/h2_cluster_analysis.py
../notebooks/h2_deduplicate.py
../notebooks/h2_find_factor_outliers.py
../notebooks/h2_find_conditional_outliers.py
../notebooks/h2_trace_findings_to_source.py
../notebooks/h2_detect_undersampling.py
../notebooks/h2_add_intrinsic_factors.py
../notebooks/h2_custom_image_stats_object_detection.py
../notebooks/h2_measure_channel_groups.py
../notebooks/h2_control_factor_binning.py
../notebooks/h2_sync_and_cache_metadata.py
../notebooks/h2_reconcile_labels_ontology.py
../notebooks/h2_align_label_spaces.py
../notebooks/h2_conform_and_merge_datasets.py
```

### Data preparation

:::{list-table}
:widths: 30 50 20
:header-rows: 0

- - [](../notebooks/h2_wrap_dataframe_dataset.py)
  - Wrap a pandas DataFrame catalog of images or bounding boxes as a
    DataEval dataset, and defer image decoding to speed up
    metadata-only analysis
  - [![Open In Colab][colab-badge]][wdf-colab]
- - [](../notebooks/h2_wrap_video_files.py)
  - Wrap video files and their tracking annotations as a multi-object
    tracking dataset, and serve each file's frame rate and duration as
    named factors
  - [![Open In Colab][colab-badge]][wvf-colab]
- - [](../notebooks/h2_metadata_from_dataframe.py)
  - Build a MetadataLike object from a DataFrame for bias
    analysis without loading images
  - [![Open In Colab][colab-badge]][mdf-colab]
- - [](../notebooks/h2_build_dataset_views.py)
  - Filter, balance, reorder, and relabel a dataset by composing
    a {class}`.View` from {class}`.Operation` steps
  - [![Open In Colab][colab-badge]][view-colab]
- - [](../notebooks/h2_create_embeddings.py)
  - Create image embeddings with PyTorch, ONNX or a model-free extractor
  - [![Open In Colab][colab-badge]][onx-colab]
- - [](../notebooks/h2_embed_detection_crops.py)
  - Embed object detection box crops and visualize class clusters
  - [![Open In Colab][colab-badge]][crop-colab]
- - [](../notebooks/h2_place_transforms.py)
  - Respond to a {class}`~dataeval.exceptions.StatsInvalidatedWarning` by
    moving the transform to the extractor, narrowing `flags=` or asserting
    what the operation invalidates
  - [![Open In Colab][colab-badge]][xfrm-colab]

:::

### Data quality

:::{list-table}
:widths: 30 50 20
:header-rows: 0

- - [](../notebooks/h2_cluster_analysis.py)
  - Cluster a dataset and find outliers and near duplicates in the clusters
  - [![Open In Colab][colab-badge]][clst-colab]
- - [](../notebooks/h2_deduplicate.py)
  - Find exact and near-duplicate images, and object detection items that share or disagree on an
    annotation
  - [![Open In Colab][colab-badge]][dupe-colab]
- - [](../notebooks/h2_find_factor_outliers.py)
  - Threshold metadata factors, each against its own level, and trace a
    flagged track back to the metadata
  - [![Open In Colab][colab-badge]][fout-colab]
- - [](../notebooks/h2_find_conditional_outliers.py)
  - Judge metadata factors within strata, by rarity, given another factor,
    and in combination
  - [![Open In Colab][colab-badge]][cout-colab]
- - [](../notebooks/h2_trace_findings_to_source.py)
  - Retrieve the image, frame, track or detection a finding's
    {class}`.SourceIndex` names
  - [![Open In Colab][colab-badge]][trace-colab]
- - [](../notebooks/h2_detect_undersampling.py)
  - Detect undersampled subsets of datasets
  - [![Open In Colab][colab-badge]][cov-colab]

:::

### Metadata and bias

:::{list-table}
:widths: 30 50 20
:header-rows: 0

- - [](../notebooks/h2_add_intrinsic_factors.py)
  - Measure intrinsic factors such as brightness and sharpness with {func}`.image_stats` and
    add them to {class}`.Metadata` for bias analysis
  - [![Open In Colab][colab-badge]][imd-colab]
- - [](../notebooks/h2_custom_image_stats_object_detection.py)
  - Measure image statistics over the whole image, the bounding boxes and the background of an
    object detection dataset
  - [![Open In Colab][colab-badge]][calc-colab]
- - [](../notebooks/h2_measure_channel_groups.py)
  - Measure named groups of bands separately on multispectral imagery
  - [![Open In Colab][colab-badge]][band-colab]
- - [](../notebooks/h2_control_factor_binning.py)
  - Declare, review and reuse a factor's bin edges so results
    are comparable across collections
  - [![Open In Colab][colab-badge]][ctrl-colab]
- - [](../notebooks/h2_sync_and_cache_metadata.py)
  - Realign filtered metadata with its dataset, and cache the metadata
    so it is built once
  - [![Open In Colab][colab-badge]][sync-colab]

:::

### Ontology

:::{list-table}
:widths: 30 50 20
:header-rows: 0

- - [](../notebooks/h2_reconcile_labels_ontology.py)
  - Validate dataset class names against an ontology and
    recover their hierarchy
  - [![Open In Colab][colab-badge]][onto-colab]
- - [](../notebooks/h2_align_label_spaces.py)
  - Align two label vocabularies into typed correspondences
    and a carry-over class remapping
  - [![Open In Colab][colab-badge]][algn-colab]
- - [](../notebooks/h2_conform_and_merge_datasets.py)
  - Conform an incoming dataset to a reference vocabulary
    and merge the two
  - [![Open In Colab][colab-badge]][cmrg-colab]

:::

[onx-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_create_embeddings.ipynb
[crop-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_embed_detection_crops.ipynb
[clst-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_cluster_analysis.ipynb
[dupe-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_deduplicate.ipynb
[fout-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_find_factor_outliers.ipynb
[cout-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_find_conditional_outliers.ipynb
[trace-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_trace_findings_to_source.ipynb
[calc-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_custom_image_stats_object_detection.ipynb
[band-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_measure_channel_groups.ipynb
[imd-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_add_intrinsic_factors.ipynb
[ctrl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_control_factor_binning.ipynb
[sync-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_sync_and_cache_metadata.ipynb
[cov-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_detect_undersampling.ipynb
[wdf-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_wrap_dataframe_dataset.ipynb
[mdf-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_metadata_from_dataframe.ipynb
[view-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_build_dataset_views.ipynb
[xfrm-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_place_transforms.ipynb
[onto-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_reconcile_labels_ontology.ipynb
[algn-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_align_label_spaces.ipynb
[cmrg-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_conform_and_merge_datasets.ipynb
[wvf-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_wrap_video_files.ipynb

## Model development

These guides cover tasks related to assessing data feasibility and sufficiency
for model training.

```{toctree}
:hidden:

../notebooks/h2_measure_ic_feasibility.py
../notebooks/h2_measure_ic_sufficiency.py
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0

- - [](../notebooks/h2_measure_ic_feasibility.py)
  - Estimate the {term}`Bayes Error Rate (BER)` with {func}`.ber_mst` and check a classification
    requirement against it
  - [![Open In Colab][colab-badge]][ber-colab]
- - [](../notebooks/h2_measure_ic_sufficiency.py)
  - Determine the amount of data needed to meet
    image classification performance requirements
  - [![Open In Colab][colab-badge]][suff-colab]

:::

[ber-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_measure_ic_feasibility.ipynb
[suff-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_measure_ic_sufficiency.ipynb

## Monitoring

These guides cover tasks related to comparing datasets and detecting distribution
shifts in deployed systems.

```{toctree}
:hidden:

../notebooks/h2_measure_divergence.py
../notebooks/h2_measure_label_independence.py
../notebooks/h2_detect_metadata_drift.py
../notebooks/h2_detect_uncertainty_drift.py
../notebooks/h2_detect_drift_from_model_outputs.py
```

:::{list-table}
:widths: 30 50 20
:header-rows: 0

- - [](../notebooks/h2_measure_divergence.py)
  - Measure the divergence between two datasets, such as a training set and a test set
  - [![Open In Colab][colab-badge]][div-colab]
- - [](../notebooks/h2_measure_label_independence.py)
  - Compare label distributions between 2 datasets
  - [![Open In Colab][colab-badge]][lbl-colab]
- - [](../notebooks/h2_detect_metadata_drift.py)
  - Detect drift in named metadata factors with {class}`.Metadata`
    as a feature extractor, and trace a p-value back to its source
  - [![Open In Colab][colab-badge]][mdd-colab]
- - [](../notebooks/h2_detect_uncertainty_drift.py)
  - Detect distribution shift from a MAITE model's prediction
    uncertainty, with no custom decoding
  - [![Open In Colab][colab-badge]][unc-colab]
- - [](../notebooks/h2_detect_drift_from_model_outputs.py)
  - Detect distribution shift using only precomputed model outputs
  - [![Open In Colab][colab-badge]][mdo-colab]

:::

[div-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_measure_divergence.ipynb
[lbl-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_measure_label_independence.ipynb
[mdd-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_detect_metadata_drift.ipynb
[unc-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_detect_uncertainty_drift.ipynb
[mdo-colab]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v1.2.0-rc0/notebooks/h2_detect_drift_from_model_outputs.ipynb

<!-- Google collab badge icon for all collab links -->

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
