# What data does each tool need?

Every DataEval tool answers a different question, and each needs a different
combination of inputs to answer it. This page is the index: find the tool you
want, read across its row, and you know what to assemble before you can run it.

If you do not yet know *which* tool you need, start with
[Which DataEval tool should I use?](which-tool.md). For which computer-vision
tasks each algorithm supports, see the
[Functional Overview](../reference/FunctionalOverview.md).

:::{note}
DataEval imposes no restrictions on image type. It accepts any image modality
(RGB, IR, EO, multispectral, greyscale, and others) at any bit depth (8-bit,
16-bit, 32-bit, etc.) and channel count (1+).
:::

## The five input shapes

Nearly every evaluator reduces to one of five shapes. Pick the shape you can
supply and the table below narrows to a handful of candidates.

| Input shape | What you assemble | Evaluators |
| ----------- | ----------------- | ---------- |
| **Dataset only** — raw images | A `Dataset` of images; nothing else | {class}`.Duplicates`, {class}`.Outliers` |
| **Dataset + extractor** — embeddings | A dataset plus a {class}`.FeatureExtractor`, or pre-computed {class}`.Embeddings` | {class}`.Coverage`, {class}`.Prioritize`, every `Drift*` and `OOD*` detector |
| **Dataset + binning** — metadata | A {class}`.Metadata` (or {class}`.MetadataLike`) whose factors are already binned | {class}`.Balance`, {class}`.Diversity`, {class}`.Parity` |
| **Labels + ontology** | Class-label counts plus an {class}`.Ontology` | {class}`.Representation` |
| **Model + training loop** | A model, train and test datasets, and training/evaluation strategies | {class}`.Sufficiency` |

An extractor is not the same thing as a model. {class}`.FlattenExtractor` and
{class}`.BoVWExtractor` need no model at all, while {class}`.TorchExtractor`,
{class}`.OnnxExtractor` and
{class}`.ScoresExtractor` wrap one. Only
{class}`.DriftReconstruction` and {class}`.OODReconstruction` take a model
directly, as a constructor argument rather than through the extractor slot.

## Evaluators

**Primary input** is the data the evaluator actually consumes. The remaining
columns are what else has to be present: ***Required***, *Optional*, or blank
for not used.

| Module | Evaluator | Primary input | Labels | Metadata | Extractor | Model | Reference data |
| ------ | --------- | ------------- | ------ | -------- | --------- | ----- | -------------- |
| `bias` | {class}`.Balance` | Metadata factors{sup}`1` | ***Required***{sup}`2` | ***Required*** | | | |
| `bias` | {class}`.Diversity` | Metadata factors{sup}`1` | ***Required***{sup}`2` | ***Required*** | | | |
| `bias` | {class}`.Parity` | Metadata factors{sup}`1` | ***Required***{sup}`2` | ***Required*** | | | |
| `quality` | {class}`.Duplicates` | Raw images{sup}`3` | | | *Optional* (cluster mode) | | |
| `quality` | {class}`.Outliers` | Raw images{sup}`3` | | *Optional* (`per_class`) | *Optional* (cluster mode) | | |
| `scope` | {class}`.Coverage` | Embeddings | ***Required*** | | ***Required***{sup}`4` | | |
| `scope` | {class}`.Prioritize` | Embeddings | *Optional* (`class_balanced`) | | ***Required***{sup}`4` | | |
| `scope` | {class}`.Representation` | Class labels + {class}`.Ontology` | ***Required*** | | | | |
| `shift` | {class}`.DriftUnivariate` | Arrays or embeddings{sup}`5` | | | *Optional* | | ***Required***{sup}`6` |
| `shift` | {class}`.DriftMMD` | Arrays or embeddings{sup}`5` | | | *Optional* | | ***Required***{sup}`6` |
| `shift` | {class}`.DriftKNeighbors` | Arrays or embeddings{sup}`5` | | | *Optional* | | ***Required***{sup}`6` |
| `shift` | {class}`.DriftDomainClassifier` | Arrays or embeddings{sup}`5` | | | *Optional* | | ***Required***{sup}`6` |
| `shift` | {class}`.DriftWasserstein` | Arrays or embeddings{sup}`5` | | | *Optional* | | ***Required***{sup}`6` |
| `shift` | {class}`.DriftReconstruction` | Arrays or embeddings{sup}`5` | | | *Optional* | ***Required*** (torch) | ***Required***{sup}`6` |
| `shift` | {class}`.OODKNeighbors` | Arrays or embeddings{sup}`5` | | | *Optional* | | ***Required***{sup}`6` |
| `shift` | {class}`.OODDomainClassifier` | Arrays or embeddings{sup}`5` | | | *Optional* | | ***Required***{sup}`6` |
| `shift` | {class}`.OODReconstruction` | Arrays or embeddings{sup}`5` | | | *Optional* | ***Required*** (torch) | ***Required***{sup}`6` |
| `shift` | {class}`.ChunkedDrift` | Wraps another detector{sup}`7` | | | | | ***Required***{sup}`6` |
| `performance` | {class}`.Sufficiency` | Train + test datasets{sup}`3` | ***Required*** | | | ***Required***{sup}`8` | |

```{note}
{sup}`1` Factors must already be binned. {class}`.Metadata` bins continuous
factors for you — automatically via `auto_bin_method`, or explicitly via
`continuous_factor_bins`. See the [Binning concept page](../concepts/Binning.md).  
{sup}`2` Required unless `label=` names one or more factors to condition on in
place of the class labels.  
{sup}`3` Input data must be wrapped together in a `Dataset`.  
{sup}`4` Required unless pre-computed embeddings are supplied instead — as
`Coverage.evaluate(dataset, embeddings=...)`, or as an {class}`.Embeddings` or
array passed to `Prioritize.evaluate()`.  
{sup}`5` Without an extractor the input must be array-like or an
{class}`.Embeddings`; a MAITE dataset is rejected. With an extractor
configured, anything the extractor accepts works, including a full dataset or
raw images. Giving [embeddings](../concepts/Embeddings.md) rather than raw
pixels is strongly recommended either way.  
{sup}`6` Detectors are fit on reference data with `fit()` before `predict()`
scores new data, so two datasets are needed.  
{sup}`7` {class}`.ChunkedDrift` wraps another drift detector; its input
requirements are whatever the wrapped detector needs.  
{sup}`8` {class}`.Sufficiency` retrains the model repeatedly, so it also needs a
{class}`.TrainingStrategy` and an {class}`.EvaluationStrategy`. Bounding boxes
are required for object-detection tasks.
```

Two shortcuts are worth knowing. {class}`.Duplicates` and {class}`.Outliers`
can skip images entirely — `from_stats()` takes a pre-computed `StatsResult`
from {func}`.compute_stats` in their place. Both also accept `per_target=True`
to score individual detections rather than whole images.

## Core functions

The stateless functions in {mod}`dataeval.core` take arrays and labels
directly. This table covers the primary analysis entry points; see the
[API reference](../reference/autoapi/dataeval/core/index.rst) for the full set.

| Function | Images | Labels | Bounding boxes | Metadata | Scores |
| -------- | ------ | ------ | -------------- | -------- | ------ |
| {func}`.ber_knn` / {func}`.ber_mst` | ***Required***{sup}`a` | ***Required*** | | | |
| {func}`.completeness` | ***Required***{sup}`a` | | | | |
| {func}`.compute_ratios` | ***Required***{sup}`b` | | ***Required*** | | |
| {func}`.compute_stats` | ***Required***{sup}`b` | | | | |
| {func}`.coverage_adaptive` / {func}`.coverage_naive` | ***Required***{sup}`a` | | | | |
| {func}`.divergence_fnn` / {func}`.divergence_mst` | ***Required***{sup}`a` | | | | |
| {func}`.factor_deviation` | | | | ***Required***{sup}`c` | ***Required***{sup}`e` |
| {func}`.factor_predictors` | | | | ***Required***{sup}`c` | ***Required***{sup}`e` |
| {func}`.feature_distance` | ***Required***{sup}`a` | | | | |
| {func}`.label_errors` | ***Required***{sup}`a` | ***Required*** | | | |
| {func}`.label_parity` | | ***Required*** | | | |
| {func}`.label_stats` | | ***Required*** | | | |
| {func}`.nullmodel_metrics` | | ***Required*** | | | |
| {func}`.parity` | | ***Required*** | | ***Required*** | |
| {func}`.uap` | | ***Required*** | | | ***Required***{sup}`d` |

## Data selection

| Tool | Images | Labels | Bounding boxes | Metadata | Scores |
| ---- | ------ | ------ | -------------- | -------- | ------ |
| {func}`.split_dataset`{sup}`b` | | *Optional* | | *Optional*{sup}`c` | |
| {class}`.View`{sup}`b` | *Optional* | *Optional* | | *Optional* | |

```{note}
{sup}`a` It is highly recommended to give [embeddings](../concepts/Embeddings.md)
over raw images using {class}`.Embeddings`.  
{sup}`b` Input data must be wrapped together in a `Dataset`.  
{sup}`c` When using only metadata, it must be wrapped in DataEval's {class}`.Metadata` class.  
{sup}`d` These scores are the raw outputs of a model.  
{sup}`e` These scores are retrieved by DataEval's Out Of Distribution (OOD) functions.  
```
