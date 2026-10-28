# Functional overview

This page lists the dataset tasks that each DataEval tool accepts, and what you do to your data first. Find a tool in the
tables, read across its row and check the cell for your task. Each tool name links to its API reference.

To choose a tool for a problem, see [Which DataEval tool should I use?](../getting-started/which-tool.md). For the inputs
that each tool consumes, including whether it needs class labels, see
[What data does each tool need?](../getting-started/input-requirements.md).

## How to read the tables

DataEval accepts three kinds of dataset: image classification, object detection and video tracking. Each tool works on
one unit, such as an image, an object or a frame. A dataset of another task reaches that unit through an adapter, and
the cell names it. [Adapting your data](#adapting-your-data) explains each adapter.

| Column or mark | Meaning |
| -------------- | ------- |
| Image classification, Object detection, Video tracking | The task of your dataset. A function on the Metrics tab takes arrays, so the mark means that the arrays come from a dataset of that task with the arguments the function documents. |
| ✔ | The tool takes the dataset as it is. |
| Crops | The tool needs one embedding for each label. Present each box as an image with {class}`.DetectionCrops`. |
| Frames | The tool needs one image for each frame. Present the tracking dataset with {class}`.SequenceFrames`. |
| Groups | The tool treats each row as an independent sample, but the rows of one image or video are not independent. Pass `groups` so that they count as one unit. |
| ⚠️ | The tool is experimental and its API may change. See [Stability status](../concepts/ValidationAndTrust.md#stability-status). |
| Blank | The tool does not take that task. |

## Computer vision task compatibility

The tabs follow the public API. Metrics lists the functions of {mod}`dataeval.core`, Evaluators lists the classes of
`dataeval.bias`, `dataeval.quality`, `dataeval.scope` and `dataeval.shift`, Performance lists `dataeval.performance`,
Metadata lists the metadata functions of `dataeval.core`, and Data selection lists `dataeval.data`. The functions that
other tools call, such as the hash and ranking functions, and the extractors and models that create embeddings, appear
only in the [API reference](autoapi/dataeval/index.rst).

`````{tab-set}
:sync-group: func

````{tab-item} Metrics
:sync: metrics

```{list-table}
:widths: 32 38 10 10 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image classification
  - Object detection
  - Video tracking
* - {func}`Bayes error rate (KNN) <.ber_knn>`\
{func}`Bayes error rate (MST) <.ber_mst>`
  - Bounds the irreducible classification error caused by class overlap in an embedding space. The KNN version takes
    `groups`
  - ✔
  - Crops
  - Crops, Groups
* - {func}`Box-to-image ratio statistics <.compute_ratios>`
  - Computes the ratio of each box statistic to the same statistic of its image. Deprecated in v1.2 and removed in
    v1.3. See the [v1.2 migration guide](../migration/v1.2.md) for the replacement
  -
  - ✔
  - Frames
* - {func}`Clustering <.cluster>`
  - Clusters embeddings with KMeans or HDBSCAN
  - ✔
  - ✔
  - Frames
* - {func}`Completeness <.completeness>`
  - Measures the dimensional utilization of the embedding space through eigenvalue entropy
  - ✔
  - ✔
  - Frames
* - {func}`Coverage (adaptive) <.coverage_adaptive>`\
{func}`Coverage (naive) <.coverage_naive>`
  - Measures how well the distribution of samples in a dataset covers the embedding space
  - ✔
  - ✔
  - Frames
* - {func}`Divergence (FNN) <.divergence_fnn>`\
{func}`Divergence (MST) <.divergence_mst>`
  - Measures the difference between the distributions of two datasets
  - ✔
  - ✔
  - Frames
* - {func}`Feature distance <.feature_distance>`
  - Measures the feature-wise distance between two continuous distributions
  - ✔
  - ✔
  - Frames
* - {func}`Image and box statistics <.image_stats>`
  - Computes statistics of each image and, when given boxes, of each box
  - ✔
  - ✔
  - Frames
* - {func}`Label alignment <.label_alignment>`
  - Aligns a label vocabulary with an ontology
  - ✔
  - ✔
  - ✔
* - {func}`Label co-occurrence <.label_cooccurrence>`
  - Reports which classes appear together in an item, with lift, Jaccard index and adjusted p-values
  -
  - ✔
  - ✔
* - {func}`Label coverage <.label_coverage>`
  - Reports how the label counts of a dataset are distributed over an ontology
  - ✔
  - ✔
  - ✔
* - {func}`Label errors <.label_errors>`
  - Computes potential label errors in a dataset using embeddings
  - ✔
  - Crops
  - Crops
* - {func}`Label parity <.label_parity>`
  - Tests whether two datasets share one class distribution (label shift). Detections take one group for each label,
    the image or video it came from
  - ✔
  - Groups
  - Groups
* - {func}`Label reconciliation <.label_reconciliation>`
  - Reconciles class labels with an ontology and recovers their hierarchy
  - ✔
  - ✔
  - ✔
* - {func}`Label statistics <.label_stats>`
  - Computes statistical summaries of labels in a dataset
  - ✔
  - ✔
  - ✔
* - {func}`Null model metrics <.nullmodel_metrics>`
  - Calculates performance metrics for random classifiers on training and testing labels based on the class
    distributions
  - ✔
  - ✔
  - ✔
* - {func}`Ontology validation <.ontology_validation>`
  - Validates an ontology and reports its structure and naming
  - ✔
  - ✔
  - ✔
* - {func}`Parity <.parity>` ⚠️
  - Tests each factor, and each factor-level and class cell, for association with the class labels over independent
    groups
  - ✔
  - ✔
  - ✔
* - {func}`UAP <.uap>` ⚠️
  - Measures the average precision one crop classifier reaches on ground-truth object detection boxes, the
    classification part of the task without localization error
  -
  - ✔
  - ✔
```

````

````{tab-item} Evaluators
:sync: evaluators

```{list-table}
:widths: 32 38 10 10 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image classification
  - Object detection
  - Video tracking
* - {class}`.Balance`
  - Measures the normalized mutual information between the class label and the metadata factors, and between the
    factors
  - ✔
  - ✔
  - ✔
* - {class}`Chunked drift <.ChunkedDrift>`
  - Wraps a drift detector to report drift for each chunk of a stream
  - ✔
  - ✔
  - Frames
* - {class}`.Coverage`
  - Measures per-class coverage and dispersion of the embedding space of a dataset
  - ✔
  - Crops
  - Crops
* - {class}`.Diversity`
  - Measures how evenly the metadata factors are distributed, overall and for each class
  - ✔
  - ✔
  - ✔
* - {class}`Drift domain classifier <.DriftDomainClassifier>`\
{class}`Drift K-nearest neighbors <.DriftKNeighbors>`\
{class}`Drift known domains <.DriftKnownDomains>` ⚠️\
{class}`Drift MMD <.DriftMMD>`\
{class}`Drift reconstruction <.DriftReconstruction>`\
{class}`Drift univariate <.DriftUnivariate>`\
{class}`Drift Wasserstein <.DriftWasserstein>`
  - Detects data distribution shift of new data from the reference data
  - ✔
  - ✔
  - Frames
* - {class}`Duplicate detection <.Duplicates>`
  - Finds exact and near-duplicate images and videos by hashing and embedding clusters
  - ✔
  - ✔
  - ✔
* - {class}`Out-of-distribution domain classifier <.OODDomainClassifier>`\
{class}`Out-of-distribution K-nearest neighbors <.OODKNeighbors>`\
{class}`Out-of-distribution reconstruction <.OODReconstruction>`
  - Detects samples that fall outside the reference distribution
  - ✔
  - ✔
  - Frames
* - {class}`.Outliers`
  - Flags images, boxes and metadata values that lie far from the rest of the dataset, by statistics or by embedding
    clusters
  - ✔
  - ✔
  - Frames
* - {class}`.Parity` ⚠️
  - Tests each factor, and each factor-level and class cell, for association with the class labels over independent
    images or videos
  - ✔
  - ✔
  - ✔
* - {class}`Prioritization <.Prioritize>`
  - Orders samples by their position in the embedding space
  - ✔
  - Crops
  - Crops
* - {class}`.Representation`
  - Measures how well the labels of a dataset cover an ontology, and which classes to collect
  - ✔
  - ✔
  - ✔
```

````

````{tab-item} Performance
:sync: performance

```{list-table}
:widths: 32 38 10 10 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image classification
  - Object detection
  - Video tracking
* - {class}`.Feasibility` ⚠️
  - Decides whether a classification target (accuracy, AP or recall) is demonstrated, ruled out under recorded
    assumptions, or neither, and returns the evidence behind each determination
  - ✔
  -
  - Crops, Groups
* - {class}`.Sufficiency`
  - Determines how much training data a performance target needs, by training the model on growing subsets and fitting
    a power law
  - ✔
  - ✔
  - ✔
```

````

````{tab-item} Metadata
:sync: metadata

```{list-table}
:widths: 32 38 10 10 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image classification
  - Object detection
  - Video tracking
* - {func}`Factor deviation <.factor_deviation>`
  - Computes the deviation of each flagged sample from the reference median, for each metadata factor
  - ✔
  - ✔
  - ✔
* - {func}`Factor predictors <.factor_predictors>`
  - Measures, on a scale from 0 to 1, how strongly each metadata factor is associated with the flagged samples
  - ✔
  - ✔
  - ✔
```

````

````{tab-item} Data selection
:sync: data_selection

```{list-table}
:widths: 32 38 10 10 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image classification
  - Object detection
  - Video tracking
* - {func}`Dataset splitter <.split_dataset>`
  - Returns the indices of train, validation and test splits, stratified on labels and metadata. A tracking dataset
    splits by video
  - ✔
  - ✔
  - ✔
* - {func}`Merge datasets <.merge_datasets>`
  - Concatenates datasets that share a label vocabulary into one dataset view
  - ✔
  - ✔
  - ✔
* - {class}`.View`
  - Builds a dataset view from an ordered pipeline of operations, listed below. An operation that reads labels needs
    them
  - ✔
  - ✔
  - ✔
```

The operations of a {class}`.View`:

```{list-table}
:widths: 32 38 10 10 10
:header-rows: 1
:class: table-text

* - Operation
  - Description
  - Image classification
  - Object detection
  - Video tracking
* - {class}`.ClassBalance`
  - Selects a subset of images with a balanced class distribution
  - ✔
  - ✔
  - Frames
* - {class}`.Filter`
  - Keeps the items that match a condition on class labels and metadata factors, and the matching detections in them
  - ✔
  - ✔
  - ✔
* - {class}`.Relabel`
  - Conforms the class labels of a dataset to a target vocabulary through a class mapping
  - ✔
  - ✔
  - Frames
* - {class}`.Crop`\
{class}`.Resize`\
{class}`.SelectChannels`\
{class}`.TorchvisionTransform`
  - Change the images and rewrite the bounding boxes to match
  - ✔
  - ✔
  - Frames
* - {class}`.Indices`\
{class}`.Limit`\
{class}`.Reverse`\
{class}`.Shuffle`
  - Keep, cap or reorder the items
  - ✔
  - ✔
  - ✔
* - {class}`.DetectionCrops`
  - Presents the ground-truth boxes of an object detection dataset as an image classification dataset
  -
  - ✔
  - Frames
```

````

`````

## Adapting your data

A tool works on one unit: an image, one object, a frame or a whole video. When your dataset presents another unit,
convert it first. The cells use three conversions, and a cell can name two.

### Crops

{class}`.DetectionCrops` presents each ground-truth box as an image of its own. A tool that needs one embedding for each
class label, such as {class}`.Coverage`, {func}`.label_errors` and {func}`.ber_knn`, reads the crops, because an
object detection dataset has one label for each box and one embedding for each image. A tracking dataset has labels in
its frames, so take the crops of its frames:

```python
crops = DetectionCrops(SequenceFrames(dataset))
```

### Frames

{class}`.SequenceFrames` presents a tracking dataset as an object detection dataset with one item for each frame. A tool
that reads images, embeddings or detections takes the frames. Pass a frame selector, such as {class}`.Stride`, to keep
fewer frames, because the number of frames sets the cost of every tool that reads them.

A drift or out-of-distribution detector can instead take the tracking dataset as it is, one item for each video, when you
give it an extractor that reads a video.

### Groups

The frames of one video are not independent samples, and neither are the boxes of one image. A tool that tests a
distribution or searches for neighbors reads each row as independent, so near-identical frames of one video make its
result look stronger than it is. Pass `groups`, one id for each row, so that the rows of one video count as one unit.
{func}`.label_parity` and {func}`.ber_knn` take `groups`, and {class}`.Feasibility` takes it for grouped data, such as
video. {class}`.Parity` reads its groups from the {class}`.Metadata` of the dataset.

## Tools for video

A tracking dataset holds one video in each item. Its labels are the detections in every frame, so a tool that counts
labels counts a track once for each frame it appears in, and a tool that reports for each item reports for each video.
For a worked example, see [Identify duplicate video](../notebooks/tt_identify_duplicate_video.py).

These tools apply only to video and have no row in the tables.

| Tool | Use |
| ---- | --- |
| {class}`.SequenceFrames` | Presents the tracking dataset as an object detection dataset of frames, as described in [Frames](#frames). |
| {class}`.Stride`, {class}`.EvenlySpaced`, {class}`.FrameRate`, {class}`.Redundancy`, {class}`.Representative`, {class}`.QualityGate`, {class}`.CameraTravel`, {class}`.BestPerTrack`, {class}`.HasLabels`, {class}`.TrackEvents`, {class}`.FrameIndices`, {class}`.AllFrames` and {class}`.Chain` | Frame selectors. Pass one to {class}`.SequenceFrames` to choose which frames it keeps. |
| {class}`.VideoSegments`, {class}`.Window`, {class}`.Cuts` | Present the tracking dataset as video segments, cut into fixed-length windows or at frame boundaries you name. |
| {class}`.VideoStitch` | Combines videos of the tracking dataset into stitched sequences. |
| {func}`.ego_stats` | Measures the camera motion of each frame. Needs the `dataeval[opencv]` extra. |
| {func}`.video_stats` | Computes statistics of each video as a whole, one row for each video. The `PERSISTENCE` statistics are experimental, see [Stability status](../concepts/ValidationAndTrust.md#stability-status). |
| {func}`.track_stats` | Computes motion statistics of each track. |
| {func}`.build_tracks` | Builds a dict of tracks for one target or a whole dataset. |
