<!-- markdownlint-disable MD018 -->

# Which DataEval tool should I use?

If you already know which metric or class you need, see the
[Functional Overview](../reference/FunctionalOverview.md) or the
[API reference](../reference/autoapi/dataeval/index.rst). This page is for
practitioners who know where they are in the
[machine learning lifecycle](../getting-started/roles/ML_Lifecycle.md) and
what problem they are facing, but are not yet sure which DataEval capability
addresses it.

The sections below follow the operational ML lifecycle stages where DataEval
applies. DataEval does not provide tools specific to the Deployment or Analysis
stages of the lifecycle — those stages are addressed by the
[Acting on Results concept page](../concepts/ActingOnResults.md), which explains
how to interpret findings and what actions they support.

---

## Data Engineering

Data quality problems — duplicates, corrupted images, mislabeled samples, and
statistical outliers — inflate benchmark scores and produce models that
underperform in deployment. Catching these problems before training is orders of
magnitude cheaper than discovering them after deployment. See the
[Data Integrity concept page](../concepts/DataIntegrity.md) for an explanation
of each failure mode and its training impact.

### Are there near-duplicate or exact-duplicate images in my dataset?

Use {class}`.Duplicates`. Exact duplicates are found via hash comparison;
near-duplicates via image statistics or embedding-based clustering.
See the [deduplication how-to guide](../notebooks/h2_deduplicate.md).

### Are there images that look visually corrupted, blurry, or otherwise anomalous?

Use {func}`.compute_stats` to compute per-image statistics (brightness,
contrast, sharpness, aspect ratio, entropy), then {class}`.Outliers` to flag
images at the extremes of those distributions.
See the [visualizing cleaning issues how-to guide](../notebooks/h2_visualize_cleaning_issues.md).

### Are there samples that are statistical outliers in embedding space?

Use {class}`.Outliers` in cluster mode. This detects semantic anomalies —
samples that look statistically normal at the pixel level but do not belong
to any established class or scene type.
See the [Data Integrity concept page](../concepts/DataIntegrity.md) for when
embedding-based outlier detection applies versus image statistics-based detection.

### Are any of my samples mislabeled?

Use {func}`.label_errors` to identify samples whose labels are inconsistent
with their nearest neighbors in embedding space.
See the [Data Integrity concept page](../concepts/DataIntegrity.md).

### What does the label distribution of my dataset look like?

Use {func}`.label_stats` to audit class counts and per-class sample sizes
before any bias analysis.

### Are my class labels unevenly distributed?

Use {class}`.Balance` to measure class frequency imbalance.
See the [Dataset Bias and Coverage concept page](../concepts/DatasetBias.md) and the
[detecting undersampling how-to guide](../notebooks/h2_detect_undersampling.md).

### Are certain attribute groups underrepresented in my dataset?

Use {class}`.Diversity` to measure representation across metadata factors
(image conditions, object characteristics, collection circumstances).
See the [Dataset Bias and Coverage concept page](../concepts/DatasetBias.md).

### Are label distributions inconsistent across subgroups?

Use {class}`.Parity` or {func}`.label_parity` to test for statistical
disparities in labeling across groups.
See the [identify bias tutorial](../notebooks/tt_identify_bias.md).

### Does my dataset cover the range of conditions the model will encounter in deployment?

Use {func}`.coverage_naive` or {func}`.coverage_adaptive` to identify gaps in the feature space and
{func}`.completeness` to measure how effectively the dataset spans available
embedding dimensions.
See the [Dataset Bias and Coverage concept page](../concepts/DatasetBias.md).

### Are my training and test splits drawn from the same distribution?

Use any of the drift detectors (see [Monitoring](#monitoring) below) to test
for train/test shift. A significant result before deployment is a data
curation problem, not a monitoring problem.

---

## Model Development

Model development decisions — architecture selection, training data curation,
and pre-deployment evaluation — depend on understanding what the current dataset
is capable of producing. See the
[Performance Limits concept page](../concepts/PerformanceLimits.md).

```{important}
{func}`.ber_knn` and {func}`.ber_mst` apply to **image classification only**.
For object detection tasks, use {func}`.uap`,
which extends feasibility analysis by treating ground-truth bounding box crops
as a classification problem. See the
[Performance Limits concept page](../concepts/PerformanceLimits.md) for details.
```

### Can a model plausibly learn to classify this dataset at all?

Use {func}`.ber_knn` or {func}`.ber_mst` to estimate the Bayes Error Rate (BER) — the theoretical
lower bound on classification error given the data as it is. A high BER means
the dataset is the limiting factor, not the model architecture.

### Can a detector plausibly achieve my mAP requirement on this dataset?

Use {func}`.uap` to estimate the classification
ceiling on bounding box crops. If UAP falls below your program's mAP requirement,
no object detector trained on the current data can meet it.

### Do I have enough labeled examples to train a reliable model?

Use {class}`.Sufficiency` to estimate how model performance scales with dataset
size and whether collecting more data is likely to improve results.

### Which unlabeled samples should I prioritize for labeling?

Use {class}`.Prioritize` to rank unlabeled samples by how novel they are
relative to already-labeled data. Hard-first ranking identifies the samples
most unlike the current training distribution — the additions most likely to
improve coverage and generalization.

### Should I collect more data, or is my current dataset already saturated?

Run {class}`.Sufficiency` first to assess the learning curve trajectory. If
the curve has flattened, also run {func}`.coverage_naive` or {func}`.coverage_adaptive` and
{func}`.completeness` to determine whether gaps in the feature space — rather than
total sample count — are the binding constraint.

---

## Monitoring

Distribution Shift — the gap between training distribution and operational
distribution — is a planning assumption for any long-fielded system. See the
[Distribution Shift concept page](../concepts/DistributionShift.md) for a
taxonomy of shift types and guidance on choosing the right detector.

DataEval provides two complementary monitoring capabilities: **drift detection**
operates at the population level; **out-of-distribution (OOD) detection**
operates at the per-sample level. Both are necessary; neither is sufficient alone.

### Has the distribution of my operational data shifted away from my training data?

Start with the [Distribution Shift concept page](../concepts/DistributionShift.md)
to choose the right detector for your data characteristics:

| If your data...                                           | Use                                                     |
| --------------------------------------------------------- | ------------------------------------------------------- |
| Has continuous, well-behaved distributions                | {class}`.DriftUnivariate` (method="ks")                 |
| Has small sample sizes                                    | {class}`.DriftMMD` (Maximum Mean Discrepancy)           |
| Has heavy tails or is non-parametric                      | {class}`.DriftUnivariate` (method="bws")                |
| Has labeled outputs you can evaluate against              | {class}`.DriftUnivariate` with an uncertainty extractor |
| Needs a flexible, model-based test                        | {class}`.DriftDomainClassifier`                         |
| Uses pre-computed embeddings; needs lightweight detection | {class}`.DriftKNeighbors` (K-nearest neighbor)          |
| Needs reconstruction-based detection or streaming support | {class}`.DriftReconstruction`                           |

See the [monitor shift tutorial](../notebooks/tt_monitor_shift.md).

### How far has my operational data shifted from training, quantitatively?

Use {func}`.divergence_fnn` or {func}`.divergence_mst` to compute HP divergence —
a continuous score between 0 and 1. Drift detectors tell you _whether_ a shift
occurred; divergence tells you _how much_. Tracking divergence over time reveals
gradual drift that may never trigger a single-batch test but is nonetheless eroding
the relevance of the training distribution.
See the [Divergence concept page](../concepts/Divergence.md) and the
[measure divergence how-to guide](../notebooks/h2_measure_divergence.md).

### Are there individual samples in my operational data that fall outside the training distribution?

Use DataEval's out-of-distribution (OOD) detection to flag inputs that are
anomalous relative to what the model was trained on. Three approaches are
available:

| Approach                      | Best for                                                      |
| ----------------------------- | ------------------------------------------------------------- |
| {class}`.OODReconstruction`   | Structural defects; when feature-level localization is needed |
| {class}`.OODKNeighbors`       | Fast deployment; strong pre-trained embeddings available      |
| {class}`.OODDomainClassifier` | Semantic anomalies; complex feature interactions              |

See the [Distribution Shift concept page](../concepts/DistributionShift.md) and
the [identify OOD samples tutorial](../notebooks/tt_identify_ood_samples.md).

### Are sensor inputs degrading in quality during operation?

Use {func}`.compute_stats` on incoming data to track per-image quality metrics
over time. Systematic shifts in brightness, contrast, or sharpness distributions
indicate sensor degradation or environmental change before they manifest as model
performance drops.

---

## I'm not sure where to start

If you are new to DataEval, start with the
[data cleaning tutorial](../notebooks/tt_clean_dataset.md). It walks through
the most common first-pass analysis tasks — duplicates, outliers, and image
quality — and gives you a working foundation before moving on to bias analysis,
performance limits, or drift detection.

Once you have results, the
[Acting on Results concept page](../concepts/ActingOnResults.md) explains what
each category of output means and what actions it supports.
