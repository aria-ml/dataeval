<!-- markdownlint-disable MD051 -->

# Data Integrity

The performance of any machine learning (ML) model is strictly bounded by the
quality of its training data. This is the **garbage in, garbage out** principle:
no amount of model sophistication can compensate for training data that is
redundant, corrupted, or statistically anomalous. Data Integrity refers to the
degree to which a dataset is free from these three categories of noise.

In test and evaluation (T&E) contexts, data integrity failures carry
operational consequences that go beyond poor benchmark scores. A model trained
on sensor-corrupted imagery may learn to associate artifacts with specific
targets. A dataset inflated with near-duplicates from a single collection event
may pass validation but fail badly when deployed against a different sensor
platform or environmental condition. Catching these problems before training is
orders of magnitude cheaper than discovering them after deployment ([Polyzotis et al., 2018](#ref2)).

## What is it

Data Integrity is a comprehensive condition of a dataset, not a single metric. A dataset has
high integrity when every sample it contains is loadable, non-redundant, within
the expected statistical range, and relevant to the problem domain. Conversely,
a dataset has low integrity when it contains a significant proportion of samples
that are duplicated, corrupted, or anomalous — even if those samples are
technically valid files.

DataEval approaches data integrity through four complementary tools:
{class}`.Duplicates` for redundancy detection, {class}`.Outliers` for anomaly
identification, {func}`.label_errors` for detecting mislabeled samples, and
{func}`.label_stats` for auditing label distribution structure. Both
{class}`.Duplicates` and {class}`.Outliers` can operate on raw image statistics;
{class}`.Outliers` can additionally operate on {term}`embeddings <Embeddings>`
from a pre-trained feature extractor, and {class}`.Duplicates` can use
embedding-based clustering as a third detection mode alongside its hash methods.

## Taxonomy of data noise

### Redundancy and duplication

A {term}`duplicate <Duplicates>` is any sample that is identical or
near-identical to another sample already in the dataset. Near-duplicates
include images with minor lighting shifts, JPEG re-compression artifacts,
slight crops of the same underlying scene, or rotated and flipped versions
of the same image.

The training impact of redundancy is well established. When a sample appears
multiple times, the model receives repeated gradient updates from information
it has already processed. This creates two compounding problems. First,
redundant samples inflate the effective size of the training set without adding
new signal, making training less compute-efficient. Second — and more
importantly for T&E — duplicate samples in a validation set cause its metrics
to be non-representative. If 30% of your validation images are near-duplicates
of training images, your reported accuracy is not a valid estimate of
out-of-distribution generalization.

[Birodkar et al. (2019)](#ref1) demonstrated that large benchmark datasets contain
substantial rates of semantic redundancy, and that removing it does not degrade
— and in some cases improves — model performance.

This is not an argument against data augmentation. Augmentation and deduplication
address different problems and operate at different points in the pipeline.
Deduplication is a pre-training step applied to the raw dataset: removing
near-duplicates before splitting reduces the risk of accidental train/validation
leakage and ensures that held-out metrics reflect genuine generalization. Runtime
augmentation — applied to the clean, split dataset during training — introduces
controlled variation that the model has not already memorized, which is precisely
what improves generalization. The two practices are complementary, not competing.

### Statistical outliers and semantic anomalies

Not all anomalous samples are the same, and the distinction matters for how
you handle them.

**Statistical outliers** are samples that fall at the extreme edges of a
measurable distribution — an extremely overexposed image, a frame with
near-zero entropy, a dimension ratio far outside the norm for the dataset.
These samples may or may not be semantically valid. A statistically extreme
image might be a rare but operationally important edge case, or it might be a
broken sensor frame. The statistical test alone cannot tell you which; it can
only flag the sample for inspection.

**Semantic anomalies** are samples that are statistically unremarkable but
belong to the wrong problem domain entirely — an image of a domestic animal in
an industrial inspection dataset, or a clear daytime frame in a dataset
intended for low-light detection. These pass all pixel-level quality checks but
actively harm the model by introducing irrelevant class structure. Embedding-
based outlier detection catches many semantic anomalies that statistical linting
misses, because the samples arrange differently in the embedding space, even when
their pixel statistics look normal.

### Corruption and sensor artifacts

Physical sensors introduce systematic errors that statistical linting is
specifically designed to catch. Common examples in computer vision include
motion blur from high-speed collection platforms, lens flare from direct sun
exposure, dead pixel rows from sensor damage, and blocking artifacts from
aggressive JPEG compression in transmission pipelines.

The integrity concern is subtler than simple corruption. A model that trains on
a dataset with a consistent rate of blur artifacts may train successfully — but
against the wrong distribution. If blur is consistently associated with a
particular collection scenario, the model can learn blur as a feature rather
than noise, creating a form of sensor-conditioned bias that only manifests when
the deployment sensor differs from the collection sensor. In situations, where
collection hardware is often upgraded or substituted over multiple phases, this
is a practical risk rather than an edge case.

### Label noise and mislabeling

Image-level and pixel-level integrity checks catch problems with the data
itself. A fourth category of integrity failure is quieter and harder to detect:
**label noise** — samples that are correctly captured but incorrectly annotated.

Label noise is common in large-scale annotation pipelines. Annotators make
mistakes, instructions are interpreted inconsistently, class boundaries are
ambiguous at the margins, and adversarial or rushed labeling produces systematic
errors in particular categories. In T&E datasets where labels may be generated
from automated pipelines or inherited from third parties, label quality is
often assumed rather than verified ([Sculley et al., 2015](#ref3)).

The consequences are severe. A mislabeled sample is not just useless — it
actively injects false gradient signal during training, pushing the model toward
the wrong decision boundary. Clusters of mislabeled samples around class
boundaries can shift the learned boundary significantly. In evaluation data,
mislabeled samples produce incorrect assessments of model performance:
a correct model prediction on a mislabeled sample counts as an error, and a
wrong model prediction counts as correct.

{func}`.label_errors` detects potential mislabeling by examining embedding
geometry. A correctly labeled sample should be closer to other samples of its
own class than to samples of any other class. The metric is the
**intra/extra class distance ratio**: the mean distance to the $k$ nearest
neighbors within the same class, divided by the mean distance to the $k$ nearest
neighbors in any other class. A ratio ≥ 1.0 means the sample is closer to a
different class than to its own — strong evidence of a labeling problem.

**Label statistics** ({func}`.label_stats`) address a different but related
concern: structural problems in the label distribution that are not about
correctness but about completeness and consistency. The function counts
`label_counts_per_class`, `image_counts_per_class`, `label_counts_per_image`,
and — critically for object detection — `empty_image_indices` (images with no
annotations). Empty images are a common annotation error: an image that was
included in the dataset but never labeled, or one where annotators missed all
objects. An unlabeled image trains the model to predict nothing, which is
situation dependent and often wrong.

## Theory

### Information density and gradient efficiency

From an information theory perspective, the goal of data integrity work is to
maximize the **information density** of the training set — the ratio of novel
signal to total sample count.

Consider a training set of $N$ samples, of which a fraction $r$ are
near-duplicates of existing samples. The effective number of unique training
examples is $N(1 - r)$. The duplicate samples still contribute gradients during
training, but those gradients point in directions the optimizer has already
processed. In the best case this wastes compute. In the worse case, on samples
that are nearly but not exactly identical, it biases the gradient toward the
over-represented region of the input space, effectively re-weighting the loss
landscape without the practitioner realizing it.

### Duplicate detection: hashing and clustering

{class}`.Duplicates` uses three complementary detection approaches, each
suited to a different kind of redundancy.

**Exact duplicate detection** uses xxHash, a fast non-cryptographic hash of
the raw image bytes. Two images with identical xxHash values are guaranteed to
be pixel-for-pixel identical. This is the most reliable signal: no threshold
tuning is required and there are no false positives.

**Near-duplicate detection** uses perceptual hashing, where two images that
are visually similar but not pixel-identical — due to compression, minor
cropping, or brightness adjustment — produce similar hash values. DataEval
supports two perceptual hash algorithms:

- **pHash (perceptual hash):** The image is resized to a square $N \times N$
  grid, a discrete cosine transform (DCT) is applied, and the lowest-frequency
  components are encoded as a bit array relative to the median coefficient value.
  The result is a compact hex string that is robust to minor photometric
  perturbations. [Zauner (2010)](#ref4) provides the foundational treatment.

- **dHash (difference hash):** Horizontal adjacent-pixel differences are
  computed on a downsampled image and encoded as a bit string. This approach
  is particularly robust to brightness and contrast shifts.

Both algorithms are available in **D4 variants** (`phash_d4`, `dhash_d4`) that
apply the hash across all eight orientations of the dihedral group (four
rotations × two reflections) and take the minimum, producing a hash invariant
to 90°/180°/270° rotations and horizontal/vertical flips. This matters for
datasets where images may be re-oriented during ingestion.

Two images are considered near-duplicates when the Hamming distance between
their hash values falls below a configurable threshold. Groups detected by
multiple methods carry higher confidence. When both basic and D4 hashes are
computed, the `orientation` column in the duplicates DataFrame is automatically
set to `"same"` (detected by basic hashes) or `"rotated"` (detected only by D4
hashes), letting practitioners distinguish the two cases.

**Cluster-based detection** is an optional third mode that operates in
embedding space rather than pixel space. When a {mod}`feature extractor <.extractors>`
and a `cluster_sensitivity` are provided, images are projected into embedding
space, clustered, and pairs whose embeddings fall within the threshold distance
are treated as near-duplicates. Because embeddings are approximate
representations, cluster-based matches are always reported as near rather than
exact duplicates, even when their embedding distance is zero. This mode catches
**semantic duplicates** — distinct photographs of the same object or scene that
are not similar at the pixel level but occupy the same region of embedding
space.

### Outlier detection: image statistics and embeddings

{class}`.Outliers` identifies anomalous samples through two independent paths
that can be used separately or together.

**Image statistics-based detection** computes pixel, visual, and dimension
statistics for each image using {func}`.compute_stats` with
{class}`.ImageStats` flags, then applies a statistical threshold test to each
metric distribution to flag samples at the extremes. Three tests are available:

The **z-score** method measures how many standard deviations a sample's value
$x_i$ lies from the distribution mean $\mu$:

$$z_i = \frac{|x_i - \mu|}{\sigma}$$

Samples exceeding the threshold (default: 3.0) are flagged. This method works
well for roughly normal distributions but is sensitive to the influence of
existing extreme values on the mean and standard deviation.

The **modified z-score** method substitutes the median $\tilde{x}$ and median
absolute deviation (MAD) for the mean and standard deviation, making it robust
to that influence:

$$\tilde{z}_i = \frac{0.6745 \cdot |x_i - \tilde{x}|}{\text{MAD}}$$

The constant 0.6745 is the 75th percentile of the standard normal distribution,
chosen so that the modified z-score is on the same scale as the standard z-score
for normally distributed data. The default threshold is 3.5.

The **interquartile range (IQR)** method flags samples whose distance from the
nearest quartile boundary exceeds a multiple of the IQR:

$$d_i = \max(Q_1 - x_i,\ x_i - Q_3) > \text{threshold} \times (Q_3 - Q_1)$$

The default threshold of 1.5 corresponds to the standard Tukey fence. This
method is the most robust to extreme values and requires no distributional
assumptions.

Each statistical test operates independently on each metric. A sample is
flagged if it exceeds the threshold on any single metric. The output records
both which metric triggered the flag and the measured value, so practitioners
can distinguish a brightness outlier from a dimension outlier and prioritize
accordingly.

**Cluster-based detection** operates in embedding space. When a feature
extractor is provided, images are projected into embedding space and clustered.
For each sample, the distance to its nearest cluster center is computed and
expressed as a number of standard deviations from that cluster's mean
intra-cluster distance. Samples exceeding the `cluster_threshold` (default: 2.5)
are flagged with a `cluster_distance` metric value. This path catches semantic
anomalies — samples that look statistically normal at the pixel level but do
not belong to any established class or scene type in the dataset.

Both detection paths can run simultaneously and their results are merged into
a single output DataFrame. A sample flagged by both paths warrants immediate
inspection.

### Image statistics as a linting vocabulary

{func}`.compute_stats` is the underlying computation engine for image
statistics in DataEval. It accepts any iterable of images or a MAITE-compliant
`Dataset` and computes whichever statistics are requested via the
{class}`.ImageStats` flag parameter. `ImageStats` is a selector, not a
processor — it is a `Flag` enum whose values you combine with bitwise OR to
specify which statistics you want, and `compute_stats` handles the rest.

The four flag categories and their integrity signals:

| Category    | Key statistics                                   | Integrity signal                                          |
| ----------- | ------------------------------------------------ | --------------------------------------------------------- |
| `PIXEL`     | Mean, std, entropy, skewness, kurtosis           | Corrupted or near-empty frames; distribution anomalies    |
| `VISUAL`    | Brightness, contrast, sharpness, darkness        | Exposure and focus failures; sensor artifacts             |
| `DIMENSION` | Width, height, aspect ratio, channels, bit depth | Incorrectly resized, cropped, or re-encoded samples       |
| `HASH`      | xxHash, pHash, dHash (+ D4 variants)             | Exact and near-duplicate detection (used by `Duplicates`) |

Some flags have dependencies that are resolved automatically: requesting
`PIXEL_ENTROPY` also enables `PIXEL_HISTOGRAM` (which entropy requires);
requesting `VISUAL_BRIGHTNESS`, `VISUAL_CONTRAST`, or `VISUAL_DARKNESS`
also enables `VISUAL_PERCENTILES`.

For **object detection** datasets, `compute_stats` can compute statistics
separately for each bounding box (`per_target=True`, the default when boxes are
present) and for the full image (`per_image=True`). Setting `per_channel=True`
produces per-channel breakdowns rather than aggregating across channels. The
`source_index` field in the output identifies whether each row corresponds to
a full image, a specific bounding box within an image, or a specific channel.

The default configuration for {class}`.Outliers` uses `DIMENSION | PIXEL |
VISUAL`, covering the full space of sensor-level integrity failures without
hash computation (which belongs to {class}`.Duplicates`).

### Label error detection: embedding geometry

{func}`.label_errors` operates on embeddings and class labels. For each sample,
it computes:

$$\text{score}_i = \frac{\bar{d}_{\text{intra},i}}{\bar{d}_{\text{extra},i}}$$

where $\bar{d}_{\text{intra},i}$ is the mean distance from sample $i$ to its
$k$ nearest neighbors within the same class, and $\bar{d}_{\text{extra},i}$
is the mean distance to its $k$ nearest neighbors in any other class ($k$
defaults to 50, capped at `min_class_size - 1`).

The output contains three fields:

**`errors`**: a dictionary mapping sample index to `(original_label, [suggested_labels])`
for all samples with score ≥ 1.0. Suggested labels are derived from
rank-weighted voting over the $k$ nearest out-of-class neighbors — closer
neighbors receive higher weight. The suggestion logic applies two thresholds:
if the top candidate's vote share is below `min_confidence` (default 0.4), no
suggestion is returned; if the margin between the top two candidates is smaller
than `ambiguity_threshold` (default 0.2), both are returned as a tie.

**`error_rank`**: all sample indices sorted by descending score — a triage
list for human review, starting with the samples most likely to be mislabeled.

**`scores`**: the raw distance ratio for every sample, useful for setting custom
thresholds or examining the distribution of scores.

A score well above 1.0 is stronger evidence of mislabeling than a score just
at the boundary. The `error_rank` allows reviewers to prioritize the highest-
confidence detections and stop reviewing when the confidence drops below a
practical threshold.

## When to use it

Data Integrity assessment is appropriate at three points in the data lifecycle.

**Before any training run.** Running the full integrity pipeline on a new or
merged dataset is standard practice before committing compute to training.
Integrity failures caught here cost a data review. Integrity failures caught
after a failed training run cost a training run plus a data review.

**After merging datasets.** Near-duplicate rates spike when datasets from
different collection events or public sources are combined without
deduplication. A dataset assembled from three sources with 10% internal
redundancy each can easily reach 25–30% redundancy at the merged level.

**When evaluation metrics look suspicious.** Anomalously high validation
accuracy — especially accuracy that degrades sharply in operational testing —
is a common symptom of validation set contamination by training-set duplicates.
If your held-out metrics seem too good, run deduplication across the
train/validation split boundary.

To better understand what to do after running an assessment, review the
[Data Integrity section in the Acting on Results explanation page](ActingOnResults.md#data-integrity-findings).

## Limitations

Statistical linting and perceptual hashing are effective at catching the
integrity failures they were designed to detect, but both have blind spots
practitioners should understand.

Hash-based detection will not catch **semantic duplicates** — two different
photographs of the same object taken from different angles or lighting
conditions that are genuinely different images at the pixel level. Cluster-based
duplicate detection in embedding space addresses this, but requires a feature
extractor whose embedding space is meaningful for the target domain.

Image statistics-based outlier detection is only sensitive to pixel-level and
dimensional anomalies. A semantically incorrect image with normal brightness,
contrast, and dimensions will not be flagged by statistical tests. Cluster-based
outlier detection addresses this case, again contingent on embedding quality.

Neither approach addresses label quality and fails to detect **label errors** —
images that are correctly exposed, non-duplicated, and statistically normal but
assigned the wrong class label. To understand label quality, the functions
{func}`.label_stats` and {func} `.label_errors` have to be run independently.

The statistical outlier tests flag samples relative to the distribution of the
dataset being evaluated. Results are therefore dataset-dependent: adding or
removing samples changes what counts as an outlier. When comparing results
across dataset versions, re-run the full analysis rather than assuming prior
flags remain valid.

## Related concept pages

- [Clustering](Clustering.md) — the underlying algorithm used by Duplicates
  (cluster mode), Outliers (cluster mode), and label_errors
- [Dataset Bias and Coverage](DatasetBias.md) — when your data is clean but
  still unrepresentative
- [Distribution Shift](DistributionShift.md) — when your data was clean at
  training time but the deployment distribution has changed
- [Acting on Results](ActingOnResults.md) — how to prioritize remediation
  across integrity, bias, and performance findings

## See this in practice

### How-to guides

- [How to detect and remove duplicates](../notebooks/h2_deduplicate.md)
- [How to visualize data cleaning issues](../notebooks/h2_visualize_cleaning_issues.md)
- [How to perform cluster analysis](../notebooks/h2_cluster_analysis.md)

### Tutorials

- [Data cleaning tutorial](../notebooks/tt_clean_dataset.md) — end-to-end
  walkthrough of integrity assessment on a realistic dataset

## References

1. [Birodkar, V., Mobahi, H., & Bengio, S. (2019). Semantic redundancy in image
   classification datasets. _arXiv preprint arXiv:1901.11409._ [paper](https://arxiv.org/abs/1901.11409)]{#ref1}

2. [Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2018). Data management
   challenges in production machine learning. In _Proceedings of the 2017 ACM
   SIGMOD International Conference on Management of Data_ (pp. 1723–1726). [paper](https://dl.acm.org/doi/10.1145/3035918.3054782)]{#ref2}

3. [Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D.,
   Chaudhuri, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden
   technical debt in machine learning systems. In _Advances in Neural Information
   Processing Systems_ (Vol. 28). [paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)]{#ref3}

4. [Zauner, C. (2010). Implementation and benchmarking of perceptual image hash
   functions. _Bachelor's thesis, Upper Austria University of Applied Sciences._ [thesis](https://www.phash.org/docs/pubs/thesis_zauner.pdf)]{#ref4}
