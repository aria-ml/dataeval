<!-- markdownlint-disable MD051 -->

# Data integrity

Training data quality is a major determinant of machine learning (ML) model performance and generalization.
Redundant samples, corrupted samples, statistical and semantic anomalies, and label noise can reduce training
efficiency, introduce unintended statistical associations, and compromise the validity of evaluation results. Data
integrity is the degree to which a dataset is free from these four kinds of noise.

In test and evaluation (T&E) contexts, data integrity failures can have operational consequences beyond a poor
benchmark score. A model trained on sensor-corrupted imagery may learn to associate artifacts with specific targets. A
dataset inflated with near-duplicates from a single collection event may pass validation and still perform poorly when
deployed on a different sensor platform or in a different environment. Problems found before training are typically
less costly to correct than problems found after deployment ([Polyzotis et al., 2017](#ref2)).

## What data integrity is

Data integrity is a property of a dataset assessed along several dimensions, not a single metric. A dataset has high
integrity when its samples are loadable, non-redundant, within the expected statistical range, and relevant to the
problem domain. A dataset has low integrity when a significant proportion of its samples are duplicated, corrupted, or
anomalous, even if those samples are technically valid files.

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

A {term}`duplicate <Duplicates>` is more than "a sample identical or
near-identical to another". That description only covers content, and content is
one of three projections a duplicate can be found in. A duplicate is
a **collision in some projection of the datum**: reduce a sample through its
content, through its annotation, or through its coded metadata factors, and two
samples whose reduction agrees are duplicates in that projection, whether or not
they agree in any other.

| Projection | What it reduces to |
| --- | --- |
| Content | phash / dhash / xxhash over decoded frames |
| Annotation | a digest over boxes, labels, and track ids |
| Metadata factors | the coded {class}`.Metadata` factor row |

Content is the common case: for images and video frames, the pixels. Near-duplicates here include images with minor
lighting shifts, JPEG re-compression artifacts, slight crops of the same underlying scene, or rotated and flipped
versions of the same image.

Annotation duplicates share a box/label/track-id digest even when their content differs or was never compared.
Factor duplicates share a named {class}`.Metadata` factor row (a capture timestamp, GPS fix, or source filename) even
before either image is decoded.

The pattern of agreement across projections indicates the kind of redundancy. Agreement on content with disagreement
on annotation is consistent with one collection event carrying two conflicting label passes. Agreement on annotation
with disagreement on content is consistent with a synthetically augmented copy. Agreement on both is consistent with a
re-ingest under a new name. {class}`.Duplicates` reports which projections it checked and which of those agreed on
every group. See [Duplicates and near-duplicates](ActingOnResults.md#duplicates-and-near-duplicates) for how to read
that against a specific result.

Redundancy affects training and evaluation in two ways. When a sample appears multiple times, the model receives
repeated gradient updates from information it has already processed. First, redundant samples inflate the effective
size of the training set without adding new signal, which makes training less compute-efficient. Second, and more
important for T&E, duplicate samples shared between training and validation sets make the validation metrics
non-representative. If 30% of the validation images are near-duplicates of training images, the reported accuracy does
not estimate out-of-distribution generalization.

[Birodkar et al. (2019)](#ref1) demonstrated that large benchmark datasets
contain substantial rates of semantic redundancy, and that removing it does not
degrade, and in some cases improves, model performance.

Deduplication does not replace data augmentation. The two address different problems at different points in the
pipeline. Deduplication is a pre-training step applied to the raw dataset: removing near-duplicates before splitting
reduces the risk of accidental train/validation leakage, so held-out metrics are less likely to overstate
generalization. Runtime augmentation, applied to the clean, split dataset during training, introduces controlled
variation that the model has not already memorized, which can improve generalization. The two practices are
complementary.

### Statistical outliers and semantic anomalies

Anomalous samples differ in kind, and the kind determines how a flagged sample is handled.

**Statistical outliers** are samples that fall at the extreme edges of a
measurable distribution, such as an extremely overexposed image, a frame with
near-zero entropy, or a dimension ratio far outside the norm for the dataset.
These samples may or may not be semantically valid. A statistically extreme
image might be a rare but operationally important edge case, or it might be a
broken sensor frame. The statistical test does not distinguish the two cases; it flags the sample for inspection.

**Semantic anomalies** are samples that are statistically unremarkable but
belong to the wrong problem domain entirely, such as an image of a domestic
animal in an industrial inspection dataset, or a clear daytime frame in a
dataset intended for low-light detection. These pass pixel-level quality checks but can reduce model performance by
introducing irrelevant class structure. Embedding-based outlier detection can find semantic anomalies that statistical
linting does not, because such samples can occupy different regions of the embedding space even when their pixel
statistics are normal.

An outlier, like a duplicate, can be found in any projection of the datum:
its content, its annotation (box geometry, labels, tracks), or its metadata
factors. The reference population a value is judged against determines the
kind of finding:

| Kind | Judged against | Example |
| --- | --- | --- |
| Marginal | every row at its level | a track with 11 gaps where tracks have 0–1 |
| Stratified | rows sharing a categorical context (class, sequence, or categorical factor) | a class that is always a small box near the corner, drawn once large and centered |
| Sequential | the row's neighbors within its sequence | a timestamp that jumps three hours for one frame; a GPS fix that moves 40 km and back |
| Joint | a multivariate or continuous-on-continuous relationship | in one class, distant boxes are small and near ones large; a large box at the horizon |

A marginal check does not find the other three kinds. A timestamp is roughly uniform over a dataset, so a wrong one
is unremarkable until it is compared with its neighbors. A large, centered box is unremarkable if other classes are
also large and centered. Many annotation and metadata defects are **conditional**: a value that is improbable only
given the rest of the datum.

### Corruption and sensor artifacts

Physical sensors can introduce systematic errors that statistical linting can detect. Common examples in computer
vision include motion blur from high-speed collection platforms, lens flare from direct sun exposure, dead pixel rows
from sensor damage, and blocking artifacts from aggressive JPEG compression in transmission pipelines.

The integrity concern extends beyond individual corrupted samples. A model trained on a dataset with a consistent
rate of blur artifacts may train successfully but on a distribution that differs from deployment. If blur is
consistently associated with a particular collection scenario, the model can learn blur as a feature, which is a form
of sensor-conditioned bias. The bias may appear only when the deployment sensor differs from the collection sensor.
Where collection hardware is upgraded or substituted over multiple phases, this risk is more likely to arise.

### Label noise and mislabeling

Image-level and pixel-level integrity checks catch problems with the data
itself. A fourth category of integrity failure is harder to detect:
**label noise**, meaning samples that are correctly captured but incorrectly
annotated.

Label noise is common in large-scale annotation pipelines. Annotators make
mistakes, instructions are interpreted inconsistently, class boundaries are
ambiguous at the margins, and adversarial or rushed labeling produces systematic
errors in particular categories. In T&E datasets where labels may be generated
from automated pipelines or inherited from third parties, label quality is
often assumed rather than verified ([Sculley et al., 2015](#ref3)).

Mislabeled samples degrade both training and evaluation. During training, a mislabeled sample contributes gradient
signal toward an incorrect decision boundary, and clusters of mislabeled samples near class boundaries can shift the
learned boundary. In evaluation data, mislabeled samples bias the assessment of model performance: a correct model
prediction on a mislabeled sample counts as an error, and a wrong prediction counts as correct.

{func}`.label_errors` detects potential mislabeling by examining embedding
geometry. If the embedding separates the classes, a correctly labeled sample is closer to other samples of its own
class than to samples of any other class. The metric is
the **intra/extra class distance ratio**: the mean distance to the $k$ nearest
neighbors within the same class, divided by the mean distance to the $k$ nearest
neighbors in any other class. A ratio ≥ 1.0 means the sample is at least as close to another class as to its own.
This is consistent with a labeling problem. It is also consistent with an ambiguous sample or an embedding that does
not separate the classes.

Label statistics ({func}`.label_stats`) address a different but related concern: structural problems in the label
distribution that concern completeness and consistency and not correctness. The function counts
`label_counts_per_class`, `image_counts_per_class`, `label_counts_per_image`, and, for object detection,
`empty_image_indices` (images with no annotations). An empty image can be an annotation error: an image that was
included in the dataset but never labeled, or one where annotators missed all objects. Training on such an image
teaches the model to predict no objects, which is incorrect when objects are present. An empty image can also be a
legitimate negative example, so whether it is an error depends on the task.

## Theory

### Information density and gradient efficiency

The **information density** of a training set is the ratio of novel signal to total sample count. Redundant samples
lower it.

Consider a training set of $N$ samples, of which a fraction $r$ are near-duplicates of existing samples. The effective
number of unique training examples is $N(1 - r)$. This is the approximate number
that remains after deduplication under the chosen similarity criterion and is not
a statistical effective sample size: dependencies can remain among samples that
are unique under that criterion. The duplicate samples still contribute gradients during training, but
those gradients largely repeat information already contributed by the originals. At minimum this wastes compute. For
near-identical samples it can also bias the gradient toward the over-represented region of the input space, which
re-weights the loss without the practitioner being aware of it.

### Duplicate detection: hashing and clustering

{class}`.Duplicates` computes the content projection through three complementary
approaches, each suited to a different kind of redundancy. Two further
projections, annotation and metadata factors, are covered afterward.

Exact duplicate detection uses xxHash (the 64-bit xxh3 variant), a fast non-cryptographic hash of the raw image
bytes. Images with identical bytes have identical hashes, and no threshold is required. The converse holds with high
probability and not with certainty. Two different byte strings can collide. If the hashes of $n$ images behave as
uniform 64-bit values, the probability of at least one collision is approximately $n^2 / 2^{65}$, which is about
$3 \times 10^{-8}$ for one million images. The hash input is the array's raw bytes, so arrays of different shape or
dtype that hold the same bytes have the same hash. DataEval groups images by hash value and does not compare the
pixels of the grouped images.

Near-duplicate detection uses perceptual hashing, where two images that are
visually similar but not pixel-identical, due to compression, minor cropping, or
brightness adjustment, produce similar hash values. DataEval supports two
perceptual hash algorithms:

- pHash (perceptual hash): the image is resized to a square $N \times N$ grid, a
  discrete cosine transform (DCT) is applied, and the lowest-frequency
  components are encoded as a bit array relative to the median coefficient
  value. The result is a compact hex string that is robust to minor photometric
  perturbations. [Zauner (2010)](#ref4) provides the foundational treatment.

- dHash (difference hash): horizontal adjacent-pixel differences are computed on
  a downsampled image and encoded as a bit string. This approach is particularly
  robust to brightness and contrast shifts.

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
hashes), letting practitioners distinguish the two cases. Video relations follow
the same rule: a whole-video match, a shared segment or a shared track found only
under a D4 hash is `"rotated"`, so a video re-uploaded a quarter turn on is
reported as related to the original. Redundancy within one video is read from
the basic hashes only, because consecutive frames of a video are not rotations of
each other.

Cluster-based detection is an optional third mode that operates in
embedding space rather than pixel space. When a {mod}`feature extractor <.extractors>`
and a `cluster_sensitivity` are provided, images are projected into embedding
space, clustered, and pairs whose embeddings fall within the threshold distance
are treated as near-duplicates. Because embeddings are approximate
representations, cluster-based matches are always reported as near rather than
exact duplicates, even when their embedding distance is zero. This mode catches
**semantic duplicates**, which are distinct photographs of the same object or
scene that are not similar at the pixel level but occupy the same region of
embedding space.

Annotation detection is on by default whenever a dataset carries targets
(object detection boxes and labels). Each item's annotation (its boxes, labels,
and, for tracking data, track ids) is reduced to a digest; items whose digest
agrees are grouped as an annotation duplicate, whether or not their content
does. Because it runs alongside content detection rather than instead of it, the
same pair of items can be checked on both projections at once: a pair agreeing
on content but not on annotation surfaces as an exact content match whose
annotation disagreed (consistent with one collection event carrying two conflicting label passes); a pair
agreeing on annotation but not on content surfaces as an annotation duplicate
whose content disagreed (consistent with a synthetically augmented copy). No feature extractor
or threshold is needed: it is an exact, transitive comparison, like xxHash for
pixels.

Factor detection is the only projection that is opt-in. Passing
`duplicate_factors` to {meth}`.Duplicates.evaluate` names one or more
{class}`.Metadata` factors whose exact agreement makes two items duplicates,
such as a capture timestamp, a GPS fix, or a source filename.

The factor names form a **conjunction**. Two items are duplicates only when
every named factor agrees. Agreeing on some factors while differing on another
does not produce a duplicate. Because matching applies to the entire row,
adding a factor can only split groups. Use additional factors to narrow an
identifier that is not unique on its own (for example,
`duplicate_factors=["capture_date", "camera_id"]` requires agreement on both).
Items with unstated factors (null, or NaN in numeric columns) are excluded from
grouping.

Factor detection is disabled unless factors are explicitly named. There is no
default set or selection heuristic. The signal is valid only when the named
factors identify an individual item rather than a shared condition. For
example, items sharing `weather=rain` reflect shared conditions that belong in
{class}`.Balance` and {class}`.Coverage`. Grouping on low-cardinality factors
produces large groups that obscure actual duplicates.

### Outlier detection: image statistics and embeddings

{class}`.Outliers` identifies anomalous samples through two independent paths
that can be used separately or together.

Image statistics-based detection computes pixel, visual, and dimension
statistics for each image using {func}`.image_stats` with
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
method is the least sensitive to extreme values of the three tests and does not assume a normal distribution.

Each statistical test operates independently on each metric. A sample is
flagged if it exceeds the threshold on any single metric. The output records
both which metric triggered the flag and the measured value, so practitioners
can distinguish a brightness outlier from a dimension outlier and prioritize
accordingly.

Cluster-based detection operates in embedding space. When a feature
extractor is provided, images are projected into embedding space and clustered.
For each sample, the distance to its nearest cluster center is computed.
Samples exceeding the `cluster_threshold` (default: 2.5) are flagged with a
`cluster_distance` metric value: the distance itself, reported beside that
cluster's own `population_mean` and `population_std`, so the number of standard
deviations it sits out by is `(metric_value - population_mean) /
population_std`. This path catches semantic anomalies: samples that look
statistically normal at the pixel level but do not belong to any established
class or scene type in the dataset.

Both detection paths can run simultaneously and their results are merged into
a single output DataFrame. Review a sample flagged by both paths first.

### Outlier detection over annotation and metadata factors

{class}`.Outliers` also reads a {class}`.Metadata`. It then judges ordered metadata factors against the rows at the
level they were measured at, categorical factors by their rarity within a stratum, combinations of factors, and values
that depart from the rows next to them along a video or track. The extension raises three questions the image path
never faced: which columns can be thresholded, how a categorical value can be an outlier, and how small a population can
support a finding. [Outliers in metadata factors](MetadataOutliers.md) describes the method behind each and the evidence
behind its defaults.

### Image statistics in integrity checks

{func}`.image_stats` is the underlying computation engine for image
statistics in DataEval. It accepts any iterable of images or a MAITE-compliant
`Dataset` and computes whichever statistics are requested via the
{class}`.ImageStats` flag parameter. `ImageStats` is a selector, not a
processor: it is a `Flag` enum whose values you combine with bitwise OR to
specify which statistics you want, and `image_stats` handles the rest.

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

For object detection datasets, `image_stats` can compute statistics separately
for each bounding box (`per_target=True`, the default when boxes are present)
and for the full image (`per_image=True`). The `source_index` field in the
output identifies whether each row corresponds to a full image or to a specific
bounding box within an image.

Bands are addressed with `channels=`, which names groups of them and measures
each group jointly, returning `<group>_<statistic>` columns on the same rows:
`channels={"r": 0, "g": 1, "b": 2}` for RGB, or
`channels={"visible": range(0, 30), "nir": range(30, 70)}` for a hyperspectral
cube. Because they are columns rather than rows, band statistics reach
{class}`.Metadata` and are visible to {class}`.Balance` and {class}`.Diversity`
like any other factor.

Object detection datasets can also be measured on the pixels that are not
annotated. `per_background=True` unions every box in an image into a mask,
excludes those pixels, and reduces the requested statistics over what is left,
meaning the scene an image was captured in, rather than the things annotated
inside it:

```python
stats = image_stats(
    dataset,
    stats=ImageStats.PIXEL | ImageStats.VISUAL,
    per_background=True,
    normalize_pixel_values=False,
)
# One value per row, as every stats array is. With the default per_target=True the
# rows are images *and* boxes, and a box has no background, so its entry is null.
# `source_index` says which is which; `add_factors` uses it to place them.
stats["stats"]["background_brightness"]
```

Background values are returned under `background_`-prefixed names on the same
rows as the whole-image ones, because they describe the same thing a whole-image
statistic describes, the image, and so belong at the `unit` level (see
[Metadata levels](MetadataLevels.md)). The background statistics have three properties:

- `background_fraction` is always returned. It is the share of the image left unmasked, and a background statistic
  measured over a few percent of an image is noise that looks like a measurement. Read it before the other background
  statistics. Where an image's boxes cover it entirely, every other background statistic for that image is `NaN`.
- Only `PIXEL` and `VISUAL` statistics are computed for the background. A masked region has no meaningful hash and no
  geometry of its own, so `HASH` and `DIMENSION` flags are computed for the image and its boxes as usual and skipped for
  the background.
- Boxes are rounded outwards and unioned, so the background excludes slightly more than the annotations strictly cover.
  This is deliberate: a retained pixel is background with high confidence, at the cost of discarding a boundary ring of
  genuine background along with the object.

The default configuration for {class}`.Outliers` uses `DIMENSION | PIXEL | VISUAL`. It does not compute hashes, which
belong to {class}`.Duplicates`.

### Label error detection: embedding geometry

{func}`.label_errors` operates on embeddings and class labels. For each sample,
it computes:

$$\text{score}_i = \frac{\bar{d}_{\text{intra},i}}{\bar{d}_{\text{extra},i}}$$

where $\bar{d}_{\text{intra},i}$ is the mean distance from sample $i$ to its
$k$ nearest neighbors within the same class, and $\bar{d}_{\text{extra},i}$
is the mean distance to its $k$ nearest neighbors in any other class ($k$
defaults to 50, capped at `min_class_size - 1`).

The output contains three fields:

`errors`: a dictionary mapping sample index to
`(original_label, [suggested_labels])` for all samples with score ≥ 1.0.
Suggested labels are derived from rank-weighted voting over the $k$ nearest
out-of-class neighbors: closer neighbors receive higher weight. The suggestion
logic applies two thresholds: if the top candidate's vote share is below
`min_confidence` (default 0.4), no suggestion is returned; if the margin between
the top two candidates is smaller than `ambiguity_threshold` (default 0.2), both
are returned as a tie.

`error_rank`: all sample indices sorted by descending score (a triage list for
human review, starting with the highest scores).

`scores`: the raw distance ratio for every sample, useful for setting custom
thresholds or examining the distribution of scores.

A higher score indicates a larger gap between the same-class and other-class neighbor distances, so `error_rank`
orders samples for review from the largest gap down. Reviewers can stop at a score cutoff they choose. The score is a
ratio of distances and is not a probability that the label is wrong.

## Interpretation

Each tool reports evidence about samples. None of them reports that a sample is invalid.

- {class}`.Duplicates` reports groups of samples that agree in a projection. An exact content match means identical
  raw bytes, apart from hash collisions. A near-duplicate match means the perceptual hashes lie within the Hamming
  threshold, or the embeddings within the cluster distance. It does not establish that two samples show the same scene.
  The projections that agree indicate the kind of redundancy, as described in
  [Redundancy and duplication](#redundancy-and-duplication).
- {class}`.Outliers` reports samples whose value lies beyond a threshold fitted to the population it was judged
  against. A flag establishes that a value is extreme relative to that population. It does not establish that the
  sample is invalid, because a rare but valid edge case is flagged in the same way. For metadata factors, a population
  below the floor is not judged (see [Small populations](MetadataOutliers.md#small-populations)), and the absence of a
  flag there is not evidence that its rows are normal.
- {func}`.label_errors` reports a score, the ratio of the mean same-class to the mean other-class neighbor distance. A
  score of at least 1.0 is consistent with a labeling problem, an ambiguous sample, or an embedding that does not
  separate the classes. The score is not a probability.
- {func}`.label_stats` reports counts. It describes the label distribution and does not assess whether labels are
  correct.

To decide what to do after running an assessment, see
[Data integrity findings](ActingOnResults.md#data-integrity-findings) in Acting on results.

## Limitations

Statistical linting and perceptual hashing detect the integrity failures they were designed for, and both have
blind spots.

Hash-based detection will not catch semantic duplicates, meaning two different
photographs of the same object taken from different angles or lighting
conditions that are different images at the pixel level. Cluster-based duplicate
detection in embedding space addresses this, but requires a feature extractor
whose embedding space is meaningful for the target domain.

Image statistics-based outlier detection is only sensitive to pixel-level and
dimensional anomalies. A semantically incorrect image with normal brightness,
contrast, and dimensions will not be flagged by statistical tests. Cluster-based
outlier detection addresses this case, again contingent on embedding quality.

Neither approach detects label errors, meaning images that are correctly exposed, non-duplicated, and statistically
normal but assigned the wrong class label. Label quality is assessed separately with {func}`.label_stats` and
{func}`.label_errors`.

The statistical outlier tests flag samples relative to the distribution of the
dataset being evaluated. Results are therefore dataset-dependent: adding or
removing samples changes what counts as an outlier. When comparing results
across dataset versions, re-run the full analysis rather than assuming prior
flags remain valid.

Stratified and categorical outlier tests assume the rows of a stratum are
exchangeable, which consecutive video frames are not, and they return no
verdict on strata too small to support one. Relationships between two ordered
quantities (box size against distance from the camera, image brightness against
capture hour) are joint and are not found by testing one column at a time,
within a stratum or not. The neighbor check (`neighbors=`) is the test that
reads consecutive frames in order: it judges a frame against the frames next to
it, and reads a run of more than five departing frames as a segment. [Outliers in metadata factors](MetadataOutliers.md)
describes these tests.

## When to use it

Assess data integrity at three points in the data lifecycle.

Before a training run: assess a new or merged dataset before committing compute to training. Problems found at this
point cost a data review. Problems found after a failed training run cost a training run and a data review.

After merging datasets: near-duplicate rates can rise when datasets from different collection events or public sources
are combined without deduplication, because the sources can overlap.

When evaluation metrics look suspicious: anomalously high validation accuracy, especially accuracy that degrades
sharply in operational testing, can indicate validation set contamination by training-set duplicates. In that case,
run deduplication across the train/validation split boundary.

## Related concept pages

- [Outliers in metadata factors](MetadataOutliers.md): thresholds, strata, combinations and neighbors over metadata
  factors
- [Clustering](Clustering.md): the underlying algorithm used by Duplicates
  (cluster mode), Outliers (cluster mode), and label_errors
- [Dataset bias](DatasetBias.md) and [Dataset coverage](DatasetCoverage.md): when your data is clean but
  still unrepresentative
- [Distribution shift](DistributionShift.md): when your data was clean at
  training time but the deployment distribution has changed
- [Acting on results](ActingOnResults.md): how to prioritize remediation
  across integrity, bias, and performance findings

## See this in practice

### How-to guides

- [Identify duplicates](../notebooks/h2_deduplicate.py)
- [Run clustering analysis](../notebooks/h2_cluster_analysis.py)
- [Specify custom statistics on object detection datasets](../notebooks/h2_custom_image_stats_object_detection.py)

### Tutorials

- [Clean a dataset](../notebooks/tt_clean_dataset.py): end-to-end
  walkthrough of integrity assessment on a realistic dataset
- [Compare objects to their surroundings](../notebooks/tt_compare_objects_to_surroundings.py): comparing a
  detection's statistics with its background and with other detections
- [Identify duplicate video](../notebooks/tt_identify_duplicate_video.py): finding duplicate sequences and redundant
  footage in a tracking dataset

## References

1. [Birodkar, V., Mobahi, H., & Bengio, S. (2019). Semantic redundancy in image
   classification datasets. *arXiv preprint arXiv:1901.11409.* [paper](https://arxiv.org/abs/1901.11409)]{#ref1}

2. [Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2017). Data management
   challenges in production machine learning. In *Proceedings of the 2017 ACM
   SIGMOD International Conference on Management of Data* (pp. 1723–1726). [paper](https://dl.acm.org/doi/10.1145/3035918.3054782)]{#ref2}

3. [Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D.,
   Chaudhuri, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden
   technical debt in machine learning systems. In *Advances in Neural Information
   Processing Systems* (Vol. 28). [paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)]{#ref3}

4. [Zauner, C. (2010). Implementation and benchmarking of perceptual image hash
   functions. *Bachelor's thesis, Upper Austria University of Applied Sciences.* [thesis](https://www.phash.org/docs/pubs/thesis_zauner.pdf)]{#ref4}
