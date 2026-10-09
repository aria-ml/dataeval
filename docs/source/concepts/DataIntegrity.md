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

A {term}`duplicate <Duplicates>` is not simply "a sample identical or
near-identical to another" — that description only covers content, and content
is one of three projections a duplicate can be found in. A duplicate is a
**collision in some projection of the datum**: reduce a sample through its
content, through its annotation, or through its coded metadata factors, and two
samples whose reduction agrees are duplicates in that projection, whether or
not they agree in any other.

| Projection | What it reduces to |
| --- | --- |
| Content | phash / dhash / xxhash over decoded frames |
| Annotation | a digest over boxes, labels, and track ids |
| Metadata factors | the coded {class}`.Metadata` factor row |

**Content** is the common case and the one most tooling assumes — for images
and video frames, the pixels. Near-duplicates here include images with minor
lighting shifts, JPEG re-compression artifacts, slight crops of the same
underlying scene, or rotated and flipped versions of the same image.

**Annotation** duplicates share a box/label/track-id digest even when their
content differs or was never compared. **Factor** duplicates share a named
{class}`.Metadata` factor row — a capture timestamp, GPS fix, or source
filename — even before either image is decoded.

Which projections agree, and which disagree, is itself the finding: two
samples agreeing on content but disagreeing on annotation is one collect
carrying two conflicting label passes; agreeing on annotation while
disagreeing on content is a synthetically augmented copy; agreeing on both is a
plain re-ingest under a new name. {class}`.Duplicates` reports which
projections it checked and which of those agreed on every group — see
[Duplicates and near-duplicates](ActingOnResults.md#duplicates-and-near-duplicates)
for how to read that against a specific result.

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

The marginal kind is the one most tooling implements, and it misses the other
three. A timestamp is roughly uniform over a dataset, so a wrong one is
unremarkable until it is compared with its neighbors. A large, centered box is
unremarkable if other classes are also large and centered. Most annotation and
metadata defects are *conditional*: a value that is improbable only given the
rest of the datum.

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

{class}`.Duplicates` computes the content projection through three complementary
approaches, each suited to a different kind of redundancy. Two further
projections — annotation and metadata factors — are covered afterward.

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

**Annotation detection** is on by default whenever a dataset carries targets
(object detection boxes and labels). Each item's annotation — its boxes,
labels, and, for tracking data, track ids — is reduced to a digest; items
whose digest agrees are grouped as an annotation duplicate, whether or not
their content does. Because it runs alongside content detection rather than
instead of it, the same pair of items can be checked on both projections at
once: a pair agreeing on content but not on annotation surfaces as an exact
content match whose annotation disagreed (one collect carrying two conflicting
label passes); a pair agreeing on annotation but not on content surfaces as an
annotation duplicate whose content disagreed (a synthetically augmented copy).
No feature extractor or threshold is needed — it is an exact, transitive
comparison, like xxHash for pixels.

**Factor detection** is the only projection that is opt-in. Passing
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
For each sample, the distance to its nearest cluster center is computed.
Samples exceeding the `cluster_threshold` (default: 2.5) are flagged with a
`cluster_distance` metric value — the distance itself, reported beside that
cluster's own `population_mean` and `population_std`, so the number of standard
deviations it sits out by is `(metric_value - population_mean) /
population_std`. This path catches semantic anomalies — samples that look statistically normal at the pixel level but do
not belong to any established class or scene type in the dataset.

Both detection paths can run simultaneously and their results are merged into
a single output DataFrame. A sample flagged by both paths warrants immediate
inspection.

### Outlier detection over annotation and metadata factors

```{note}
{class}`.Outliers` thresholds ordered metadata factors -- each against the rows
at the level it was measured at -- when a {class}`.Metadata` is passed as
`data`. The stratified, categorical, and sequential detection described below is
in development; this section describes the method it follows and the evidence
behind its defaults.
```

Extending {class}`.Outliers` beyond image statistics raises three questions the
image path never faced: which columns can be thresholded, how a categorical
value can be an outlier, and how small a population can support a finding.

#### Which columns can be thresholded

A {class}`.Metadata` holds whatever was attached to it. Several kinds of column
are numeric without being measurements -- identifiers, booleans, and the codes
a categorical factor is stored as -- and thresholding them produces findings
that are hard to act on.

The tempting rule is to threshold only factors that {class}`.Metadata`
classifies as continuous. That classification exists to choose **bins**, so it
answers a different question. It is a heuristic over the spacing of the values
(see [Binning](Binning.md)) that calls data discrete whenever the values sit on
a lattice or number fewer than 20:

| Column | Classified continuous? | Meaningful to threshold? |
| --- | --- | --- |
| gap count per track (Poisson counts, n = 3000) | no | yes |
| track duration in frames (integers, n = 3000) | no | yes |
| mean speed (floating point, n = 3000) | yes | yes |
| any per-video factor in a 12-video dataset | no — fewer than 20 rows | yes, with the small-n caveat below |
| capture time in epoch seconds at 1 Hz | no — a lattice | yes, sequentially |

Gating on that classification would silently exclude every integer count a
track produces and every factor at the sequence level. Eligibility is instead
decided by the **raw column's type**:

- **Ordered** — integers, floats, datetimes, durations. Thresholded on the raw
  values with the same tests the image path uses.
- **Categorical** — strings, booleans, and integers the caller has declared as
  a fixed vocabulary. Never given a location or scale; judged by rarity within a
  stratum (below).
- **Ineligible** — list-valued columns, and the reserved addressing columns
  (row indices, `track_id`, `item_id`), by construction rather than by
  heuristic.

Integer identifiers are the one case a type check cannot catch -- a numeric
`camera_serial` looks ordered. Declare it as a fixed vocabulary, or name the
factors to analyze explicitly. A cardinality heuristic is deliberately not
used: it would add a threshold to tune and defend.

#### Why bins are never used

A factor's coded form — its bin indices — describes the cut, not the
measurement: a z-score over bin codes `[0, 1, 2, 3]` measures the bin edges.
Two further problems exist. A categorical factor's numeric reading is its
codes, and a missing categorical value is stored as one code past the last
level rather than as NaN, so a missing value would read as the highest level
instead of being excluded. Stratifying on a binned continuous factor makes
every stratum depend on the bin edges.

Detection therefore reads raw values only, and stratifies only on categorical
columns. A continuous conditioning variable is a joint relationship, not a
stratum. As a result, **re-binning a metadata does not change an outlier
result.**

#### Relative box geometry

Box width, height, area, and position are measured in pixels. Stratifying them
by class compares a box in a 4K frame against one in a 640-pixel frame, and
resolution dominates the comparison. Geometry relative to the image — area as a
fraction of the frame, offsets as fractions of width and height, distance from
the center as a fraction of the half-diagonal — turns "this class is usually a
small box near the corner" into a statement about composition, not the camera.
Absolute size remains its own question: it determines whether an object is
detectable at all.

#### Categorical rarity within a stratum

A categorical value is an outlier when it is **improbable given its stratum**:
one frame stamped `weather="snow"` inside an otherwise clear 300-frame
sequence, or three frames of a sequence reporting a different `sensor_id`.

No widely used data validation tool ships a default for this. Great
Expectations, TensorFlow Data Validation, Deequ, ydata-profiling, Evidently,
and whylogs flag values outside a known domain or shares the user sets; none
flags a rare value with a default frequency cutoff, and none uses a minimum
count. The default below is derived rather than borrowed.

**The test.** For a stratum of $n$ rows in which level $v$ appears $k$ times
(counting the row being judged), the value is flagged when the exact one-sided
binomial test rejects "$v$ occurs in this stratum with probability at least
$p$":

$$P(X \le k \mid X \sim \mathrm{Binomial}(n, p)) \le \alpha$$

Equivalently, the one-sided Clopper–Pearson upper confidence bound on
$P(v \mid s)$ falls below $p$ ([Clopper & Pearson, 1934](#ref5)); that bound is
the number reported, because it reads directly as "this level occurs in at most
this share of its stratum". Every row of a level within a stratum shares one
test, so the unit of the test is the (stratum, level) cell.

**Defaults.** Rarity $p = 0.05$ at 95 % confidence ($\alpha = 0.05$). The
confidence level follows the conventional limit used for zero- and
small-numerator bounds ([Hanley & Lippman-Hand, 1983](#ref6)). The rarity is a
judgment: of $p \in \{0.01, 0.02, 0.05\}$ it is the only value under which
three wrong-sensor frames in a 300-frame sequence are flagged.

**The minimum group size.** A singleton can only be flagged once the stratum is
large enough for $P(X \le 1) \le \alpha$, which at the defaults is $n \ge 93$
— approximately $4.74 / p$, the one-occurrence extension of the "rule of
three" for zero occurrences. [Das & Schneider (2007)](#ref7) derive the minimum
support of their categorical detector from its significance level in the same
way. One setting, $p$, moves both the rarity and the minimum. Strata below the
minimum are reported as a ranking, without a verdict, and are left out of the
multiple-testing count — a discrete test that cannot reach significance need
not count against the ones that can ([Tarone, 1990](#ref8)). Across the
testable cells at a level, the Benjamini–Hochberg procedure controls the false
discovery rate at 0.05 ([Benjamini & Hochberg, 1995](#ref9)).

**Levels rare everywhere are not outliers.** A class that is rare in every
stratum is a finding about the dataset's balance (see [Dataset Bias and
Coverage](DatasetBias.md)). [Das & Schneider (2007)](#ref7) make the same
argument — a pairing that is rare because both of its parts are rare "can be
explained" — and divide by the marginal frequencies to remove it. That
correction fails a common video case: with 100 sequences each captured on its
own sensor, every sensor has a global share of about 1 %, so three frames of
sensor B inside sequence A look expected. The rule used instead: a level is
withheld as rare everywhere only when it is rare in *every* stratum large enough
to test. A level common in at least one stratum — the snow videos, sensor B's
own sequence — stays flagged for its rare appearances elsewhere.

| Case | Upper bound on the level's share | Flagged |
| --- | --- | --- |
| 1 snow frame in a clear 300-frame sequence | 0.016 | yes |
| 3 wrong-sensor frames in a 300-frame sequence | 0.026 | yes |
| 2 "truck" frames in a 60-frame "car" track | 0.101 | no — 60 rows cannot support it |

The last case is not a coverage gap. Label drift within a track is already
measured as an ordered, per-track quantity — the chosen label's share of the
track — and thresholded against other tracks. The categorical test covers strata
too large for such a summary to exist.

**The rejected alternative.** A conformal p-value — the share of the stratum
made of levels at least as rare as the row's own — is the other candidate, and
is what the "q-value" of [Das & Schneider (2007)](#ref7) computes. Its validity
is exact at any sample size ([Laxhammar, 2014](#ref10); [Bates et al., 2023]
(#ref11)), and it can flag a singleton in a stratum of $1/\alpha$ rows. But
scored within the stratum it bounds the *number* of flags — at most
$\lfloor \alpha n \rfloor$ rows per stratum — rather than giving evidence that
the value is improbable. [Laxhammar (2014)](#ref10) notes that a conformal
anomaly may be "a relatively rare … example generated from the same probability
distribution".

| | Conformal p-value | Exact binomial |
| --- | --- | --- |
| What a flag claims | the row is in the rarest $\alpha$ of its stratum | its level's share is below $p$, with 95 % confidence |
| Smallest stratum that can flag a singleton | $1/\alpha$ (20 at $\alpha$ = 0.05) | 93 at $p$ = 0.05 |
| Rows flagged in a clean stratum of 20 classes with Zipf-distributed shares | about $\alpha$ of them — 0.8 % at $\alpha$ = 0.01 | 0.07–0.26 % |
| A legitimate level with true share 0.5 %, $n$ = 300, $\alpha$ = 0.01 | flagged in 71 % of simulations | flagged only if its share is truly below $p$ |
| Benjamini–Hochberg across strata | loses power: no p-value can fall below $1/n$ | usable |

The binomial rule is the one that does not flood a clean, diverse dataset with
findings about its rare classes.

**Calibration beside the ordered tests.** On clean data the default adaptive
threshold flags about 0.0004 % of normal values, 0.09 % of gamma-distributed
values, and 0.6 % of log-normal values (5,000 values, 50 draws each). The
binomial rule's 0.07–0.26 % on a clean stratum is the same order, so ordered and
categorical findings carry comparable weight in one result.

**Known limits.** Both tests assume the rows of a stratum are exchangeable.
Consecutive video frames are not: a three-frame burst of a wrong value is
closer to one event than to three independent ones. A stratum smaller than the
minimum cannot support a verdict on its own; borrowing strength across strata
with an empirical-Bayes prior is the principled route to judging them, and is
not part of the default.

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
present) and for the full image (`per_image=True`). The `source_index` field in
the output identifies whether each row corresponds to a full image or to a
specific bounding box within an image.

Bands are addressed with `channels=`, which names groups of them and measures
each group jointly, returning `<group>_<statistic>` columns on the same rows —
`channels={"r": 0, "g": 1, "b": 2}` for RGB, or
`channels={"visible": range(0, 30), "nir": range(30, 70)}` for a hyperspectral
cube. Because they are columns rather than rows, band statistics reach
{class}`.Metadata` and are visible to {class}`.Balance` and {class}`.Diversity`
like any other factor.

Object detection datasets can also be measured on the pixels that *are not*
annotated. `per_background=True` unions every box in an image into a mask,
excludes those pixels, and reduces the requested statistics over what is left —
the scene an image was captured in, rather than the things annotated inside it:

```python
stats = compute_stats(
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
statistic describes — the image — and so belong at the `unit` level (see
[Metadata Levels](MetadataLevels.md)). Three properties are worth knowing:

- **`background_fraction` is always returned**, and should be read before the
  rest. It is the share of the image left unmasked, and a background statistic
  measured over a few percent of an image is noise wearing a measurement's
  clothes. Where an image's boxes cover it entirely, every other background
  statistic for that image is `NaN`.
- **Only `PIXEL` and `VISUAL` statistics are computed for the background.** A
  masked region has no meaningful hash and no geometry of its own, so `HASH` and
  `DIMENSION` flags are computed for the image and its boxes as usual and skipped
  for the background.
- **Boxes are rounded outwards and unioned**, so the background excludes slightly
  more than the annotations strictly cover. This is deliberate: a retained pixel
  is background with high confidence, at the cost of discarding a boundary ring
  of genuine background along with the object.

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
{func}`.label_stats` and {func}`.label_errors` have to be run independently.

The statistical outlier tests flag samples relative to the distribution of the
dataset being evaluated. Results are therefore dataset-dependent: adding or
removing samples changes what counts as an outlier. When comparing results
across dataset versions, re-run the full analysis rather than assuming prior
flags remain valid.

Stratified and categorical outlier tests assume the rows of a stratum are
exchangeable, which consecutive video frames are not, and they return no
verdict on strata too small to support one. Relationships between two ordered
quantities — box size against distance from the camera, image brightness
against capture hour — are joint and are not found by testing one column at a
time, within a stratum or not.

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

- [How to detect and remove duplicates](../notebooks/h2_deduplicate.py)
- [How to find outliers in metadata factors](../notebooks/h2_find_factor_outliers.py)
- [How to visualize data cleaning issues](../notebooks/h2_visualize_cleaning_issues.py)
- [How to perform cluster analysis](../notebooks/h2_cluster_analysis.py)

### Tutorials

- [Data cleaning tutorial](../notebooks/tt_clean_dataset.py) — end-to-end
  walkthrough of integrity assessment on a realistic dataset

## References

1. [Birodkar, V., Mobahi, H., & Bengio, S. (2019). Semantic redundancy in image
   classification datasets. *arXiv preprint arXiv:1901.11409.* [paper](https://arxiv.org/abs/1901.11409)]{#ref1}

2. [Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2018). Data management
   challenges in production machine learning. In *Proceedings of the 2017 ACM
   SIGMOD International Conference on Management of Data* (pp. 1723–1726). [paper](https://dl.acm.org/doi/10.1145/3035918.3054782)]{#ref2}

3. [Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D.,
   Chaudhuri, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden
   technical debt in machine learning systems. In *Advances in Neural Information
   Processing Systems* (Vol. 28). [paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)]{#ref3}

4. [Zauner, C. (2010). Implementation and benchmarking of perceptual image hash
   functions. *Bachelor's thesis, Upper Austria University of Applied Sciences.* [thesis](https://www.phash.org/docs/pubs/thesis_zauner.pdf)]{#ref4}

5. [Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial
   limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404–413. [paper](https://academic.oup.com/biomet/article-abstract/26/4/404/291538)]{#ref5}

6. [Hanley, J. A., & Lippman-Hand, A. (1983). If nothing goes wrong, is
   everything all right? Interpreting zero numerators. *JAMA*, 249(13),
   1743–1745. [paper](https://jhanley.biostat.mcgill.ca/c607/ch08/zero_numerator.pdf)]{#ref6}

7. [Das, K., & Schneider, J. (2007). Detecting anomalous records in categorical
   datasets. In *Proceedings of the 13th ACM SIGKDD International Conference on
   Knowledge Discovery and Data Mining* (pp. 220–229). [paper](https://dl.acm.org/doi/10.1145/1281192.1281219)]{#ref7}

8. [Tarone, R. E. (1990). A modified Bonferroni method for discrete data.
   *Biometrics*, 46(2), 515–522. [paper](https://pubmed.ncbi.nlm.nih.gov/2364136/)]{#ref8}

9. [Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
   a practical and powerful approach to multiple testing. *Journal of the Royal
   Statistical Society: Series B*, 57(1), 289–300. [paper](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)]{#ref9}

10. [Laxhammar, R. (2014). *Conformal anomaly detection: Detecting abnormal
    trajectories in surveillance applications.* PhD thesis, University of
    Skövde. [thesis](https://www.diva-portal.org/smash/get/diva2:690997/FULLTEXT02.pdf)]{#ref10}

11. [Bates, S., Candès, E., Lei, L., Romano, Y., & Sesia, M. (2023). Testing for
    outliers with conformal p-values. *The Annals of Statistics*, 51(1),
    149–178. [paper](https://arxiv.org/abs/2104.08279)]{#ref11}
