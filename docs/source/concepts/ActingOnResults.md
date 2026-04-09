# Acting on Results

DataEval produces diagnostic outputs, not decisions. Every evaluator returns
scores, indices, flags, or p-values — but none of them tell you what to do next.
That judgment requires understanding what the result means in the context of
your program, your data collection constraints, and your performance
requirements.

This page maps each category of DataEval output to the decisions it informs and
the actions available to you. It is not a procedure to follow linearly. Real
datasets present multiple issues simultaneously, and fixing one can reveal or
interact with another. Use this page as a reference: when a diagnostic produces
a result you need to act on, find the relevant section and understand your
options before committing to a course of action.

One framing applies across all of it: DataEval finds _evidence_ of problems. The
interpretation is always yours. A high BER upper bound is evidence that class
overlap may be irreducible — it is not proof that your requirement is
unachievable. A detected duplicate is evidence of potential redundancy — it is
not an instruction to delete. Every DataEval output is a prompt to investigate,
not a verdict.

## Data Integrity findings

### Duplicates and near-duplicates

{class}`.Duplicates` returns three groups of indices: exact matches (by xxHash),
near-matches (by perceptual hash with D4 variants for rotation/flip invariance),
and semantic clusters (by cluster-based grouping). The available actions differ
by group.

**Exact duplicates** are unambiguous. An image appearing twice contributes
nothing new to either training or evaluation. In training data, exact duplicates
artificially inflate the effective weight of those samples during gradient
updates. In test data, they introduce data leakage if the duplicate also appears
in training. Remove all but one instance per exact-match group. The exception is
when exact duplicates appear with _different labels_ — that is a label error,
not a redundancy issue, and requires human review before any removal.

**Near-duplicates** require judgment. A perceptual hash match means two images
are visually similar but not pixel-identical — typically the same image with
different compression, slight crop, minor brightness adjustment, or rotation.
Whether to remove one depends on whether the variation is meaningful for your
task. A slightly brighter version of the same image adds little; the same scene
from a marginally different angle may add enough viewpoint diversity to keep. In
evaluation datasets, near-duplicates between train and test are the primary
concern — they degrade the validity of held-out metrics. Check the
near-duplicate groups for cross-split contamination before deciding on removals.

**Semantic clusters** (from cluster-based detection) identify images that are
different in appearance but close in embedding space — similar scene
composition, same background type, or same target in very similar
configurations. These represent redundancy at a higher level of abstraction.
High semantic redundancy in a region of the feature space means training
examples are concentrated there at the expense of underrepresented regions. This
is where {class}`.Prioritize` becomes relevant: rather than deciding which
images to delete, use prioritization to select a subset that preserves coverage
while reducing redundancy.

For large datasets with known high redundancy — full motion video, overhead
imagery with abundant background, repeated collection passes over the same area
— systematic deduplication before any other analysis is worth doing first. It
reduces the size of the problem that all subsequent evaluators have to process
and removes the noise that redundant samples introduce into coverage and bias
metrics. It also helps prevent leakage between dataset splits.

### Outliers

{class}`.Outliers` flags images with unusual image statistics (brightness, blur,
contrast, size) relative to the distribution of the dataset. The output includes
per-metric z-scores and which threshold each flagged image exceeded.

The first question with any outlier is: is this a data quality problem or a
legitimate but unusual sample? Low brightness outliers might be genuinely dark
operational images (keep) or corrupted/miscalibrated sensor captures (remove).
High blur might be motion blur in a class where motion blur matters for the task
(keep) or a focus failure (remove).

**Systematic outliers** — where many images from a particular collection event,
location, or sensor appear as outliers together — indicate a collection
condition that was not present in the rest of the dataset. Depending on whether
that condition represents a real operational scenario, the appropriate response
is either to collect more data under similar conditions (to make the unusual
condition normal) or to exclude the collection event (if it was a known
calibration or setup error).

**Isolated outliers** — individual unusual images not associated with a pattern
— are candidates for manual review. If the image is valid and represents a real
scenario, it may be worth keeping but worth flagging as a hard case. If it is a
processing artifact or error, remove it.

Outliers in evaluation data deserve extra scrutiny. An outlier in the test set
that is unlike anything in the training set will produce a confident wrong
prediction — not because the model is bad, but because it was never shown
anything like it. This is evidence for expanding the training distribution, not
evidence of model failure.

### Image statistics

{func}`.compute_stats` computes per-image and per-channel statistics (mean,
standard deviation, entropy, sharpness, and others), with {class}`.ImageStats`
flags controlling which statistics are calculated. These are diagnostic inputs
to outlier detection and dataset characterization rather than pass/fail outputs
on their own.

When the statistics reveal systematic distributional differences between
conditions (day vs. night, sensor A vs. sensor B, collection campaign 1 vs. 2),
the action is to make those differences visible in your metadata and analyze
them explicitly with {class}`.Balance` and {class}`.Diversity`. A dataset where
two-thirds of the bright images are from one class and two-thirds of the dark
images are from another has a confound between image brightness and class label.
That confound will not be visible in aggregate statistics — only in the
correlation structure that Balance measures.

---

### Label errors and label statistics

**Label error detection** ({func}`.label_errors`) flags samples whose embedding
geometry is inconsistent with their assigned label — specifically, samples that
are closer to a different class than to their own. The `error_rank` output is a
triage list: the highest-ranked samples are the strongest candidates for
mislabeling and should be reviewed first.

The output includes suggested replacement labels derived from rank-weighted
voting over the flagged sample's out-of-class nearest neighbors. Three outcomes
are possible for each flagged sample: a single confident suggestion (vote share
≥ 0.4 with a clear winner), an ambiguous pair (top two candidates within 0.2 of
each other), or no suggestion (neighborhood too noisy to recommend). No
suggestion does not mean the label is correct — it means the neighborhood is
mixed enough that a single replacement cannot be recommended with confidence.
Human review is required in all cases; the suggestion is a starting point, not a
verdict.

The `scores` array provides the raw intra/extra class distance ratio for every
sample, not just those above the threshold. Plotting the score distribution
often reveals whether flagged samples cluster sharply above 1.0 (clean dataset
with a few clear errors) or form a long tail (systematic mislabeling across a
class boundary).

**Label statistics** ({func}`.label_stats`) should be run before any other
analysis as a basic audit. The `empty_image_indices` field is the most
immediately actionable output: images with no annotations in an object detection
dataset are almost always a pipeline error, and training on them tells the model
to predict nothing. Verify whether empty images are intentional (background-only
images explicitly included as negatives) or accidental (annotation tool failure
or images dropped from the labeling queue). The `label_counts_per_class` and
`image_counts_per_class` outputs provide the input to class imbalance analysis
and are the prerequisite for {class}`.ClassBalance`.

---

## Dataset Bias and Coverage findings

### Balance and diversity

{class}`.Balance` measures mutual information (normalized to [0,1]) between
metadata factors and class labels, and between metadata factors themselves. A
high correlation between a metadata factor and class means the model can use
that factor as a shortcut to predict class without learning the actual signal of
interest.

The action depends on whether the correlation is _removable_. Some correlations
are collection artifacts — if all images of class A were collected in summer and
all images of class B in winter, the seasonal metadata factor is spuriously
correlated with class. The fix is to collect more data that breaks the confound:
class A images in winter, class B images in summer. Other correlations are
intrinsic to the task — targets of a certain class genuinely tend to appear at
certain sizes or in certain contexts. Those correlations cannot be removed; they
can only be acknowledged and accounted for when interpreting model performance
on operational data.

High inter-factor correlation (factor-to-factor rather than factor-to-class)
indicates redundancy in the metadata itself — two factors are measuring the same
underlying variation. This matters when designing metadata collection for future
data: capturing highly correlated factors is wasted effort. It also matters for
interpreting Parity results, where correlated factors can confound each other.

{class}`.Diversity` measures evenness of sampling across factor values. A low
diversity score means sampling is concentrated in a few values — most images
have the same lighting condition, the same background type, or the same target
aspect. The action is data collection: identify which factor values are
underrepresented and target collection toward them. When collection is not
possible, the low diversity should be documented as a known limitation and
matched against expected operational conditions. If the operational data will
also have low diversity in that factor (it always will be daylight, the terrain
is always desert), the training imbalance may not matter. If the operational
distribution is more diverse than the training distribution, the imbalance is a
real risk.

### Parity

{class}`.Parity` (experimental) measures statistical dependence between metadata
factors and labels using bias-corrected Cramér's V with a G-test for
significance. It is the most rigorous of the three bias metrics, requiring
sufficient per-cell sample counts (minimum five per contingency cell) to produce
valid results.

The `insufficient_data` flag in the output identifies factor-label combinations
where the sample count requirement was not met. These combinations cannot be
assessed for parity — the absence of a flag does not mean parity exists, it
means the data was too sparse to test. This is itself actionable: it identifies
collection targets where the current dataset has gaps too large to characterize.

Where Parity flags significant dependence, the interpretation is the same as for
Balance: determine whether the dependence is a collection artifact (addressable
by rebalancing data) or an intrinsic property of the task (documentable as a
limitation).

### Coverage and completeness

{func}`.coverage` identifies regions of the embedding space with no reference
samples nearby — gaps in the feature space where the model will be operating
without having seen similar data during training. {func}`.completeness` measures
the effective dimensionality utilization of the embedding space: whether the
dataset spans the available dimensions or is concentrated in a low-dimensional
subspace.

**Coverage gaps** indicate specific underrepresented regions. The
`uncovered_indices` output identifies existing samples nearest to the gaps — the
samples at the frontier of current coverage. These are the samples most valuable
to use as seeds for targeted collection: find more images that look like them,
or find images that would be neighbors to them in embedding space.

Low **completeness** (effective dimensionality well below the total embedding
dimensions) means the dataset is not exploring the full representational space.
This can indicate that the dataset is too narrow in its variation — all images
from similar conditions, similar viewpoints, similar backgrounds. Increasing
diversity on the identified low-utilization dimensions requires understanding
what those dimensions correspond to in the embedding model, which is an
interpretation task requiring domain knowledge.

Both metrics are contingent on the quality of the embedding. Coverage gaps and
completeness scores are only as meaningful as the embedding model is
discriminative for your task. See the [Embeddings](Embeddings.md) concept page
for guidance on embedding selection.

---

## Prioritization: acting on redundancy and coverage together

{class}`.Prioritize` is the primary _remediation_ tool in DataEval — it does not
just diagnose, it returns a ranked list of sample indices you can act on
directly. Where the other evaluators tell you what is wrong, Prioritize tells
you which samples to include in a representative subset or which unlabeled
samples to collect and annotate next.

### What Prioritize ranks

Prioritize ranks samples by their position in embedding space, from prototypical
(dense, central, low distance to neighbors) to atypical (sparse, peripheral,
high distance to neighbors). Five ranking methods are available, all operating
on embeddings:

- **`knn`**: each sample's score is its mean distance to its $k$ nearest
  neighbors. Simple, fast, no hyperparameters beyond $k$. Default
  $k = \sqrt{n}$.
- **`kmeans_distance`**: distance to assigned k-means cluster centroid.
  Prototypical samples are close to their centroid; challenging or boundary
  samples are far.
- **`kmeans_complexity`**: uses the product of intra-cluster and inter-cluster
  distances to score clusters, then samples from each cluster proportionally.
  Designed to mitigate class imbalance without requiring labels.
- **`hdbscan_distance`**: distance to assigned HDBSCAN cluster centroid. HDBSCAN
  discovers cluster structure without requiring a predetermined cluster count,
  making it more appropriate for datasets with irregular or variable-density
  clusters.
- **`hdbscan_complexity`**: analogous to `kmeans_complexity` but using HDBSCAN
  clustering.

### Selecting a policy

The `PrioritizeOutput` returned by `evaluate()` stores the raw ranking and
computes the final index order lazily based on a _policy_. Changing policy is
cheap — it reorders the stored result without re-running the embedding or
ranking. Three policies are available:

**`difficulty`** returns samples in direct order of their difficulty score
(`easy_first` or `hard_first`). Easy-first (prototypical samples first) is
appropriate when the dataset is small and the model needs to learn basic
concepts before seeing hard cases. Hard-first (atypical samples first) tends to
produce better models than easy-first for moderate-sized datasets because hard
samples sample the decision boundary rather than the cluster core. Neither
policy consistently outperforms random decimation on clean, well-balanced image
datasets, however.

**`stratified`** bins the score range and draws samples uniformly from each bin,
producing a mixture of easy and hard samples rather than concentrating on one
end. Across DataEval's testing this was the most consistent policy for
downstream model improvement — it avoids oversampling the dense, redundant core
while not ignoring prototypical samples entirely.

**`class_balanced`** reorders to ensure equal representation across classes
while maintaining priority order within each class. This requires class labels
and is most valuable when the initial dataset has extreme class imbalance —
where simple difficulty-based pruning would itself worsen the imbalance.

The library's own recommendation: start with `stratified` and either `knn` or
`kmeans_distance`. Neither scoring method consistently outperformed the other
across datasets; both outperformed random decimation on several datasets.

### Reference-based prioritization for new data

A distinct and important use case is evaluating _new, unlabeled data_ against an
_existing labeled dataset_. Pass the existing labeled dataset as the `reference`
argument:

```python
prioritizer = Prioritize.knn(extractor, k=10, reference=labeled_data)
result = prioritizer.evaluate(new_unlabeled_data)
most_novel = result.hard_first().indices
```

In hard-first order, the top-ranked samples are those most unlike the reference
— the candidates that would add the most new information to the dataset. This is
the labeling budget allocation problem: given limited annotation resources,
which unlabeled samples are worth labeling? Prioritize with a reference answers
that question directly.

### What Prioritize does not do

Prioritize does not remove samples — it ranks them. The decision about where to
draw the cutoff (how many samples to include in the prioritized subset) is
yours, and it depends on your computational budget, target model performance,
and downstream evaluation requirements. Prioritize gives you a principled
ordering; the threshold is a program decision.

Prioritize also does not account for label quality, metadata correctness, or
class distribution in its scoring (unless `class_balanced` policy is used). A
prioritized subset of a dataset with label errors will still contain label
errors. Run {class}`.Duplicates` and {class}`.Outliers` before prioritizing to
clean the pool being prioritized.

---

## Class balance remediation

{class}`.ClassBalance` is a _selection_ tool, not a diagnostic. It takes a
dataset with class frequency information and returns indices for a
class-balanced subset, using one of two strategies:

**`global`** samples with probability proportional to the inverse square root of
class frequency: $w_c = \max(1, \sqrt{\alpha / f_c})$, where $f_c$ is the
frequency of class $c$ and $\alpha$ is a scaling factor. This upweights rare
classes without completely equalizing them — it produces a gradient from common
to rare rather than forcing uniform counts.

**`interclass`** samples equal numbers from each class, completely equalizing
class frequency. Background class is excluded from the count and not drawn from.

Use `global` when you want to reduce imbalance while preserving some of the
original frequency information. Use `interclass` when you need strict balance
for evaluation purposes or when extreme imbalance is creating a known
performance problem.

`ClassBalance` addresses _sampling_ imbalance — it helps you select a balanced
subset from what you have. If the imbalance exists because certain classes are
simply underrepresented in the dataset (not enough images of that class exist),
`ClassBalance` cannot fix that. The only remedy for collection imbalance is
collection: acquiring more data for the underrepresented classes.

---

## Performance Limits findings

### BER upper and lower bounds

{func}`.ber_mst` and {func}`.ber_knn` returns upper and lower bounds on the
Bayes Error Rate — the irreducible classification error given the current
feature representation. The upper bound is the actionable number: if the BER
upper bound exceeds your operational accuracy requirement, no classifier trained
on this data in this feature space will meet the requirement.

**The BER upper bound exceeds the requirement.** This does not mean the task is
impossible — it means it is impossible with the current data and feature
representation. Three levers are available:

1. **Better embeddings**: if the current feature representation conflates
   classes that should be separable, a better embedding model may achieve lower
   BER. BER is always contingent on the embedding; recomputing with different
   embeddings is a cheap way to check whether the representation is the
   bottleneck.

2. **More discriminative data**: if the class overlap is genuine in the current
   feature space because the data does not contain the signal needed to separate
   the classes, additional data of the same type will not help. Different data —
   different sensors, different modalities, different collection conditions —
   may reduce the irreducible overlap.

3. **Task redefinition**: if two classes are genuinely indistinguishable in the
   available feature space under real operational conditions, the task taxonomy
   may need revision. This is a program-level decision.

**The BER upper and lower bounds are far apart.** A wide gap means the estimate
is uncertain — the available data is insufficient to tightly characterize the
irreducible error. Collect more data, especially in regions of the feature space
where class overlap appears high (near decision boundaries), and recompute.

**BER applies to classification only.** Object detection users: BER and
{func}`.uap` measure classification feasibility without localization error. A
favorable BER result does not guarantee that a detection model will meet its
requirements — localization error is an additional, separate constraint.

### UAP

{func}`.uap` computes the Upper-bound Average Precision over bounding box crops,
treating the problem as classification and removing localization error. The
result is a strict upper bound on what any detector can achieve in mAP: your
operational detector must also solve localization, so actual mAP will be lower.

A UAP that already fails to meet the requirement means the classification
problem itself is infeasible — even with perfect localization, the requirement
cannot be met. A UAP that passes gives you headroom to lose to localization
error. The gap between UAP and your requirement is the tolerance budget for
localization and detection-specific factors.

### Sufficiency learning curves

{class}`.Sufficiency` fits a power law $f(n) = c \cdot n^{-m} + c_0$ to your
model's learning curve. The three parameters tell distinct stories:

**$c_0$ (asymptote)** is the most important parameter. It is the performance
ceiling: what the model converges to as $n \to \infty$. If $c_0$ is below your
operational requirement, collecting more data will not close the gap — the model
has hit its ceiling under the current data distribution and architecture. The
appropriate response is a change in the data (different conditions, different
sensors, different label taxonomy) or a change in the model.

**$m$ (exponent)** governs how quickly performance improves with additional
data. A steep curve (large $m$) means you are currently on the steep part of the
learning curve and modest data additions will produce measurable gains. A
shallow curve (small $m$) means you are in the diminishing returns regime.

**`inv_project(target_performance)`** returns the number of samples estimated to
be required to reach a target performance level. When this returns -1, the
target exceeds $c_0$ and is unachievable under the current conditions regardless
of data volume. When it returns a finite number, the result is an estimate with
confidence intervals based on how well the power law fits the observed data. The
estimate assumes the learning curve continues along the same trajectory —
distribution shift, label errors, or changes in model architecture will all
violate that assumption.

---

## Distribution Shift findings

### Drift detection results

All drift detectors return a binary `drifted` flag and a p-value (or AUROC for
the domain classifier). A drift detection is a signal to investigate, not a
signal to immediately retrain.

**First response to drift detection**: characterize the drift before acting.
{class}`.DriftUnivariate` identifies _which_ features drifted.
{class}`.DriftDomainClassifier` provides feature importances indicating _which_
features most distinguish the operational batch from the reference. Use this
information to understand what changed: a seasonal shift in lighting, a sensor
hardware change, a change in the operational scenario, or a change in the target
population.

**Drift that affects model performance** (visible in uncertainty-based drift
detection or in ground-truth performance metrics when labels are available)
warrants retraining or fine-tuning on operational data. If operational data with
ground truth is available, it should be incorporated into the training set.
Prioritize the most novel operational samples for labeling using
{class}`.Prioritize` with the current training set as reference — the hard-first
ranking identifies the operational samples most unlike the training
distribution.

**Drift that does not affect model performance** (feature distribution changed
but uncertainty-based detection shows no confidence degradation) may not require
retraining. Document the drift and increase monitoring frequency.

**Persistent structural drift** — where the operational distribution has
systematically departed from the training distribution and is not expected to
return — is a dataset problem, not a monitoring problem. The training reference
needs to be updated to reflect the operational distribution. This is a program
lifecycle decision involving data collection, labeling, and model qualification.

### OOD detection results

OOD detectors return per-sample scores and a binary flag. The actionable
question is not just whether a sample is flagged, but _why_ it is distant from
the training distribution.

**Isolated OOD samples in a batch** that drift detection did not flag (the batch
as a whole is not shifted, but individual samples are anomalous) are candidates
for human review. They may represent: legitimate novel scenarios the model has
never seen (worth collecting more of), sensor or collection errors (remove), or
genuinely rare events that are operationally significant (flag for model
attention, consider oversampling in retraining).

**High OOD rate in a batch** that coincides with drift detection signals that
the distribution has moved into a region the training data did not cover. The
combination of drift detection and OOD scoring provides a more complete picture
than either alone: drift detection says the population has shifted; high OOD
rate says the new distribution is in a sparse region of the training manifold.
This is the scenario most likely to produce confident wrong predictions and most
likely to warrant immediate retraining.

**OOD detection with reconstruction-based methods** provides feature-level error
maps that localize _where_ in the image the anomaly is. When per-sample OOD
flags are reviewed, use the error maps to understand whether the anomaly is a
foreground change (new target type, new object configuration) or a background
change (new environmental condition). The former is usually more consequential
for model performance.

---

## Diagnosing findings with metadata

When any DataEval evaluator flags samples — whether as outliers, OOD, drifted,
or mislabeled — the immediate question is _why_. The flagged indices are a list
of samples; the metadata attached to those samples is often the fastest path to
understanding what they have in common.

DataEval provides two metadata diagnostic functions for this purpose:
{func}`.factor_deviation` and {func}`.factor_predictors`. Both take flagged
sample indices as input and operate on the metadata factors (image-level or
object-level attributes) that were collected alongside the samples.

### Identifying what is unusual about flagged samples

{func}`.factor_deviation` answers the question: _for each flagged sample, which
metadata factors deviate most from the reference distribution?_

It computes the reference median for each factor, then measures how far each
flagged sample deviates from that median. Deviations are scaled asymmetrically:
positive deviations are normalized by the median of the positive half of the
reference distribution; negative deviations by the median of the negative half.
This prevents factors with wide positive tails from dominating factors with wide
negative tails.

The output is one dictionary per flagged sample, mapping factor names to their
scaled deviation magnitudes, sorted in descending order. The top-ranked factor
for a given sample is the metadata dimension that most strongly distinguishes
that sample from the reference.

A practical use: after {class}`.Outliers` flags a set of images, run
`factor_deviation` on the flagged indices against the reference metadata from
the full dataset. If the top factor for most flagged samples is `altitude`, that
suggests the outliers were collected at unusual altitudes — actionable
information for both data collection planning and model limitation
documentation.

At least three reference samples are required for a meaningful deviation
calculation. The function returns empty dictionaries if fewer reference samples
are available.

### Identifying which factors predict flagged samples

{func}`.factor_predictors` answers the question: _which metadata factors are
most associated with being flagged?_

It treats the flagged/not-flagged binary as a target variable and computes the
normalized mutual information between each metadata factor and that target,
using scikit-learn's `mutual_info_classif`. The result is a dictionary mapping
each factor name to its normalized mutual information. A score of 0 means the
factor carries no information about which samples were flagged; a score near 1
means the factor is predictive of flagging status.

Unlike `factor_deviation`, which characterizes individual flagged samples,
`factor_predictors` characterizes the flagged set as a whole. It answers: "Is
there a systematic metadata reason why these samples were flagged?" This is most
useful when there are many flagged samples and you want to understand the
population-level pattern rather than inspect each sample individually.

**Example workflow after drift detection:**

```python
# drift_result.drifted is True; flagged_indices are samples in the drifted batch
flagged = [i for i, score in enumerate(ood_scores) if score > threshold]

# Which factors predict OOD membership?
predictors = factor_predictors(operational_metadata.factors, flagged)
# {'time_of_day': 0.84, 'sensor_id': 0.61, 'altitude': 0.02, ...}
# → time_of_day and sensor_id are the likely drivers

# For the top flagged samples, what specifically is different?
top_flagged = flagged[:20]
deviations = factor_deviation(reference_metadata.factors, operational_metadata.factors, top_flagged)
# deviations[0] → {'time_of_day': 4.2, 'sensor_id': 1.8, 'altitude': 0.3}
# → first flagged sample is 4.2 scaled deviations from reference in time_of_day
```

**Mutual information is association, not causation.** A high normalized MI (NMI)
score means the factor correlates with being flagged. It does not mean the
factor caused the problem, nor does it mean other factors with lower scores are
irrelevant. Always interpret NMI scores alongside domain knowledge about what
the factors represent and how data was collected.

Both functions require factors to be provided as dictionaries mapping factor
names to arrays. If your metadata is stored in DataEval's {class}`.Metadata`
class, the `.factors` attribute provides this format directly.

## Divergence scores

Where drift detectors return a binary signal — drifted or not — HP divergence
({func}`.divergence`) returns a continuous score between 0 and 1. The two are
complementary: drift detection tells you when to act; divergence tells you _how
much_ the situation has changed, which is the information needed to calibrate
the urgency and scale of the response.

**A divergence score near 0** between training and operational data confirms
that the model is operating close to its training envelope. Formal drift
detection results should be interpreted in this context: a statistically
significant drift result with low divergence means a real but small shift —
worth documenting and monitoring, but unlikely to require immediate retraining.

**A divergence score approaching 1** means the two distributions are nearly
separable — the operational data looks fundamentally different from anything in
training. Any model performance measurements from training are likely to be poor
predictors of operational performance. This is a strong signal for immediate
data collection, retraining, or escalation depending on the operational stakes.

**Tracking divergence over time** is more informative than any single score. A
slowly rising trend — each operational batch slightly more diverged from the
reference than the last — identifies gradual drift that may never trigger a
single-batch hypothesis test but is nonetheless eroding the relevance of the
training distribution. If you are running drift monitoring in production,
logging divergence scores alongside p-values gives you the trend visibility that
p-values alone cannot provide.

**Using divergence to localize drift.** When a drift detector fires on a large
batch, run divergence on subsets — by collection date, sensor, geographic
region, or operational condition — to identify which subpopulation is driving
the shift. The subset with the highest divergence against the reference is where
to focus investigation and targeted data collection. This avoids the
multiple-testing penalty of running separate drift tests on each subset.

**Pre-deployment gap assessment.** Before a model enters service, run divergence
between the training set and a sample of expected operational data. A high
divergence at this stage — before any monitoring has begun — is evidence that
the training distribution was not representative of the deployment environment.
This is the point at which the gap is cheapest to close: collect more data from
the operational distribution, or document the limitation explicitly in the test
report.

## Keeping results in context

DataEval's diagnostics are snapshots. They characterize the dataset at the time
of analysis. Datasets change — new data is collected, labels are corrected,
splits are revised — and monitoring data changes continuously. Any diagnostic
result that drives a significant decision should be recomputed after the
relevant changes have been made to verify that the action had the expected
effect.

The most important integration is the connection between pre-deployment and
operational findings. BER and UAP results from pre-deployment data evaluation
establish expected performance bounds. Sufficiency curves establish expected
trajectories. When operational monitoring detects drift or OOD conditions, those
pre-deployment baselines provide the context for interpreting whether the
operational change is within the expected envelope or represents a genuine
threat to the established performance bounds.

## Related concept pages

- [Data Integrity](DataIntegrity.md) — what the cleaning, label error, and label
  statistics diagnostics measure
- [Clustering](Clustering.md) — the algorithm underlying cluster-based
  Duplicates, Outliers, label_errors, and Prioritize
- [Dataset Bias and Coverage](DatasetBias.md) — what the bias diagnostics
  measure
- [Performance Limits](PerformanceLimits.md) — what BER, UAP, and Sufficiency
  tell you about achievability
- [Distribution Shift](DistributionShift.md) — what drift and OOD detection
  measure and how detectors work
- [Divergence](Divergence.md) — the quantitative distance metric covered in the
  divergence scores section above
- [Embeddings](Embeddings.md) — the representation that most DataEval evaluators
  depend on

## See this in practice

### How-to guides

- [How to deduplicate a dataset](../notebooks/h2_deduplicate.py)
- [How to cluster and analyze embeddings](../notebooks/h2_cluster_analysis.py)
- [How to measure IC feasibility](../notebooks/h2_measure_ic_feasibility.py)
- [How to measure IC sufficiency](../notebooks/h2_measure_ic_sufficiency.py)
- [How to measure divergence](../notebooks/h2_measure_divergence.py)
- [How to detect undersampling](../notebooks/h2_detect_undersampling.py)

### Tutorials

- [Data cleaning tutorial](../notebooks/tt_clean_dataset.py) — finding and
  handling duplicates and outliers
- [Identify bias and correlations tutorial](../notebooks/tt_identify_bias.py) —
  acting on balance and diversity findings
- [Assessing the data space tutorial](../notebooks/tt_assess_data_space.py) —
  coverage gaps and embedding-space decisions
- [Monitoring distribution shift tutorial](../notebooks/tt_monitor_shift.py) —
  responding to drift detection
- [Identifying OOD samples tutorial](../notebooks/tt_identify_ood_samples.py) —
  responding to OOD detection
